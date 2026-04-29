"""Wide & Deep 评估主程序"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from evaluation.metrics import compute_classification_metrics  # noqa: E402
from evaluation.ranking_metrics import (  # noqa: E402
    compute_precision_at_k,
    compute_recall_at_k,
)
from models.wide_deep.dataset import WideDeepDataProcessor, WideDeepDataset  # noqa: E402
from models.wide_deep.model import WideDeepModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.io import read_csv_safe  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("wide_deep_eval")


def find_latest_run(output_root: Path) -> str | None:
    """在 output_root 中查找最近一次 run 的 run_id。"""
    latest_file = output_root / "latest_run.txt"
    if latest_file.exists():
        return latest_file.read_text().strip()
    runs = sorted(
        [d.name for d in output_root.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Wide & Deep 评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wide_deep/wide_deep_base.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="指定 run_id（默认使用最新一次 run）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="直接指定输出目录（优先级高于 run_id）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────
    config = load_config(args.config)
    project_root = Path(_project_root)
    output_root = project_root / config["output_root"]

    # ── 2. 确定输出目录 ─────────────────────────────────────
    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif args.run_id:
        run_dir = output_root / args.run_id
    else:
        run_id = find_latest_run(output_root)
        if run_id is None:
            logger.error("未找到任何 run，请先运行 train.py 或指定 --run_id")
            sys.exit(1)
        run_dir = output_root / run_id
        logger.info(f"使用最新 run: {run_id}")

    if not run_dir.exists():
        logger.error(f"输出目录不存在: {run_dir}")
        sys.exit(1)

    logger.info(f"评估目录: {run_dir}")

    # ── 3. 加载特征配置 ─────────────────────────────────────
    feature_config_path = run_dir / "feature_config_used.json"
    if not feature_config_path.exists():
        logger.error(f"特征配置文件不存在: {feature_config_path}")
        sys.exit(1)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    # ── 4. 恢复处理器 ───────────────────────────────────────
    processor = WideDeepDataProcessor.from_config(feature_config)

    # ── 5. 加载数据 ─────────────────────────────────────────
    eval_path = project_root / config["eval_data_path"]
    eval_df, _ = read_csv_safe(str(eval_path))
    logger.info(f"评估样本数: {len(eval_df)}")

    eval_data = processor.transform(eval_df)
    eval_dataset = WideDeepDataset(
        eval_data["numeric"],
        eval_data["categorical"],
        eval_data["wide"],
        eval_data["labels"],
    )
    batch_size = config.get("batch_size", 64)
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ── 6. 加载模型 ─────────────────────────────────────────
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = WideDeepModel(
        numeric_dim=feature_config.get("numeric_cols", []).__len__(),
        cat_embed_dims=processor.cat_embed_dims,
        wide_vocab_sizes=processor.wide_vocab_sizes,
        deep_hidden_units=config.get("deep_hidden_units", [64, 32]),
        dropout=config.get("dropout", 0.3),
        wide_embedding_dim=config.get("wide_embedding_dim", 1),
    ).to(device)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载完成: {model_path}")

    # ── 7. 推理 ─────────────────────────────────────────────
    all_scores: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in eval_loader:
            numeric_b = batch["numeric"].to(device)
            cat_b = batch["categorical"].to(device)
            wide_b = batch["wide"].to(device)
            logits = model(numeric_b, cat_b, wide_b)
            scores = torch.sigmoid(logits)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)
    threshold = config.get("threshold", 0.5)
    all_preds_arr = (all_scores_arr >= threshold).astype(int)

    # ── 8. 计算指标 ─────────────────────────────────────────
    cls_metrics, cls_warnings = compute_classification_metrics(
        all_labels_arr, all_scores_arr, all_preds_arr, threshold
    )

    k_values = [5, 10, 20]
    pk_metrics, pk_warnings = compute_precision_at_k(
        all_labels_arr, all_scores_arr, k_values
    )
    rk_metrics, rk_warnings = compute_recall_at_k(
        all_labels_arr, all_scores_arr, k_values
    )

    all_warnings = cls_warnings + pk_warnings + rk_warnings
    n_pos = int(all_labels_arr.sum())
    n_neg = int(len(all_labels_arr) - n_pos)

    metrics = {
        "model_name": "wide_deep",
        "split": "eval",
        "sample_count": len(all_labels_arr),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "auc": cls_metrics.get("auc"),
        "accuracy": cls_metrics.get("accuracy"),
        "precision": cls_metrics.get("precision"),
        "recall": cls_metrics.get("recall"),
        "f1": cls_metrics.get("f1"),
        "precision_at_k": pk_metrics,
        "recall_at_k": rk_metrics,
        "threshold": threshold,
        "label_definition": "流程验证伪标签: interaction_score >= threshold",
        "warnings": all_warnings,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "当前指标仅基于 sample0427 样本数据计算，仅用于流程级验证。",
            "不表示正式推荐系统效果结论。",
            "标签为 interaction_score 伪标签，不代表真实曝光/点击/转化目标。",
            "当前仅有 eval 评估，无独立 test 集。",
        ],
    }

    # ── 9. 保存 metrics.json ────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    # ── 10. 保存 predictions.csv ────────────────────────────
    ids_df = eval_data["ids"]  # 包含 id_cols
    pred_df = pd.DataFrame(
        {
            "label": all_labels_arr,
            "score": all_scores_arr,
            "pred": all_preds_arr,
            "split": "eval",
            "model_name": "wide_deep",
        }
    )
    if ids_df is not None:
        ids_df.reset_index(drop=True, inplace=True)
        pred_df = pd.concat([ids_df.reset_index(drop=True), pred_df], axis=1)

    pred_path = run_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"预测结果已保存: {pred_path}")

    # ── 11. 打印指标摘要 ────────────────────────────────────
    logger.info("=== 评估结果 ===")
    logger.info(f"样本数: {metrics['sample_count']} (正例: {n_pos}, 负例: {n_neg})")
    logger.info(f"AUC: {metrics['auc']}")
    logger.info(f"Accuracy: {metrics['accuracy']}")
    logger.info(f"Precision: {metrics['precision']}")
    logger.info(f"Recall: {metrics['recall']}")
    logger.info(f"F1: {metrics['f1']}")
    for k in k_values:
        pk = pk_metrics.get(f"precision_at_{k}", "N/A")
        rk = rk_metrics.get(f"recall_at_{k}", "N/A")
        logger.info(f"Precision@{k}: {pk}, Recall@{k}: {rk}")
    if all_warnings:
        logger.warning(f"Warnings: {all_warnings}")


if __name__ == "__main__":
    main()