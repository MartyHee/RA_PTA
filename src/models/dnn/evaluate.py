"""DNN 评估主程序 — 支持 train/val/test 三路切分"""

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
import torch.nn as nn
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
from models.dnn.dataset import DNNDataProcessor, TabularDataset  # noqa: E402
from models.dnn.model import DNNModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.io import read_csv_safe  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("dnn_eval")


def find_latest_run(output_root: Path) -> str | None:
    """在 output_root 中查找最近一次 run 的 run_id。"""
    latest_file = output_root / "latest_run.txt"
    if latest_file.exists():
        return latest_file.read_text().strip()
    runs = sorted(
        [d.name for d in output_root.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    return runs[-1] if runs else None


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: float,
) -> dict:
    """评估模型，返回所有结果。"""
    model.eval()
    all_logits: list[float] = []
    all_scores: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            numeric_b = batch["numeric"].to(device)
            cat_b = batch["categorical"].to(device)
            labels_b = batch["label"].to(device)

            logits = model(numeric_b, cat_b)
            scores = torch.sigmoid(logits)

            all_logits.extend(logits.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())

    all_logits_arr = np.array(all_logits)
    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = (all_scores_arr >= threshold).astype(int)

    eval_loss = criterion(
        torch.tensor(all_logits_arr), torch.tensor(all_labels_arr)
    ).item()

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

    n_pos = int(all_labels_arr.sum())
    n_neg = int(len(all_labels_arr) - n_pos)

    return {
        "logits": all_logits_arr,
        "scores": all_scores_arr,
        "preds": all_preds_arr,
        "labels": all_labels_arr,
        "eval_loss": eval_loss,
        "cls_metrics": cls_metrics,
        "pk_metrics": pk_metrics,
        "rk_metrics": rk_metrics,
        "warnings": cls_warnings + pk_warnings + rk_warnings,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN 评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dnn/dnn_base.yaml",
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
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="评估指定 split (val/test)，默认全部",
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

    run_id_final = run_dir.name
    logger.info(f"评估目录: {run_dir}")
    logger.info(f"Run ID: {run_id_final}")

    # ── 3. 加载特征配置 ─────────────────────────────────────
    feature_config_path = run_dir / "feature_config_used.json"
    if not feature_config_path.exists():
        logger.error(f"特征配置文件不存在: {feature_config_path}")
        sys.exit(1)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    dataset_name = feature_config.get("dataset_name", "sample0427")

    # ── 4. 恢复处理器 ───────────────────────────────────────
    processor = DNNDataProcessor.from_config(feature_config)

    # ── 5. 确定评估数据路径 ─────────────────────────────────
    is_three_way = "test_data_path" in config

    splits_to_eval: list[str] = []
    split_paths: dict[str, Path] = {}

    if args.split:
        splits_to_eval = [args.split]
    elif is_three_way:
        splits_to_eval = ["val", "test"]
    else:
        splits_to_eval = ["eval"]

    if is_three_way:
        if "val" in splits_to_eval or not args.split:
            split_paths["val"] = project_root / config["val_data_path"]
        if "test" in splits_to_eval or not args.split:
            split_paths["test"] = project_root / config["test_data_path"]
    else:
        split_paths["eval"] = project_root / config["eval_data_path"]

    # ── 6. 加载模型 ─────────────────────────────────────────
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cat_embed_dims = processor.cat_embed_dims
    model = DNNModel(
        numeric_dim=feature_config.get("numeric_cols", []).__len__(),
        cat_embed_dims=cat_embed_dims,
        hidden_units=config.get("hidden_units", [64, 32]),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载完成: {model_path}")

    # ── 7. 对每个 split 评估 ────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    threshold = config.get("threshold", 0.5)

    all_metrics: dict = {}

    for split_name, data_path in split_paths.items():
        logger.info(f"评估 split: {split_name} ({data_path})")

        df, _ = read_csv_safe(str(data_path))
        logger.info(f"  {split_name} 样本数: {len(df)}")

        data = processor.transform(df)
        dataset = TabularDataset(
            data["numeric"], data["categorical"], data["labels"]
        )
        loader = DataLoader(
            dataset, batch_size=config.get("batch_size", 64), shuffle=False
        )

        result = evaluate(model, loader, device, criterion, threshold)

        # 保存 predictions
        pred_df = pd.DataFrame(
            {
                "label": result["labels"],
                "score": result["scores"],
                "pred": result["preds"],
                "split": split_name,
                "model_name": "dnn",
                "dataset_name": dataset_name,
                "run_id": run_id_final,
            }
        )
        ids_data = data.get("ids")
        if ids_data is not None:
            ids_df = ids_data.reset_index(drop=True)
            pred_df = pd.concat([ids_df, pred_df], axis=1)

        pred_path = run_dir / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"  predictions 已保存: {pred_path} ({len(pred_df)} 条)")

        # 构建 metrics
        label_definition = feature_config.get(
            "label_definition",
            "interaction_score >= threshold (离线实验伪标签)",
        )
        metrics = {
            "model_name": "dnn",
            "dataset_name": dataset_name,
            "run_id": run_id_final,
            "split": split_name,
            "sample_count": len(result["labels"]),
            "positive_count": result["n_pos"],
            "negative_count": result["n_neg"],
            "eval_loss": result["eval_loss"],
            "auc": result["cls_metrics"].get("auc"),
            "accuracy": result["cls_metrics"].get("accuracy"),
            "precision": result["cls_metrics"].get("precision"),
            "recall": result["cls_metrics"].get("recall"),
            "f1": result["cls_metrics"].get("f1"),
            "precision_at_k": result["pk_metrics"],
            "recall_at_k": result["rk_metrics"],
            "threshold": threshold,
            "label_definition": label_definition,
            "warnings": result["warnings"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        all_metrics[f"{split_name}_metrics"] = metrics

        logger.info(
            f"  {split_name}: AUC={metrics['auc']}, "
            f"F1={metrics['f1']}, Loss={metrics['eval_loss']:.6f}"
        )

    # ── 8. 保存更新后的 metrics ─────────────────────────────
    # 保留原有内容并添加新评估结果
    existing_metrics_path = run_dir / "metrics.json"
    if existing_metrics_path.exists():
        with open(existing_metrics_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(all_metrics)
        existing["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        existing = all_metrics
        existing["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(existing_metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {existing_metrics_path}")

    # ── 9. 打印摘要 ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("评估完成！")
    for split_name in split_paths:
        m = all_metrics.get(f"{split_name}_metrics", {})
        logger.info(
            f"  {split_name}: "
            f"AUC={m.get('auc')}, F1={m.get('f1')}, "
            f"样本={m.get('sample_count')}, "
            f"正例={m.get('positive_count')}, "
            f"负例={m.get('negative_count')}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()