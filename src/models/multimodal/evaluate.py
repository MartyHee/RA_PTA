"""多模态模型评估主程序"""

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

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from evaluation.metrics import compute_classification_metrics  # noqa: E402
from evaluation.ranking_metrics import (  # noqa: E402
    compute_precision_at_k,
    compute_recall_at_k,
)
from models.multimodal.dataset import MultimodalDataset  # noqa: E402
from models.multimodal.fusion_model import MultimodalFusionModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("multimodal_eval")


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
    parser = argparse.ArgumentParser(description="多模态模型评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal/multimodal_base.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="指定 run_id（默认使用最新一次 run）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────
    config = load_config(args.config)
    project_root = Path(_project_root)
    output_root = project_root / config["output_root"]
    eval_npz_path = project_root / config["eval_npz_path"]
    feature_info_path = project_root / config["feature_info_path"]
    metrics_config_path = project_root / config["metrics_config_path"]

    # ── 2. 确定输出目录 ─────────────────────────────────────
    if args.run_id:
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
    run_id = run_dir.name

    # ── 3. 加载 feature_info ────────────────────────────────
    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    text_dim = feature_info.get("text_dim", 32)
    visual_dim = feature_info.get("visual_dim", 10)
    structured_dim = feature_info.get("structured_dim", 20)

    # ── 4. 加载 feature_config_used ─────────────────────────
    feature_config_path = run_dir / "feature_config_used.json"
    if feature_config_path.exists():
        with open(feature_config_path, "r", encoding="utf-8") as f:
            feature_config = json.load(f)
        logger.info(f"特征配置已加载: {feature_config_path}")
    else:
        feature_config = {}

    # ── 5. 加载 metrics config ──────────────────────────────
    from utils.config import load_config as load_yaml
    metrics_config = load_yaml(metrics_config_path)
    k_values = metrics_config.get("k_values", [5, 10, 20])

    # ── 6. 加载评估数据 ─────────────────────────────────────
    eval_dataset = MultimodalDataset(eval_npz_path, feature_info)
    for warn in eval_dataset.warnings:
        logger.warning(f"评估集: {warn}")
    logger.info(f"评估样本数: {len(eval_dataset)}")

    # ── 7. 加载模型 ─────────────────────────────────────────
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"

    model = MultimodalFusionModel(
        text_dim=text_dim,
        visual_dim=visual_dim,
        structured_dim=structured_dim,
        text_hidden_dim=config.get("text_hidden_dim", 32),
        visual_hidden_dim=config.get("visual_hidden_dim", 16),
        structured_hidden_dim=config.get("structured_hidden_dim", 32),
        fusion_hidden_dim=config.get("fusion_hidden_dim", 64),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载完成: {model_path}")

    # ── 8. 推理 ─────────────────────────────────────────────
    threshold = config.get("threshold", 0.5)
    all_scores: list[float] = []
    all_labels: list[float] = []
    all_sample_ids: list[int] = []
    all_video_ids: list[int] = []
    all_author_ids: list[str] = []

    with torch.no_grad():
        for i in range(len(eval_dataset)):
            item = eval_dataset[i]
            text_t = item["text"].unsqueeze(0).to(device)
            visual_t = item["visual"].unsqueeze(0).to(device)
            struct_t = item["structured"].unsqueeze(0).to(device)

            logit = model(text_t, visual_t, struct_t)
            score = torch.sigmoid(logit)

            all_scores.append(score.cpu().item())
            all_labels.append(item["label"].item())
            all_sample_ids.append(int(item["sample_id"]))
            all_video_ids.append(int(item["video_id"]))
            all_author_ids.append(str(item["author_id"]))

    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = (all_scores_arr >= threshold).astype(int)

    # ── 9. 计算指标 ─────────────────────────────────────────
    cls_metrics, cls_warnings = compute_classification_metrics(
        all_labels_arr, all_scores_arr, all_preds_arr, threshold
    )

    pk_metrics, pk_warnings = compute_precision_at_k(
        all_labels_arr, all_scores_arr, k_values
    )
    rk_metrics, rk_warnings = compute_recall_at_k(
        all_labels_arr, all_scores_arr, k_values
    )

    all_warnings = cls_warnings + pk_warnings + rk_warnings
    n_pos = int(all_labels_arr.sum())
    n_neg = int(len(all_labels_arr) - n_pos)

    # ── 10. 组装 metrics ───────────────────────────────────
    metrics = {
        "model_name": "multimodal",
        "run_id": run_id,
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
        "eval_loss": None,
        "threshold": threshold,
        "label_definition": feature_info.get(
            "label_definition", "流程验证伪标签: interaction_score >= threshold"
        ),
        "feature_shapes": {
            "text_dim": text_dim,
            "visual_dim": visual_dim,
            "structured_dim": structured_dim,
        },
        "no_image_download_confirmed": config.get("no_image_download", True),
        "no_external_api_confirmed": config.get("no_external_api", True),
        "no_large_pretrained_model_confirmed": config.get(
            "no_large_pretrained_model", True
        ),
        "warnings": all_warnings,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "当前指标仅基于 sample0427 样本数据计算，仅用于流程级验证。",
            "不表示正式推荐系统效果结论。",
            "标签为 interaction_score 伪标签，不代表真实曝光/点击/转化目标。",
            "visual_features 仅包含媒体元信息，不包含图像语义特征。",
            "未下载图片。",
            "未调用外部 API。",
            "未使用大型预训练模型。",
        ],
    }

    # ── 11. 保存 metrics.json ──────────────────────────────
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    # ── 12. 保存 predictions.csv ───────────────────────────
    pred_df = pd.DataFrame(
        {
            "sample_id": all_sample_ids,
            "video_id": all_video_ids,
            "author_id": all_author_ids,
            "label": all_labels_arr,
            "score": all_scores_arr,
            "pred": all_preds_arr,
            "split": "eval",
            "model_name": "multimodal",
            "run_id": run_id,
        }
    )
    pred_path = run_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"预测结果已保存: {pred_path}")

    # ── 13. 打印指标摘要 ──────────────────────────────────
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