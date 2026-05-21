"""多模态模型评估主程序 — 支持 val/test 双路评估（三路切分模式）"""

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


def evaluate_model(
    model: nn.Module,
    dataset: MultimodalDataset,
    device: str,
    criterion: nn.Module,
    threshold: float,
) -> dict:
    """评估模型，返回所有结果。"""
    model.eval()
    all_logits: list[float] = []
    all_scores: list[float] = []
    all_labels: list[float] = []
    all_sample_ids: list[int] = []
    all_video_ids: list[int] = []
    all_author_ids: list[str] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            text_t = item["text"].unsqueeze(0).to(device)
            visual_t = item["visual"].unsqueeze(0).to(device)
            struct_t = item["structured"].unsqueeze(0).to(device)
            cat_t = item.get("categorical")
            if cat_t is not None:
                cat_t = cat_t.unsqueeze(0).to(device)

            logit = model(text_t, visual_t, struct_t, cat_t)
            score = torch.sigmoid(logit)

            all_logits.append(logit.cpu().item())
            all_scores.append(score.cpu().item())
            all_labels.append(item["label"].item())
            all_sample_ids.append(int(item["sample_id"]))
            all_video_ids.append(int(item["video_id"]))
            all_author_ids.append(str(item["author_id"]))

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

    k_values = [5, 10, 20, 50]
    pk_metrics, pk_warnings = compute_precision_at_k(
        all_labels_arr, all_scores_arr, k_values
    )
    rk_metrics, rk_warnings = compute_recall_at_k(
        all_labels_arr, all_scores_arr, k_values
    )

    n_pos = int(all_labels_arr.sum())
    n_neg = int(len(all_labels_arr) - n_pos)

    return {
        "sample_ids": all_sample_ids,
        "video_ids": all_video_ids,
        "author_ids": all_author_ids,
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
    feature_info_path = project_root / config["feature_info_path"]
    metrics_config_path = project_root / config["metrics_config_path"]
    dataset_name = config.get("dataset_name", "sample0427")
    dataset_variant = config.get("dataset_variant", "")
    model_name = config.get("model_name", "multimodal")

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
    label_definition = feature_info.get(
        "label_definition",
        "interaction_score >= threshold (离线实验伪标签)",
    )

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

    # ── 5a. 从 feature_config_used 中读取消融信息 ──────────
    eval_enabled_modalities = feature_config.get(
        "enabled_modalities", ["structured", "text", "media"]
    )
    logger.info(f"评估时 enabled_modalities: {eval_enabled_modalities}")

    # ── 6. 加载数据 ─────────────────────────────────────────
    is_three_way = "val_npz_path" in config

    # ── Categorical 配置（从 feature_config 或 feature_info 读取） ──────
    cat_enabled_eval = feature_config.get("categorical_enabled", False)
    if not cat_enabled_eval:
        # 回退检查 config
        cat_block = config.get("categorical", {})
        if isinstance(cat_block, dict):
            cat_enabled_eval = cat_block.get("enabled", False)
        else:
            cat_enabled_eval = False

    val_npz_path = project_root / config.get(
        "val_npz_path", config.get("eval_npz_path", "")
    )
    val_dataset = MultimodalDataset(val_npz_path, feature_info, categorical_enabled=cat_enabled_eval)
    for warn in val_dataset.warnings:
        logger.warning(f"验证集: {warn}")
    logger.info(f"验证样本数: {len(val_dataset)}")

    test_dataset = None
    if is_three_way:
        test_npz_path = project_root / config["test_npz_path"]
        test_dataset = MultimodalDataset(test_npz_path, feature_info, categorical_enabled=cat_enabled_eval)
        for warn in test_dataset.warnings:
            logger.warning(f"测试集: {warn}")
        logger.info(f"测试样本数: {len(test_dataset)}")

    # ── 7. 加载模型 ─────────────────────────────────────────
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"

    # Categorical embed dims from feature_config or feature_info
    cat_embed_dims_eval = feature_config.get("cat_embed_dims", [])
    if not cat_embed_dims_eval:
        cat_embed_dims_eval = feature_info.get("cat_embed_dims", [])

    model = MultimodalFusionModel(
        text_dim=text_dim,
        visual_dim=visual_dim,
        structured_dim=structured_dim,
        text_hidden_dim=config.get("text_hidden_dim", 32),
        visual_hidden_dim=config.get("visual_hidden_dim", 16),
        structured_hidden_dim=config.get("structured_hidden_dim", 32),
        fusion_hidden_dim=config.get("fusion_hidden_dim", 64),
        dropout=config.get("dropout", 0.3),
        enabled_modalities=eval_enabled_modalities,
        categorical_enabled=cat_enabled_eval,
        cat_embed_dims=cat_embed_dims_eval,
    ).to(device)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载完成: {model_path}")

    # ── 8. 推理与评估 ─────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    threshold = config.get("threshold", 0.5)

    # Val 评估
    logger.info("评估 val split...")
    val_result = evaluate_model(model, val_dataset, device, criterion, threshold)

    # Test 评估（三路模式）
    test_result = None
    if test_dataset is not None:
        logger.info("评估 test split...")
        test_result = evaluate_model(model, test_dataset, device, criterion, threshold)

    # ── 9. 保存 predictions ──────────────────────────────
    def save_predictions(eval_result: dict, split_name: str) -> None:
        pred_df = pd.DataFrame(
            {
                "sample_id": eval_result["sample_ids"],
                "video_id": eval_result["video_ids"],
                "author_id": eval_result["author_ids"],
                "label": eval_result["labels"],
                "score": eval_result["scores"],
                "pred": eval_result["preds"],
                "split": split_name,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_variant": dataset_variant,
                "run_id": run_id,
            }
        )
        pred_path = run_dir / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"预测结果已保存: {pred_path} ({len(pred_df)} 条)")

    save_predictions(val_result, "val")
    if test_result is not None:
        save_predictions(test_result, "test")

    # ── 10. 构建 metrics ─────────────────────────────────
    def build_metrics_dict(eval_result: dict, split: str) -> dict:
        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "run_id": run_id,
            "split": split,
            "sample_count": len(eval_result["labels"]),
            "positive_count": eval_result["n_pos"],
            "negative_count": eval_result["n_neg"],
            "eval_loss": eval_result["eval_loss"],
            "auc": eval_result["cls_metrics"].get("auc"),
            "accuracy": eval_result["cls_metrics"].get("accuracy"),
            "precision": eval_result["cls_metrics"].get("precision"),
            "recall": eval_result["cls_metrics"].get("recall"),
            "f1": eval_result["cls_metrics"].get("f1"),
            "precision_at_k": eval_result["pk_metrics"],
            "recall_at_k": eval_result["rk_metrics"],
            "threshold": threshold,
            "label_definition": label_definition,
            "warnings": eval_result["warnings"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    val_metrics = build_metrics_dict(val_result, "val")
    metrics_output: dict = {"val_metrics": val_metrics}

    if test_result is not None:
        test_metrics = build_metrics_dict(test_result, "test")
        metrics_output["test_metrics"] = test_metrics

    metrics_output["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    # ── 11. 打印指标摘要 ──────────────────────────────────
    logger.info("=== 评估结果 ===")
    logger.info(f"Val: 样本数={val_metrics['sample_count']}, "
                f"AUC={val_metrics['auc']}, F1={val_metrics['f1']}")
    if test_result:
        logger.info(f"Test: 样本数={test_metrics['sample_count']}, "
                    f"AUC={test_metrics['auc']}, F1={test_metrics['f1']}")

    all_warnings = val_result["warnings"]
    if test_result:
        all_warnings += test_result["warnings"]
    if all_warnings:
        logger.warning(f"Warnings: {all_warnings}")


if __name__ == "__main__":
    main()