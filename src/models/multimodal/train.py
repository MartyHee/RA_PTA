"""多模态模型训练主程序 — 支持 train/val/test 三路切分"""

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
from models.multimodal.dataset import MultimodalDataset  # noqa: E402
from models.multimodal.fusion_model import MultimodalFusionModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("multimodal_train")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="多模态模型训练与评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal/multimodal_base.yaml",
        help="配置文件路径（相对于项目根目录）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────
    config = load_config(args.config)
    logger.info(f"配置加载完成: {args.config}")
    dataset_name = config.get("dataset_name", "sample0427")
    dataset_variant = config.get("dataset_variant", "")
    model_name = config.get("model_name", "multimodal")

    # ── 2. 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    train_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {dataset_name}")

    # ── 3. 随机种子 ─────────────────────────────────────────
    set_seed(config.get("random_seed", 2026))

    # ── 4. 路径 ─────────────────────────────────────────────
    project_root = Path(_project_root)
    train_npz_path = project_root / config["train_npz_path"]
    val_npz_path = project_root / config["val_npz_path"]
    feature_info_path = project_root / config["feature_info_path"]
    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ── 5. 加载 feature_info ────────────────────────────────
    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    text_dim = feature_info.get("text_dim", 32)
    visual_dim = feature_info.get("visual_dim", 18)
    structured_dim = feature_info.get("structured_dim", 38)

    # ── Categorical 配置 ─────────────────────────────────────
    # 支持两种 override 方式: 嵌套 categorical.enabled 或扁平 categorical_enabled
    cat_block = config.get("categorical", {})
    if isinstance(cat_block, dict):
        categorical_enabled = cat_block.get("enabled", config.get("categorical_enabled", False))
    else:
        categorical_enabled = config.get("categorical_enabled", False)
    categorical_enabled = bool(categorical_enabled)

    cat_embed_dims: list[list[int]] | None = None
    cat_features_list: list[str] = feature_info.get("categorical_features", [])
    if categorical_enabled:
        cat_embed_dims = feature_info.get("cat_embed_dims", [])
        if not cat_embed_dims:
            logger.warning("categorical_enabled=True 但 multimodal_feature_info 中无 cat_embed_dims，禁用 categorical")
            categorical_enabled = False
        elif not cat_features_list:
            logger.warning("categorical_enabled=True 但 multimodal_feature_info 中无 categorical_features，禁用 categorical")
            categorical_enabled = False
        else:
            logger.info(f"Categorical embedding 已启用: features={cat_features_list}, embed_dims={cat_embed_dims}")

    # ── 消融实验配置 ─────────────────────────────────────────
    enabled_modalities = config.get("enabled_modalities", ["structured", "text", "media"])
    ablation_name = config.get("ablation_name", "all_modalities")
    if isinstance(enabled_modalities, str):
        # 兼容 YAML 字符串列表解析
        import json as _json
        try:
            enabled_modalities = _json.loads(enabled_modalities.replace("'", '"'))
        except Exception:
            enabled_modalities = [m.strip() for m in enabled_modalities.strip("[]").split(",")]
    logger.info(
        f"消融配置: ablation_name={ablation_name}, "
        f"enabled_modalities={enabled_modalities}"
    )

    label_definition = feature_info.get(
        "label_definition",
        "interaction_score >= threshold (离线实验伪标签)",
    )

    logger.info(
        f"模态维度: text={text_dim}, visual={visual_dim}, "
        f"structured={structured_dim}"
    )

    # ── 5a. 泄漏控制检查 ────────────────────────────────────
    leakage_control_passed = True
    leakage_check_errors: list[str] = []
    leakage_report_path_cfg = config.get("leakage_report_path")
    if leakage_report_path_cfg:
        lr_path = project_root / leakage_report_path_cfg
        if lr_path.exists():
            with open(lr_path, "r", encoding="utf-8") as f:
                lr_report = json.load(f)
            leakage_control_passed = lr_report.get("leakage_check_passed", False)
            if not leakage_control_passed:
                msg = f"泄漏检查未通过: {lr_report.get('errors', [])}"
                leakage_check_errors.append(msg)
                logger.error(msg)
                sys.exit(1)
            logger.info(f"泄漏检查已通过: {lr_path}")
        else:
            logger.warning(f"泄漏检查报告不存在: {lr_path}")

    # ── 6. 加载数据 ─────────────────────────────────────────
    # 三路切分模式 (real_raw_1000): 配置中含 test_npz_path
    is_three_way = "test_npz_path" in config

    train_dataset = MultimodalDataset(train_npz_path, feature_info, categorical_enabled=categorical_enabled)

    if is_three_way:
        test_npz_path = project_root / config["test_npz_path"]
        val_dataset = MultimodalDataset(val_npz_path, feature_info, categorical_enabled=categorical_enabled)
        test_dataset = MultimodalDataset(test_npz_path, feature_info, categorical_enabled=categorical_enabled)
        logger.info(f"测试样本: {len(test_dataset)}")
    else:
        # 二路切分模式 (sample0427): val 用作 eval
        val_dataset = MultimodalDataset(val_npz_path, feature_info)
        test_dataset = None

    for warn in train_dataset.warnings:
        logger.warning(f"训练集: {warn}")
    for warn in val_dataset.warnings:
        logger.warning(f"验证集: {warn}")
    if test_dataset:
        for warn in test_dataset.warnings:
            logger.warning(f"测试集: {warn}")

    logger.info(
        f"训练样本: {len(train_dataset)} "
        f"(正={int(train_dataset.label.sum())}, "
        f"负={int(len(train_dataset) - train_dataset.label.sum())})"
    )
    logger.info(
        f"验证样本: {len(val_dataset)} "
        f"(正={int(val_dataset.label.sum())}, "
        f"负={int(len(val_dataset) - val_dataset.label.sum())})"
    )

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # ── 7. 初始化模型 ──────────────────────────────────────
    device = config.get("device", "cuda")
    device_fallback_reason = None
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device_fallback_reason = "CUDA not available, fallback to CPU"
        device = "cpu"
    logger.info(f"设备: {device}")

    # ── Fusion 类型 ──────────────────────────────────────────
    fusion_type = config.get("fusion_type", "concat_mlp")
    late_fusion_mode = config.get("late_fusion_mode", "weighted_sum")
    logger.info(f"Fusion 类型: {fusion_type}")
    if fusion_type == "late_fusion":
        logger.info(f"Late fusion 模式: {late_fusion_mode}")

    model = MultimodalFusionModel(
        text_dim=text_dim,
        visual_dim=visual_dim,
        structured_dim=structured_dim,
        text_hidden_dim=config.get("text_hidden_dim", 32),
        visual_hidden_dim=config.get("visual_hidden_dim", 16),
        structured_hidden_dim=config.get("structured_hidden_dim", 32),
        fusion_hidden_dim=config.get("fusion_hidden_dim", 64),
        dropout=config.get("dropout", 0.3),
        enabled_modalities=enabled_modalities,
        categorical_enabled=categorical_enabled,
        cat_embed_dims=cat_embed_dims,
        fusion_type=fusion_type,
        late_fusion_mode=late_fusion_mode,
    ).to(device)

    # 获取消融元信息，供后续 metadata 使用
    ablation_info = model.get_ablation_info()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数: {n_params}")
    logger.info(
        f"模型结构: "
        f"ablation={ablation_name}, "
        f"modalities={enabled_modalities}, "
        f"text({text_dim}->{config.get('text_hidden_dim', 32)}), "
        f"visual({visual_dim}->{config.get('visual_hidden_dim', 16)}), "
        f"structured({structured_dim}->{config.get('structured_hidden_dim', 32)}), "
        f"fusion->{config.get('fusion_hidden_dim', 64)}->1"
    )

    # ── 8. 优化器 + 损失 ───────────────────────────────────
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # ── 9. 训练循环 ───────────────────────────────────────
    epochs = config.get("epochs", 50)
    threshold = config.get("threshold", 0.5)
    patience = config.get("early_stopping_patience", 8)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_log: list[dict] = []

    k_values = [5, 10, 20, 50]

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_train_losses: list[float] = []
        for batch in train_loader:
            text_b = batch["text"].to(device)
            visual_b = batch["visual"].to(device)
            struct_b = batch["structured"].to(device)
            labels_b = batch["label"].to(device)
            cat_b = batch.get("categorical")
            if cat_b is not None:
                cat_b = cat_b.to(device)

            optimizer.zero_grad()
            logits = model(text_b, visual_b, struct_b, cat_b)
            loss = criterion(logits.squeeze(), labels_b)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_train_losses))

        # --- Val ---
        model.eval()
        epoch_val_losses: list[float] = []
        all_labels: list[float] = []
        all_scores: list[float] = []
        with torch.no_grad():
            for i in range(len(val_dataset)):
                item = val_dataset[i]
                text_t = item["text"].unsqueeze(0).to(device)
                visual_t = item["visual"].unsqueeze(0).to(device)
                struct_t = item["structured"].unsqueeze(0).to(device)
                labels_t = item["label"].unsqueeze(0).to(device)
                cat_t = item.get("categorical")
                if cat_t is not None:
                    cat_t = cat_t.unsqueeze(0).to(device)

                logit = model(text_t, visual_t, struct_t, cat_t)
                loss = criterion(logit.view(-1), labels_t.view(-1))
                epoch_val_losses.append(loss.item())

                score = torch.sigmoid(logit)
                all_labels.append(labels_t.item())
                all_scores.append(score.item())

        avg_val_loss = float(np.mean(epoch_val_losses))
        all_labels_arr = np.array(all_labels)
        all_scores_arr = np.array(all_scores)
        all_preds_arr = (all_scores_arr >= threshold).astype(int)

        cls_metrics, _ = compute_classification_metrics(
            all_labels_arr, all_scores_arr, all_preds_arr, threshold
        )

        is_best = avg_val_loss < best_val_loss

        auc_str = (
            f"{cls_metrics.get('auc', 0):.4f}"
            if cls_metrics.get("auc") is not None
            else "N/A"
        )
        log_msg = (
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {avg_val_loss:.4f} | "
            f"AUC: {auc_str}"
        )
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), str(output_dir / "model.pt"))
            log_msg += "  *"
        else:
            epochs_no_improve += 1

        logger.info(log_msg)

        train_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "auc": cls_metrics.get("auc"),
                "accuracy": cls_metrics.get("accuracy"),
                "precision": cls_metrics.get("precision"),
                "recall": cls_metrics.get("recall"),
                "f1": cls_metrics.get("f1"),
                "is_best": is_best,
            }
        )

        # Early stopping
        if not is_best and epochs_no_improve >= patience:
            logger.info(
                f"Early stopping at epoch {epoch+1}, "
                f"best epoch: {best_epoch} "
                f"(val_loss={best_val_loss:.6f})"
            )
            break

    train_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_train_loss = train_log[-1]["train_loss"]
    final_val_loss = train_log[-1]["val_loss"]

    # ── 10. 保存训练日志 ──────────────────────────────────
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info(f"训练日志已保存: {output_dir / 'train_log.csv'}")

    # ── 11. 加载最佳模型做最终评估 ──────────────────────────
    if best_epoch > 0:
        logger.info(
            f"加载最佳模型 (epoch {best_epoch}, val_loss={best_val_loss:.6f})"
        )
        model.load_state_dict(torch.load(str(output_dir / "model.pt"),
                                         map_location=device))
        # 重新获取 ablation_info（包含训练后的 late_fusion 权重）
        ablation_info = model.get_ablation_info()
    else:
        logger.warning("未保存任何最佳模型，使用当前模型直接评估")

    # Val 评估
    val_result = evaluate_model(model, val_dataset, device, criterion, threshold)

    # Test 评估（三路模式）
    test_result = None
    if is_three_way and test_dataset is not None:
        logger.info("对 test split 进行最终评估...")
        test_result = evaluate_model(model, test_dataset, device, criterion, threshold)

    # ── 12. 保存 predictions ──────────────────────────────
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
        pred_path = output_dir / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"预测结果已保存: {pred_path} ({len(pred_df)} 条)")

    save_predictions(val_result, "val")
    if test_result is not None:
        save_predictions(test_result, "test")

    # ── 13. 构建 metrics ─────────────────────────────────
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

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {output_dir / 'metrics.json'}")

    # ── 14. 保存特征配置 ──────────────────────────────────
    # Categorical metadata
    cat_feature_list = feature_info.get("categorical_features", [])
    cat_vocab_sizes = (
        ablation_info.get("categorical_vocab_sizes", [])
        if categorical_enabled else []
    )
    cat_embed_dims_only = (
        ablation_info.get("categorical_embedding_dims", [])
        if categorical_enabled else []
    )

    feature_config = {
        "dataset_name": dataset_name,
        "ablation_name": ablation_name,
        "enabled_modalities": enabled_modalities,
        "disabled_modalities": ablation_info["disabled_modalities"],
        "feature_dims": ablation_info["feature_dims"],
        "train_npz_path": str(train_npz_path),
        "val_npz_path": str(val_npz_path),
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "structured_dim": structured_dim,
        "label_definition": label_definition,
        "text_feature_method": feature_info.get("text_feature_method", "tfidf_svd"),
        "text_profile": feature_info.get("text_profile", {"name": "merged_text_v1", "mode": "merged_tfidf_svd"}),
        "visual_feature_method": feature_info.get(
            "visual_feature_method", "media_metadata_only"
        ),
        "structured_feature_method": feature_info.get(
            "structured_feature_method", "tabular_numeric_scaled"
        ),
        "branch_hidden_dims": {
            "text": config.get("text_hidden_dim", 32),
            "visual": config.get("visual_hidden_dim", 16),
            "structured": config.get("structured_hidden_dim", 32),
        },
        "fusion_hidden_dim": config.get("fusion_hidden_dim", 64),
        # Fusion info
        "fusion_type": fusion_type,
        "late_fusion_mode": late_fusion_mode if fusion_type == "late_fusion" else None,
        "late_fusion_modality_order": ablation_info.get("late_fusion_modality_order"),
        "late_fusion_weights_raw": ablation_info.get("late_fusion_weights_raw"),
        "late_fusion_weights_softmax": ablation_info.get("late_fusion_weights_softmax"),
        # Categorical info
        "categorical_enabled": categorical_enabled,
        "categorical_active": ablation_info.get("categorical_active", False),
        "categorical_features": cat_feature_list,
        "categorical_vocab_sizes": cat_vocab_sizes,
        "categorical_embedding_dims": cat_embed_dims_only,
        "categorical_embedding_total_dim": ablation_info.get("cat_total_dim", 0),
        "categorical_vocab_source": feature_info.get(
            "categorical_vocab_source",
            "data/features/real_raw_5000/tabular_feature_info.json",
        ),
        "future_categorical_candidates": feature_info.get(
            "future_categorical_candidates", []
        ),
        "structured_numeric_dim": structured_dim,
        # Misc
        "no_image_download_confirmed": config.get("no_image_download", True),
        "no_external_api_confirmed": config.get("no_external_api", True),
        "no_large_pretrained_model_confirmed": config.get(
            "no_large_pretrained_model", True
        ),
        "warnings": [
            "标签为 interaction_score 伪标签，不代表真实业务目标",
        ],
        "notes": [
            f"当前多模态模型基于 {dataset_name} 数据训练",
            "未下载图片。",
            "未调用外部 API。",
            "未使用大型预训练模型。",
        ],
    }
    if is_three_way:
        feature_config["test_npz_path"] = str(test_npz_path)
    with open(output_dir / "feature_config_used.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)
    logger.info(f"特征配置已保存: {output_dir / 'feature_config_used.json'}")

    # ── 15. 保存 run_meta ──────────────────────────────────
    run_meta = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "input_dims": {
            "text": text_dim,
            "visual": visual_dim,
            "structured": structured_dim,
        },
        "train_started_at": train_started_at,
        "train_finished_at": train_finished_at,
        "config_path": args.config,
        "train_path": str(train_npz_path),
        "val_path": str(val_npz_path),
        "test_path": str(test_npz_path) if is_three_way else None,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "ablation_name": ablation_name,
        "enabled_modalities": enabled_modalities,
        "disabled_modalities": ablation_info["disabled_modalities"],
        "feature_dims": ablation_info["feature_dims"],
        "modality_branch_mode": ablation_info["modality_branch_mode"],
        "fusion_type": fusion_type,
        "late_fusion_mode": late_fusion_mode if fusion_type == "late_fusion" else None,
        "late_fusion_modality_order": ablation_info.get("late_fusion_modality_order"),
        "late_fusion_weights_raw": ablation_info.get("late_fusion_weights_raw"),
        "late_fusion_weights_softmax": ablation_info.get("late_fusion_weights_softmax"),
        "categorical_enabled": categorical_enabled,
        "categorical_active": ablation_info.get("categorical_active", False),
        "categorical_features": cat_feature_list,
        "categorical_vocab_sizes": cat_vocab_sizes,
        "categorical_embedding_dims": cat_embed_dims_only,
        "categorical_embedding_total_dim": ablation_info.get("cat_total_dim", 0),
        "structured_numeric_dim": structured_dim,
        "structured_cat_concat_dim": (
            structured_dim + ablation_info.get("cat_total_dim", 0)
            if categorical_enabled else structured_dim
        ),
        "future_categorical_candidates": feature_info.get(
            "future_categorical_candidates", []
        ),
        "test_loss": test_result["eval_loss"] if test_result else None,
        "device": device,
        "num_params": n_params,
        "label_definition": label_definition,
        "text_profile": feature_info.get("text_profile", {"name": "merged_text_v1", "mode": "merged_tfidf_svd"}),
        "leakage_control_passed": leakage_control_passed,
        "warnings": leakage_check_errors.copy(),
        "source_tuning_run_id": config.get("source_tuning_run_id"),
        "source_best_trial_id": config.get("source_best_trial_id"),
    }
    if device_fallback_reason:
        run_meta["warnings"].append(device_fallback_reason)

    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 记录最新 run_id
    with open(output_root / "latest_run.txt", "w") as f:
        f.write(run_id)

    # ── 16. 打印摘要 ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"训练完成！")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    auc_val = val_metrics.get("auc", "N/A")
    f1_val = val_metrics.get("f1", "N/A")
    logger.info(f"Val  AUC: {auc_val}, F1: {f1_val}")
    if test_result:
        auc_test = test_metrics.get("auc", "N/A")
        f1_test = test_metrics.get("f1", "N/A")
        logger.info(f"Test AUC: {auc_test}, F1: {f1_test}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()