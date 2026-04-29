"""多模态模型训练主程序"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="多模态模型训练")
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

    # ── 2. 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    logger.info(f"Run ID: {run_id}")

    # ── 3. 随机种子 ─────────────────────────────────────────
    set_seed(config.get("random_seed", 2026))

    # ── 4. 路径 ─────────────────────────────────────────────
    project_root = Path(_project_root)
    train_npz_path = project_root / config["train_npz_path"]
    eval_npz_path = project_root / config["eval_npz_path"]
    feature_info_path = project_root / config["feature_info_path"]
    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ── 5. 加载 feature_info ────────────────────────────────
    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    text_dim = feature_info.get("text_dim", 32)
    visual_dim = feature_info.get("visual_dim", 10)
    structured_dim = feature_info.get("structured_dim", 20)
    logger.info(
        f"模态维度: text={text_dim}, visual={visual_dim}, "
        f"structured={structured_dim}"
    )

    # ── 6. 加载数据 ─────────────────────────────────────────
    train_dataset = MultimodalDataset(train_npz_path, feature_info)
    eval_dataset = MultimodalDataset(eval_npz_path, feature_info)

    for warn in train_dataset.warnings:
        logger.warning(f"训练集: {warn}")
    for warn in eval_dataset.warnings:
        logger.warning(f"评估集: {warn}")

    logger.info(
        f"训练样本: {len(train_dataset)} "
        f"(正={int(train_dataset.label.sum())}, "
        f"负={int(len(train_dataset) - train_dataset.label.sum())})"
    )
    logger.info(
        f"评估样本: {len(eval_dataset)} "
        f"(正={int(eval_dataset.label.sum())}, "
        f"负={int(len(eval_dataset) - eval_dataset.label.sum())})"
    )

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ── 7. 初始化模型 ──────────────────────────────────────
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"
    logger.info(f"设备: {device}")

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

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数: {total_params}")
    logger.info(
        f"模型结构: "
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
    epochs = config.get("epochs", 20)
    threshold = config.get("threshold", 0.5)
    best_eval_loss = float("inf")
    best_epoch = 0
    train_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_log: list[dict] = []

    k_values = [5, 10, 20]

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_train_losses: list[float] = []
        for batch in train_loader:
            text_b = batch["text"].to(device)
            visual_b = batch["visual"].to(device)
            struct_b = batch["structured"].to(device)
            labels_b = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(text_b, visual_b, struct_b)
            loss = criterion(logits.squeeze(), labels_b)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_train_losses))

        # --- Eval ---
        model.eval()
        epoch_eval_losses: list[float] = []
        all_labels: list[float] = []
        all_scores: list[float] = []
        with torch.no_grad():
            for batch in eval_loader:
                text_b = batch["text"].to(device)
                visual_b = batch["visual"].to(device)
                struct_b = batch["structured"].to(device)
                labels_b = batch["label"].to(device)

                logits = model(text_b, visual_b, struct_b)
                loss = criterion(logits.squeeze(), labels_b)
                epoch_eval_losses.append(loss.item())

                scores = torch.sigmoid(logits)
                all_labels.extend(labels_b.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        avg_eval_loss = float(np.mean(epoch_eval_losses))
        all_labels_arr = np.array(all_labels)
        all_scores_arr = np.array(all_scores)
        all_preds_arr = (all_scores_arr >= threshold).astype(int)

        cls_metrics, _ = compute_classification_metrics(
            all_labels_arr, all_scores_arr, all_preds_arr, threshold
        )

        logger.info(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"eval_loss: {avg_eval_loss:.4f} | "
            f"AUC: {cls_metrics.get('auc', 'N/A')}"
        )

        train_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "eval_loss": round(avg_eval_loss, 6),
                "eval_auc": cls_metrics.get("auc"),
                "eval_accuracy": cls_metrics.get("accuracy"),
                "eval_precision": cls_metrics.get("precision"),
                "eval_recall": cls_metrics.get("recall"),
                "eval_f1": cls_metrics.get("f1"),
            }
        )

        # --- 保存最佳模型 ---
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(output_dir / "model.pt"))
            logger.info(f"  -> 保存最佳模型 (eval_loss={avg_eval_loss:.4f})")

    train_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 10. 保存训练日志 ──────────────────────────────────
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info(f"训练日志已保存: {output_dir / 'train_log.csv'}")

    # ── 11. 保存特征配置 ──────────────────────────────────
    feature_config = {
        "train_npz_path": str(train_npz_path),
        "eval_npz_path": str(eval_npz_path),
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "structured_dim": structured_dim,
        "text_feature_method": feature_info.get("text_feature_method", "tfidf_svd"),
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
        "no_image_download_confirmed": config.get("no_image_download", True),
        "no_external_api_confirmed": config.get("no_external_api", True),
        "no_large_pretrained_model_confirmed": config.get(
            "no_large_pretrained_model", True
        ),
        "label_col": feature_info.get("label_col", "label"),
        "warnings": [
            "当前数据集仅 79 条样本，所有指标仅用于流程验证",
            "标签为 interaction_score 伪标签，不代表真实业务目标",
            "visual_features 仅包含媒体元信息，不包含图像语义特征",
        ],
        "notes": [
            "当前多模态模型基于 sample0427 样本数据训练，仅用于流程级验证。",
            "未下载图片。",
            "未调用外部 API。",
            "未使用大型预训练模型。",
        ],
    }
    with open(output_dir / "feature_config_used.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)
    logger.info(f"特征配置已保存: {output_dir / 'feature_config_used.json'}")

    # ── 12. 保存 run_meta ──────────────────────────────────
    run_meta = {
        "model_name": "multimodal",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "train_started_at": train_started_at,
        "train_finished_at": train_finished_at,
        "config_path": args.config,
        "train_npz_path": str(train_npz_path),
        "eval_npz_path": str(eval_npz_path),
        "best_epoch": best_epoch,
        "best_eval_loss": round(best_eval_loss, 6),
        "device": device,
        "notes": [
            "当前多模态模型基于 sample0427 样本数据训练，仅用于流程级验证。",
            "未下载图片。",
            "未调用外部 API。",
            "未使用大型预训练模型。",
        ],
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 记录最新 run_id
    with open(output_root / "latest_run.txt", "w") as f:
        f.write(run_id)

    logger.info(f"训练完成！输出目录: {output_dir}")
    logger.info(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()