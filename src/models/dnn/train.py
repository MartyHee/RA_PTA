"""DNN 训练主程序"""

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
from models.dnn.dataset import DNNDataProcessor, TabularDataset, get_excluded_cols  # noqa: E402
from models.dnn.model import DNNModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.io import read_csv_safe  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("dnn_train")


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dnn/dnn_base.yaml",
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
    train_path = project_root / config["train_data_path"]
    eval_path = project_root / config["eval_data_path"]
    feature_info_path = project_root / config["feature_info_path"]
    quality_check_path = project_root / config["quality_check_path"]
    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)
    with open(quality_check_path, "r", encoding="utf-8") as f:
        quality_check = json.load(f)

    # ── 5. 确定特征列 ───────────────────────────────────────
    excluded = get_excluded_cols(quality_check)
    logger.info(f"排除无效字段数: {len(excluded)}")

    # 数值特征 = numeric_cols + text_stat_cols - 排除项
    numeric_candidates = feature_info.get("numeric_cols", []) + feature_info.get(
        "text_stat_cols", []
    )
    numeric_cols = [c for c in numeric_candidates if c not in excluded]

    # 类别特征 = categorical_cols - 排除项
    categorical_candidates = feature_info.get("categorical_cols", [])
    categorical_cols = [c for c in categorical_candidates if c not in excluded]

    id_cols = feature_info.get("id_cols", [])
    label_col = feature_info.get("label_col", "label")

    logger.info(f"数值特征数: {len(numeric_cols)}, 类别特征数: {len(categorical_cols)}")
    logger.info(f"数值特征: {numeric_cols}")
    logger.info(f"类别特征: {categorical_cols}")

    # ── 6. 加载数据 ─────────────────────────────────────────
    train_df, _ = read_csv_safe(str(train_path))
    eval_df, _ = read_csv_safe(str(eval_path))
    logger.info(f"训练样本: {len(train_df)}, 评估样本: {len(eval_df)}")

    # 检查列是否存在
    for col_list, name in [
        (numeric_cols, "numeric"),
        (categorical_cols, "categorical"),
    ]:
        missing = [c for c in col_list if c not in train_df.columns]
        if missing:
            logger.warning(f"训练数据中缺少 {name} 列: {missing}")
            for c in missing:
                col_list.remove(c)

    # ── 7. 拟合处理器 + 转换数据 ────────────────────────────
    processor = DNNDataProcessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
        label_col=label_col,
    )
    processor.fit(train_df)

    train_data = processor.transform(train_df)
    eval_data = processor.transform(eval_df)

    # ── 8. Dataset / DataLoader ─────────────────────────────
    train_dataset = TabularDataset(
        train_data["numeric"], train_data["categorical"], train_data["labels"]
    )
    eval_dataset = TabularDataset(
        eval_data["numeric"], eval_data["categorical"], eval_data["labels"]
    )

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ── 9. 初始化模型 ──────────────────────────────────────
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"

    model = DNNModel(
        numeric_dim=len(numeric_cols),
        cat_embed_dims=processor.cat_embed_dims,
        hidden_units=config.get("hidden_units", [64, 32]),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    logger.info(f"模型参数数: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"设备: {device}")

    # ── 10. 优化器 + 损失 ───────────────────────────────────
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # ── 11. 训练循环 ───────────────────────────────────────
    epochs = config.get("epochs", 20)
    threshold = config.get("threshold", 0.5)
    best_eval_loss = float("inf")
    train_log: list[dict] = []

    k_values = [5, 10, 20]

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_train_losses: list[float] = []
        for batch in train_loader:
            numeric_b = batch["numeric"].to(device)
            cat_b = batch["categorical"].to(device)
            labels_b = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(numeric_b, cat_b)
            loss = criterion(logits, labels_b)
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
                numeric_b = batch["numeric"].to(device)
                cat_b = batch["categorical"].to(device)
                labels_b = batch["label"].to(device)

                logits = model(numeric_b, cat_b)
                loss = criterion(logits, labels_b)
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
            torch.save(model.state_dict(), str(output_dir / "model.pt"))
            logger.info(f"  -> 保存最佳模型 (eval_loss={avg_eval_loss:.4f})")

    # ── 12. 保存训练日志 ────────────────────────────────────
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info(f"训练日志已保存: {output_dir / 'train_log.csv'}")

    # ── 13. 保存特征配置 ────────────────────────────────────
    feature_config = processor.get_config()
    feature_config["excluded_cols_applied"] = sorted(excluded)
    feature_config["train_data_path"] = str(train_path)
    feature_config["eval_data_path"] = str(eval_path)
    feature_config["warnings"] = [
        "当前数据集仅 79 条样本，所有指标仅用于流程验证",
        "部分特征列因全零或常量已被排除",
        "标签为 interaction_score 伪标签，不代表真实业务目标",
    ]
    feature_config["notes"] = [
        "当前 DNN 基于 sample0427 样本数据训练，仅用于流程级验证。",
        "排除的字段包括：全空字段、占位字段、全-1字段、全零列、常量列。",
    ]

    with open(output_dir / "feature_config_used.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)
    logger.info(f"特征配置已保存: {output_dir / 'feature_config_used.json'}")

    # ── 14. 保存 run_meta ──────────────────────────────────
    run_meta = {
        "run_id": run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "dnn",
        "script": "train.py",
        "config": args.config,
        "output_dir": str(output_dir),
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