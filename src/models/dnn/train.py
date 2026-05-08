"""DNN 训练主程序 — 支持 train/val/test 三路切分"""

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
from models.dnn.dataset import (  # noqa: E402
    DNNDataProcessor,
    TabularDataset,
    get_excluded_cols,
    get_features_from_feature_info,
)
from models.dnn.model import DNNModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.io import read_csv_safe  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("dnn_train")


def validate_columns(df: pd.DataFrame, col_list: list, name: str) -> list:
    """检查列是否存在，记录缺失列并过滤。"""
    missing = [c for c in col_list if c not in df.columns]
    if missing:
        logger.warning(f"数据中缺少 {name} 列: {missing}")
    return [c for c in col_list if c in df.columns]


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
    parser = argparse.ArgumentParser(description="DNN 训练与评估")
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
    dataset_name = config.get("dataset_name", "sample0427")
    model_name = config.get("model_name", "dnn")

    # ── 2. 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    train_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {dataset_name}")

    # ── 3. 随机种子 ─────────────────────────────────────────
    set_seed(config.get("random_seed", 2026))

    # ── 4. 路径 ─────────────────────────────────────────────
    project_root = Path(_project_root)
    feature_info_path = project_root / config["feature_info_path"]

    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 5. 加载 feature info ────────────────────────────────
    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    label_definition = feature_info.get(
        "label_definition",
        "interaction_score >= threshold (离线实验伪标签)",
    )

    # ── 6. 确定模式 ─────────────────────────────────────────
    # 三路切分模式 (real_raw_1000): 配置中含 test_data_path
    # 二路切分模式 (sample0427): 配置中含 eval_data_path + quality_check_path
    is_three_way = "test_data_path" in config

    if is_three_way:
        # ── 三路切分模式 — 直接从 feature_info 使用特征列表 ───
        feature_lists = get_features_from_feature_info(feature_info)
        numeric_cols = feature_lists["numeric_cols"]
        categorical_cols = feature_lists["categorical_cols"]
        id_cols = feature_lists["id_cols"]
        label_col = feature_lists["label_col"]
        # text_stat_cols 只用于记录 feature_config
        text_stat_cols_used = feature_lists["text_stat_cols"]

        val_path = project_root / config["val_data_path"]
        test_path = project_root / config["test_data_path"]
    else:
        # ── 二路切分模式 (sample0427) — 基于 quality_check 过滤 ──
        quality_check_path = project_root / config.get("quality_check_path", "")
        if quality_check_path.exists():
            with open(quality_check_path, "r", encoding="utf-8") as f:
                quality_check = json.load(f)
            excluded = get_excluded_cols(quality_check)
        else:
            excluded = set()
            logger.warning("quality_check_path 不存在，跳过字段排除")

        numeric_candidates = list(feature_info.get("numeric_cols", [])) + list(
            feature_info.get("text_stat_cols", [])
        )
        numeric_cols = [c for c in numeric_candidates if c not in excluded]
        categorical_candidates = list(feature_info.get("categorical_cols", []))
        categorical_cols = [c for c in categorical_candidates if c not in excluded]
        id_cols = list(feature_info.get("id_cols", []))
        label_col = feature_info.get("label_col", "label")
        text_stat_cols_used = []

        val_path = project_root / config["eval_data_path"]
        test_path = None

    logger.info(f"数值特征数: {len(numeric_cols)}, 类别特征数: {len(categorical_cols)}")
    logger.info(f"数值特征: {numeric_cols}")
    logger.info(f"类别特征: {categorical_cols}")
    logger.info(f"ID 列: {id_cols}")

    # ── 7. 加载数据 ─────────────────────────────────────────
    train_path = project_root / config["train_data_path"]
    train_df, _ = read_csv_safe(str(train_path))
    val_df, _ = read_csv_safe(str(val_path))
    logger.info(f"训练样本: {len(train_df)}, 验证样本: {len(val_df)}")

    test_df = None
    if is_three_way:
        test_df, _ = read_csv_safe(str(test_path))
        logger.info(f"测试样本: {len(test_df)}")

    # 检查列是否存在
    numeric_cols = validate_columns(train_df, numeric_cols, "numeric")
    categorical_cols = validate_columns(train_df, categorical_cols, "categorical")

    # ── 8. 拟合处理器（仅 train） + 转换数据 ────────────────
    processor = DNNDataProcessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
        label_col=label_col,
    )
    processor.fit(train_df)

    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)

    # ── 9. Dataset / DataLoader ─────────────────────────────
    train_dataset = TabularDataset(
        train_data["numeric"], train_data["categorical"], train_data["labels"]
    )
    val_dataset = TabularDataset(
        val_data["numeric"], val_data["categorical"], val_data["labels"]
    )

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ── 10. 初始化模型 ──────────────────────────────────────
    device = config.get("device", "cuda")
    device_fallback_reason = None
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device_fallback_reason = "CUDA not available, fallback to CPU"
        device = "cpu"

    model = DNNModel(
        numeric_dim=len(numeric_cols),
        cat_embed_dims=processor.cat_embed_dims,
        hidden_units=config.get("hidden_units", [64, 32]),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数: {n_params}")
    logger.info(f"设备: {device}")

    # ── 11. 优化器 + 损失 ───────────────────────────────────
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # ── 12. 训练循环 ───────────────────────────────────────
    epochs = config.get("epochs", 40)
    threshold = config.get("threshold", 0.5)
    patience = config.get("early_stopping_patience", 8)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_log: list[dict] = []

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

        # --- Val ---
        model.eval()
        epoch_val_losses: list[float] = []
        all_labels: list[float] = []
        all_scores: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                numeric_b = batch["numeric"].to(device)
                cat_b = batch["categorical"].to(device)
                labels_b = batch["label"].to(device)

                logits = model(numeric_b, cat_b)
                loss = criterion(logits, labels_b)
                epoch_val_losses.append(loss.item())

                scores = torch.sigmoid(logits)
                all_labels.extend(labels_b.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

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

    final_train_loss = train_log[-1]["train_loss"]
    final_val_loss = train_log[-1]["val_loss"]

    # ── 13. 保存训练日志 ────────────────────────────────────
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info(f"训练日志已保存: {output_dir / 'train_log.csv'}")

    # ── 14. 加载最佳模型做最终评估 ───────────────────────────
    if best_epoch > 0:
        logger.info(
            f"加载最佳模型 (epoch {best_epoch}, val_loss={best_val_loss:.6f})"
        )
        model.load_state_dict(torch.load(str(output_dir / "model.pt"),
                                         map_location=device))
    else:
        logger.warning("未保存任何最佳模型，使用当前模型直接评估")

    # Val 评估
    val_result = evaluate(model, val_loader, device, criterion, threshold)

    # Test 评估（三路模式）
    test_result = None
    test_data = None
    if is_three_way and test_df is not None:
        test_data = processor.transform(test_df)
        test_dataset = TabularDataset(
            test_data["numeric"], test_data["categorical"], test_data["labels"]
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        test_result = evaluate(model, test_loader, device, criterion, threshold)

    # ── 15. 保存 predictions ────────────────────────────────
    def save_predictions(
        eval_result: dict, split_name: str, ids_data
    ) -> None:
        pred_df = pd.DataFrame(
            {
                "label": eval_result["labels"],
                "score": eval_result["scores"],
                "pred": eval_result["preds"],
                "split": split_name,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "run_id": run_id,
            }
        )
        if ids_data is not None:
            ids_df = ids_data.reset_index(drop=True)
            pred_df = pd.concat([ids_df, pred_df], axis=1)
        pred_path = output_dir / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"预测结果已保存: {pred_path} ({len(pred_df)} 条)")

    save_predictions(val_result, "val", val_data.get("ids"))
    if test_result is not None:
        save_predictions(test_result, "test", test_data.get("ids"))

    # ── 16. 构建 metrics ────────────────────────────────────
    def build_metrics_dict(
        eval_result: dict, split: str
    ) -> dict:
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

    # ── 17. 保存特征配置 ────────────────────────────────────
    feature_config = processor.get_config()
    feature_config["dataset_name"] = dataset_name
    feature_config["label_definition"] = label_definition
    feature_config["author_id_as_categorical"] = "author_id" in categorical_cols
    if text_stat_cols_used:
        feature_config["text_stat_features"] = text_stat_cols_used

    if is_three_way:
        feature_config["train_data_path"] = str(train_path)
        feature_config["val_data_path"] = str(val_path)
        feature_config["test_data_path"] = str(test_path)
    else:
        feature_config["eval_data_path"] = str(val_path)
        feature_config["train_data_path"] = str(train_path)

    feature_config["warnings"] = [
        "标签为 interaction_score 伪标签，不代表真实业务目标",
    ]
    feature_config["notes"] = [
        f"当前 DNN 基于 {dataset_name} 数据训练",
    ]

    with open(output_dir / "feature_config_used.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)
    logger.info(f"特征配置已保存: {output_dir / 'feature_config_used.json'}")

    # ── 18. 保存 run_meta ───────────────────────────────────
    run_meta = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "train_started_at": train_started_at,
        "train_finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.config,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path) if test_path else None,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "device": device,
        "num_params": n_params,
        "label_definition": label_definition,
        "warnings": [],
    }
    if device_fallback_reason:
        run_meta["warnings"].append(device_fallback_reason)

    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 记录最新 run_id
    with open(output_root / "latest_run.txt", "w") as f:
        f.write(run_id)

    # ── 19. 打印摘要 ────────────────────────────────────────
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