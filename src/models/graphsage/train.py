"""GraphSAGE 训练主程序（支持 train/val/test 三路 mask）"""

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
from models.graphsage.dataset import GraphData  # noqa: E402
from models.graphsage.model import GraphSAGE  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("graphsage_train")


def check_torch_geometric() -> bool:
    try:
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


def evaluate_split(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
    threshold: float,
    k_values: list[int],
) -> dict:
    """在指定 split 上计算 loss 和全部指标。

    Returns:
        dict: 包含 loss、分类指标、排序指标和 scores/labels/preds 的字典。
    """
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        loss = criterion(logits[mask], y[mask])
        loss_value = loss.item()

        scores = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = y[mask].cpu().numpy()
        preds = (scores >= threshold).astype(int)

    cls_metrics, cls_warnings = compute_classification_metrics(labels, scores, preds, threshold)
    pk_metrics, pk_warnings = compute_precision_at_k(labels, scores, k_values)
    rk_metrics, rk_warnings = compute_recall_at_k(labels, scores, k_values)

    return {
        "loss": loss_value,
        "scores": scores,
        "labels": labels,
        "preds": preds,
        "cls_metrics": cls_metrics,
        "pk_metrics": pk_metrics,
        "rk_metrics": rk_metrics,
        "warnings": cls_warnings + pk_warnings + rk_warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphSAGE 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/graphsage/graphsage_real_raw_1000.yaml",
        help="配置文件路径（相对于项目根目录）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────
    config = load_config(args.config)
    logger.info(f"配置加载完成: {args.config}")

    dataset_name = config.get("dataset_name", None)
    logger.info(f"数据集: {dataset_name or '(legacy, no dataset_name)'}")

    # ── 2. 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    logger.info(f"Run ID: {run_id}")

    # ── 3. 随机种子 ─────────────────────────────────────────
    set_seed(config.get("random_seed", 2026))

    # ── 4. 路径 ─────────────────────────────────────────────
    project_root = Path(_project_root)
    graph_data_dir = project_root / config["graph_data_dir"]
    output_root = project_root / config["output_root"]

    # 按 dataset_name 分目录输出（例如 real_raw_1000）
    if dataset_name:
        output_dir = output_root / dataset_name / run_id
    else:
        output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ── 5. 检测 torch_geometric ─────────────────────────────
    torch_geometric_available = check_torch_geometric()
    fallback_used = False
    logger.info(f"torch_geometric 可用: {torch_geometric_available}")

    # ── 6. 加载图数据 ───────────────────────────────────────
    # 向后兼容：优先使用 val_mask_path，回退到 eval_mask_path
    val_mask_path = config.get("val_mask_path") or config.get("eval_mask_path")
    test_mask_path = config.get("test_mask_path", None)

    graph_data = GraphData(
        node_features_path=str(project_root / config["node_features_path"]),
        labels_path=str(project_root / config["labels_path"]),
        train_mask_path=str(project_root / config["train_mask_path"]),
        val_mask_path=str(project_root / val_mask_path),
        test_mask_path=str(project_root / test_mask_path) if test_mask_path else None,
        edge_path=str(project_root / config["edge_path"]),
        node_path=str(project_root / config["node_path"]),
        graph_meta_path=str(project_root / config["graph_meta_path"]),
    )
    logger.info(
        f"图数据加载完成: {graph_data.num_nodes} 节点, "
        f"{graph_data.edge_index.shape[1]} 边, "
        f"feature_dim={graph_data.feature_dim}, "
        f"train_labeled={int(graph_data.train_labeled_mask.sum().item())}, "
        f"val_labeled={int(graph_data.val_labeled_mask.sum().item())}"
    )

    if graph_data.test_labeled_mask is not None:
        logger.info(f"test_labeled={int(graph_data.test_labeled_mask.sum().item())}")

    # ── 6a. 特征标准化（可选） ──────────────────────────────────
    norm_config = config.get("feature_normalization", {})
    normalization_applied = False
    if norm_config.get("enabled", False):
        logger.info("对节点数值特征进行 z-score 标准化...")
        norm_meta = graph_data.normalize_features(
            exclude_prefixes=norm_config.get("exclude_prefixes", ["node_type_"]),
        )
        logger.info(
            f"标准化列 ({len(norm_meta['normalized_feature_columns'])}): "
            f"{norm_meta['normalized_feature_columns']}"
        )
        logger.info(
            f"未标准化列 ({len(norm_meta['non_normalized_feature_columns'])}): "
            f"{norm_meta['non_normalized_feature_columns']}"
        )
        if norm_meta.get("constant_feature_columns"):
            logger.warning(f"常数列: {norm_meta['constant_feature_columns']}")
        normalization_applied = True
    else:
        logger.info("特征标准化未启用")
        norm_meta = None

    train_data = graph_data.get_train_data()
    val_data = graph_data.get_val_data()
    test_data = graph_data.get_test_data()

    # ── 7. 设备 ──────────────────────────────────────────────
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA 不可用，回退到 CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"设备: {device}")

    # ── 8. 初始化模型 ──────────────────────────────────────
    model = GraphSAGE(
        in_dim=graph_data.feature_dim,
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数: {num_params}")

    # ── 9. 优化器 + 损失 ───────────────────────────────────
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # ── 10. 训练循环 ───────────────────────────────────────
    epochs = config.get("epochs", 20)
    threshold = config.get("threshold", 0.5)
    k_values = [5, 10, 20]

    # 早停配置
    es_config = config.get("early_stopping", {})
    es_enabled = es_config.get("enabled", False)
    es_patience = es_config.get("patience", 10)
    es_counter = 0
    best_val_loss = float("inf")
    best_epoch = 0
    train_log: list[dict] = []

    train_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 全图张量移至设备
    x = train_data["x"].to(device)
    edge_index = train_data["edge_index"].to(device)
    y = train_data["y"].to(device)
    train_mask = train_data["mask"].to(device)
    val_mask = val_data["mask"].to(device)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        avg_train_loss = loss.item()

        # --- Val ---
        val_result = evaluate_split(
            model, x, edge_index, y, val_mask, criterion, threshold, k_values
        )
        avg_val_loss = val_result["loss"]
        val_cls = val_result["cls_metrics"]

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(output_dir / "model.pt"))
            es_counter = 0
            logger.info(
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {avg_val_loss:.4f} | "
                f"AUC: {val_cls.get('auc', 'N/A')} | "
                f"-> 保存最佳模型"
            )
        else:
            es_counter += 1
            logger.info(
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {avg_val_loss:.4f} | "
                f"AUC: {val_cls.get('auc', 'N/A')}"
            )

        train_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "auc": val_cls.get("auc"),
                "accuracy": val_cls.get("accuracy"),
                "precision": val_cls.get("precision"),
                "recall": val_cls.get("recall"),
                "f1": val_cls.get("f1"),
                "is_best": bool(is_best),
            }
        )

        # --- 早停 ---
        if es_enabled and es_counter >= es_patience:
            logger.info(f"早停触发: {es_patience} 个 epoch val_loss 未改善")
            break

    train_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"训练完成。最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.6f}")

    # ── 11. 加载最佳模型并评估 val / test ────────────────────
    best_model_path = output_dir / "model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"已加载最佳模型: {best_model_path}")

    # --- 最终 val 评估 ---
    x_all = graph_data.node_features.to(device)
    edge_index_all = graph_data.edge_index.to(device)
    y_all = graph_data.labels.to(device)

    final_val = evaluate_split(
        model, x_all, edge_index_all, y_all,
        val_data["mask"].to(device), criterion, threshold, k_values,
    )
    logger.info(f"最终 val_loss: {final_val['loss']:.4f}, AUC: {final_val['cls_metrics'].get('auc')}")

    # --- test 评估 ---
    test_result = None
    if test_data is not None:
        test_result = evaluate_split(
            model, x_all, edge_index_all, y_all,
            test_data["mask"].to(device), criterion, threshold, k_values,
        )
        logger.info(f"最终 test_loss: {test_result['loss']:.4f}, AUC: {test_result['cls_metrics'].get('auc')}")
    else:
        logger.warning("未提供 test_mask，跳过测试集评估")

    # ── 12. 保存训练日志 ────────────────────────────────────
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info(f"训练日志已保存: {output_dir / 'train_log.csv'}")

    # ── 13. 保存 val predictions ─────────────────────────────
    val_node_indices = torch.where(val_data["mask"].cpu())[0].numpy()
    val_nodes_df = graph_data.nodes_df.iloc[val_node_indices].copy()
    val_video_ids = val_nodes_df["raw_id"].values

    val_pred_df = pd.DataFrame(
        {
            "sample_id": val_video_ids,
            "node_id": val_node_indices,
            "video_id": val_video_ids,
            "label": final_val["labels"],
            "score": final_val["scores"],
            "pred": final_val["preds"],
            "split": "val",
            "model_name": "graphsage",
            "dataset_name": dataset_name or "",
            "run_id": run_id,
        }
    )
    val_pred_df.to_csv(output_dir / "predictions_val.csv", index=False)
    logger.info(f"Val 预测已保存: {output_dir / 'predictions_val.csv'} ({len(val_pred_df)} 行)")

    # ── 14. 保存 test predictions ────────────────────────────
    if test_result is not None:
        test_node_indices = torch.where(test_data["mask"].cpu())[0].numpy()
        test_nodes_df = graph_data.nodes_df.iloc[test_node_indices].copy()
        test_video_ids = test_nodes_df["raw_id"].values

        test_pred_df = pd.DataFrame(
            {
                "sample_id": test_video_ids,
                "node_id": test_node_indices,
                "video_id": test_video_ids,
                "label": test_result["labels"],
                "score": test_result["scores"],
                "pred": test_result["preds"],
                "split": "test",
                "model_name": "graphsage",
                "dataset_name": dataset_name or "",
                "run_id": run_id,
            }
        )
        test_pred_df.to_csv(output_dir / "predictions_test.csv", index=False)
        logger.info(f"Test 预测已保存: {output_dir / 'predictions_test.csv'} ({len(test_pred_df)} 行)")

    # ── 15. 保存 feature_config_used.json ────────────────────
    feature_config = graph_data.get_feature_config()
    feature_config["hidden_dim"] = config.get("hidden_dim", 64)
    feature_config["num_layers"] = config.get("num_layers", 2)
    feature_config["graph_backend"] = "torch_geometric" if torch_geometric_available else "manual_graphsage"
    feature_config["torch_geometric_available"] = torch_geometric_available
    feature_config["fallback_used"] = fallback_used
    feature_config["excluded_label_value"] = -1
    feature_config["label_col"] = "label"
    feature_config["warnings"] = [
        "当前图数据基于 real_raw_1000 构建。",
        "标签为 interaction_score 伪标签，不代表真实业务目标。",
    ]
    feature_config["notes"] = [
        "GraphSAGE 使用全图训练（无 mini-batch neighbor sampling）。",
        "loss 仅在 train_mask=True 且 label in {0,1} 的节点上计算。",
        "非主视频节点（label=-1）仅作为图上下文节点参与消息传递。",
    ]

    if normalization_applied:
        feature_config["feature_normalization_enabled"] = norm_meta["feature_normalization_enabled"]
        feature_config["feature_normalization_method"] = norm_meta["feature_normalization_method"]
        feature_config["normalization_fit_on"] = norm_meta["normalization_fit_on"]
        feature_config["normalized_feature_columns"] = norm_meta["normalized_feature_columns"]
        feature_config["normalized_feature_indices"] = norm_meta["normalized_feature_indices"]
        feature_config["non_normalized_feature_columns"] = norm_meta["non_normalized_feature_columns"]
        feature_config["normalization_mean"] = norm_meta["normalization_mean"]
        feature_config["normalization_std"] = norm_meta["normalization_std"]
        feature_config["constant_feature_columns"] = norm_meta["constant_feature_columns"]
        feature_config["device"] = str(device)

    with open(output_dir / "feature_config_used.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)
    logger.info(f"特征配置已保存: {output_dir / 'feature_config_used.json'}")

    # ── 16. 保存 metrics.json ──────────────────────────────
    label_definition = (graph_data.graph_meta or {}).get(
        "label_definition",
        "离线实验伪标签: interaction_score >= P60 (18147.80), 继承自 tabular",
    )

    def build_metric_block(
        split_name: str,
        result: dict,
    ) -> dict:
        n_pos = int(result["labels"].sum())
        n_neg = int(len(result["labels"]) - n_pos)
        return {
            "model_name": "graphsage",
            "dataset_name": dataset_name or "",
            "run_id": run_id,
            "split": split_name,
            "sample_count": len(result["labels"]),
            "positive_count": n_pos,
            "negative_count": n_neg,
            "eval_loss": result["loss"],
            **result["cls_metrics"],
            "precision_at_k": result["pk_metrics"],
            "recall_at_k": result["rk_metrics"],
            "threshold": threshold,
            "label_definition": label_definition,
            "warnings": result["warnings"],
        }

    metrics = {
        "val_metrics": build_metric_block("val", final_val),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if test_result is not None:
        metrics["test_metrics"] = build_metric_block("test", test_result)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {output_dir / 'metrics.json'}")

    # ── 17. 保存 run_meta.json ──────────────────────────────
    graph_meta = graph_data.graph_meta or {}
    run_meta = {
        "model_name": "graphsage",
        "dataset_name": dataset_name or "",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "graph_dir": str(graph_data_dir),
        "num_nodes": graph_data.num_nodes,
        "num_edges": graph_data.edge_index.shape[1],
        "feature_dim": graph_data.feature_dim,
        "train_labeled_count": int(graph_data.train_labeled_mask.sum().item()),
        "val_labeled_count": int(graph_data.val_labeled_mask.sum().item()),
        "test_labeled_count": int(graph_data.test_labeled_mask.sum().item()) if graph_data.test_labeled_mask is not None else 0,
        "train_started_at": train_started_at,
        "train_finished_at": train_finished_at,
        "config_path": args.config,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(train_log[-1]["train_loss"], 6) if train_log else None,
        "final_val_loss": round(final_val["loss"], 6),
        "test_loss": round(test_result["loss"], 6) if test_result is not None else None,
        "device": str(device),
        "num_params": num_params,
        "graph_backend": "torch_geometric" if torch_geometric_available else "manual_graphsage",
        "torch_geometric_available": torch_geometric_available,
        "fallback_used": fallback_used,
        "early_stopping": es_enabled,
        "early_stopping_patience": es_patience if es_enabled else None,
        "feature_normalization_enabled": normalization_applied,
        "label_definition": label_definition,
        "warnings": [],
    }

    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 记录最新 run_id（在 dataset_name 子目录内）
    if dataset_name:
        (output_root / dataset_name / "latest_run.txt").write_text(run_id)
    else:
        (output_root / "latest_run.txt").write_text(run_id)

    logger.info(f"训练评估完成！输出目录: {output_dir}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.6f}")
    if test_result is not None:
        logger.info(f"Test loss: {test_result['loss']:.6f}, Test AUC: {test_result['cls_metrics'].get('auc')}")


if __name__ == "__main__":
    main()