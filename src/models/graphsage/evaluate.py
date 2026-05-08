"""GraphSAGE 评估主程序（支持 val / test 评估）"""

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

logger = get_logger("graphsage_eval")


def find_latest_run(output_root: Path, dataset_name: str | None = None) -> str | None:
    if dataset_name:
        search_dir = output_root / dataset_name
    else:
        search_dir = output_root

    latest_file = search_dir / "latest_run.txt"
    if latest_file.exists():
        return latest_file.read_text().strip()
    runs = sorted([d.name for d in search_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    return runs[-1] if runs else None


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
    """在指定 split 上计算 loss 和全部指标。"""
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
    parser = argparse.ArgumentParser(description="GraphSAGE 评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/graphsage/graphsage_base.yaml",
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
    dataset_name = config.get("dataset_name", None)

    # ── 2. 确定输出目录 ─────────────────────────────────────
    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif args.run_id:
        if dataset_name:
            run_dir = output_root / dataset_name / args.run_id
        else:
            run_dir = output_root / args.run_id
    else:
        found_id = find_latest_run(output_root, dataset_name)
        if found_id is None:
            logger.error("未找到任何 run，请先运行 train.py 或指定 --run_id")
            sys.exit(1)
        if dataset_name:
            run_dir = output_root / dataset_name / found_id
        else:
            run_dir = output_root / found_id
        logger.info(f"使用最新 run: {found_id}")

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

    graph_backend = feature_config.get("graph_backend", "unknown")
    torch_geometric_available = feature_config.get("torch_geometric_available", False)
    fallback_used = feature_config.get("fallback_used", False)
    logger.info(f"图模型后端: {graph_backend}, torch_geometric_available: {torch_geometric_available}")

    # ── 4. 加载图数据 ───────────────────────────────────────
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
    logger.info(f"图数据加载完成: {graph_data.num_nodes} 节点, {graph_data.edge_index.shape[1]} 边")

    # ── 4a. 应用特征标准化 ─────────────────────────────────
    if feature_config.get("feature_normalization_enabled", False):
        logger.info("应用已保存的标准化参数...")
        graph_data.apply_normalization_from_meta(feature_config)
        logger.info("标准化已应用")

    # ── 5. 设备 ──────────────────────────────────────────────
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"设备: {device}")

    # ── 6. 加载模型 ─────────────────────────────────────────
    model = GraphSAGE(
        in_dim=graph_data.feature_dim,
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载完成: {model_path}")

    # ── 7. 全图推理 ─────────────────────────────────────────
    x = graph_data.node_features.to(device)
    edge_index = graph_data.edge_index.to(device)
    y = graph_data.labels.to(device)
    threshold = config.get("threshold", 0.5)
    k_values = [5, 10, 20]
    criterion = nn.BCEWithLogitsLoss()

    run_id_str = run_dir.name

    # --- Val 评估 ---
    val_mask = graph_data.val_labeled_mask.to(device)
    val_result = evaluate_split(model, x, edge_index, y, val_mask, criterion, threshold, k_values)
    logger.info(f"Val 样本数: {len(val_result['labels'])}, loss: {val_result['loss']:.4f}, AUC: {val_result['cls_metrics'].get('auc')}")

    # --- Test 评估 ---
    test_result = None
    if graph_data.test_labeled_mask is not None:
        test_mask = graph_data.test_labeled_mask.to(device)
        test_result = evaluate_split(model, x, edge_index, y, test_mask, criterion, threshold, k_values)
        logger.info(f"Test 样本数: {len(test_result['labels'])}, loss: {test_result['loss']:.4f}, AUC: {test_result['cls_metrics'].get('auc')}")

    # ── 8. 构建 metrics ─────────────────────────────────────
    label_definition = (graph_data.graph_meta or {}).get(
        "label_definition",
        "离线实验伪标签: interaction_score >= P60 (18147.80), 继承自 tabular",
    )

    def build_metric_block(split_name: str, result: dict) -> dict:
        n_pos = int(result["labels"].sum())
        n_neg = int(len(result["labels"]) - n_pos)
        return {
            "model_name": "graphsage",
            "dataset_name": dataset_name or "",
            "run_id": run_id_str,
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
        "val_metrics": build_metric_block("val", val_result),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if test_result is not None:
        metrics["test_metrics"] = build_metric_block("test", test_result)

    # ── 9. 保存 metrics.json ────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    # ── 10. 保存 predictions ────────────────────────────────
    nodes_df = graph_data.nodes_df

    val_node_indices = np.where(val_mask.cpu())[0].numpy()
    test_node_indices = np.where(test_mask.cpu())[0].numpy() if test_result is not None else None

    def save_predictions(split_name: str, result: dict, node_idx: np.ndarray, prefix: str) -> None:
        split_nodes_df = nodes_df.iloc[node_idx].copy()
        video_ids = split_nodes_df["raw_id"].values

        pred_df = pd.DataFrame(
            {
                "sample_id": video_ids,
                "node_id": node_idx,
                "video_id": video_ids,
                "label": result["labels"],
                "score": result["scores"],
                "pred": result["preds"],
                "split": split_name,
                "model_name": "graphsage",
                "dataset_name": dataset_name or "",
                "run_id": run_id_str,
            }
        )
        pred_path = run_dir / f"{prefix}_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"{split_name} 预测已保存: {pred_path} ({len(pred_df)} 行)")

    save_predictions("val", val_result, val_node_indices, "predictions")
    if test_result is not None:
        save_predictions("test", test_result, test_node_indices, "predictions")

    # ── 11. 打印指标摘要 ────────────────────────────────────
    logger.info("=== 评估结果 ===")

    def log_metrics(split_name: str, result: dict) -> None:
        n_pos = int(result["labels"].sum())
        n_neg = int(len(result["labels"]) - n_pos)
        logger.info(f"[{split_name}] 样本数: {len(result['labels'])} (正例: {n_pos}, 负例: {n_neg})")
        logger.info(f"[{split_name}] AUC: {result['cls_metrics'].get('auc')}")
        logger.info(f"[{split_name}] Accuracy: {result['cls_metrics'].get('accuracy')}")
        logger.info(f"[{split_name}] Precision: {result['cls_metrics'].get('precision')}")
        logger.info(f"[{split_name}] Recall: {result['cls_metrics'].get('recall')}")
        logger.info(f"[{split_name}] F1: {result['cls_metrics'].get('f1')}")
        logger.info(f"[{split_name}] Loss: {result['loss']:.4f}")

    log_metrics("val", val_result)
    if test_result is not None:
        log_metrics("test", test_result)

    if val_result["warnings"]:
        logger.warning(f"Val warnings: {val_result['warnings']}")
    if test_result and test_result["warnings"]:
        logger.warning(f"Test warnings: {test_result['warnings']}")


if __name__ == "__main__":
    main()