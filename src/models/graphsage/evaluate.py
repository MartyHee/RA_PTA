"""GraphSAGE 评估主程序"""

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


def find_latest_run(output_root: Path) -> str | None:
    latest_file = output_root / "latest_run.txt"
    if latest_file.exists():
        return latest_file.read_text().strip()
    runs = sorted([d.name for d in output_root.iterdir() if d.is_dir() and d.name.isdigit()])
    return runs[-1] if runs else None


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

    graph_backend = feature_config.get("graph_backend", "unknown")
    torch_geometric_available = feature_config.get("torch_geometric_available", False)
    fallback_used = feature_config.get("fallback_used", False)
    logger.info(f"图模型后端: {graph_backend}, torch_geometric_available: {torch_geometric_available}")

    # ── 4. 加载图数据 ───────────────────────────────────────
    graph_data = GraphData(
        node_features_path=str(project_root / config["node_features_path"]),
        labels_path=str(project_root / config["labels_path"]),
        train_mask_path=str(project_root / config["train_mask_path"]),
        eval_mask_path=str(project_root / config["eval_mask_path"]),
        edge_path=str(project_root / config["edge_path"]),
        node_path=str(project_root / config["node_path"]),
        graph_meta_path=str(project_root / config["graph_meta_path"]),
    )
    logger.info(f"图数据加载完成: {graph_data.num_nodes} 节点, {graph_data.edge_index.shape[1]} 边")

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

    # ── 7. 推理 ─────────────────────────────────────────────
    x = graph_data.node_features.to(device)
    edge_index = graph_data.edge_index.to(device)
    y = graph_data.labels.to(device)
    eval_mask = graph_data.eval_labeled_mask.to(device)

    with torch.no_grad():
        logits = model(x, edge_index)
        eval_loss = nn.BCEWithLogitsLoss()(logits[eval_mask], y[eval_mask])
        eval_loss_value = eval_loss.item()

        eval_scores = torch.sigmoid(logits[eval_mask]).cpu().numpy()
        eval_labels = y[eval_mask].cpu().numpy()
        eval_node_indices = torch.where(eval_mask.cpu())[0].numpy()

    threshold = config.get("threshold", 0.5)
    eval_preds = (eval_scores >= threshold).astype(int)

    logger.info(f"Eval 样本数: {len(eval_labels)}")
    logger.info(f"Eval loss: {eval_loss_value:.4f}")

    # ── 8. 计算指标 ─────────────────────────────────────────
    cls_metrics, cls_warnings = compute_classification_metrics(
        eval_labels, eval_scores, eval_preds, threshold
    )

    k_values = [5, 10, 20]
    pk_metrics, pk_warnings = compute_precision_at_k(eval_labels, eval_scores, k_values)
    rk_metrics, rk_warnings = compute_recall_at_k(eval_labels, eval_scores, k_values)

    all_warnings = cls_warnings + pk_warnings + rk_warnings
    n_pos = int(eval_labels.sum())
    n_neg = int(len(eval_labels) - n_pos)

    # 图概况
    graph_meta = graph_data.graph_meta or {}
    graph_summary = {
        "num_nodes": graph_data.num_nodes,
        "num_edges": graph_data.edge_index.shape[1],
        "feature_dim": graph_data.feature_dim,
        "train_labeled": int(graph_data.train_labeled_mask.sum().item()),
        "eval_labeled": int(graph_data.eval_labeled_mask.sum().item()),
        "unlabeled": int((graph_data.labels < 0).sum().item()),
    }

    run_id_str = run_dir.name

    metrics = {
        "model_name": "graphsage",
        "run_id": run_id_str,
        "split": "eval",
        "sample_count": len(eval_labels),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "auc": cls_metrics.get("auc"),
        "accuracy": cls_metrics.get("accuracy"),
        "precision": cls_metrics.get("precision"),
        "recall": cls_metrics.get("recall"),
        "f1": cls_metrics.get("f1"),
        "precision_at_k": pk_metrics,
        "recall_at_k": rk_metrics,
        "eval_loss": eval_loss_value,
        "label_definition": "流程验证伪标签: interaction_score >= threshold (继承自 tabular)",
        "graph_summary": graph_summary,
        "graph_backend": graph_backend,
        "torch_geometric_available": torch_geometric_available,
        "fallback_used": fallback_used,
        "threshold": threshold,
        "warnings": all_warnings,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "当前指标仅基于 sample0427 样本数据计算，仅用于流程级验证。",
            "不表示正式推荐系统效果结论。",
            "标签为 interaction_score 伪标签，不代表真实曝光/点击/转化目标。",
            "eval 仅在 eval_mask=True 且 label in {0,1} 的节点上计算指标。",
        ],
    }

    # ── 9. 保存 metrics.json ────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    # ── 10. 保存 predictions.csv ────────────────────────────
    nodes_df = graph_data.nodes_df
    eval_nodes_df = nodes_df.iloc[eval_node_indices].copy()
    eval_video_ids = eval_nodes_df["raw_id"].values

    pred_df = pd.DataFrame(
        {
            "node_id": eval_node_indices,
            "video_id": eval_video_ids,
            "label": eval_labels,
            "score": eval_scores,
            "pred": eval_preds,
            "split": "eval",
            "model_name": "graphsage",
            "run_id": run_id_str,
        }
    )
    # sample_id = video_id 兼容
    pred_df.insert(0, "sample_id", eval_video_ids)

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