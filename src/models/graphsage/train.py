"""GraphSAGE 训练主程序"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphSAGE 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/graphsage/graphsage_base.yaml",
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
    graph_data_dir = project_root / config["graph_data_dir"]
    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 5. 检测 torch_geometric ─────────────────────────────
    torch_geometric_available = check_torch_geometric()
    fallback_used = False
    logger.info(f"torch_geometric 可用: {torch_geometric_available}")

    # ── 6. 加载图数据 ───────────────────────────────────────
    graph_data = GraphData(
        node_features_path=str(project_root / config["node_features_path"]),
        labels_path=str(project_root / config["labels_path"]),
        train_mask_path=str(project_root / config["train_mask_path"]),
        eval_mask_path=str(project_root / config["eval_mask_path"]),
        edge_path=str(project_root / config["edge_path"]),
        node_path=str(project_root / config["node_path"]),
        graph_meta_path=str(project_root / config["graph_meta_path"]),
    )
    logger.info(
        f"图数据加载完成: {graph_data.num_nodes} 节点, "
        f"{graph_data.edge_index.shape[1]} 边, "
        f"feature_dim={graph_data.feature_dim}, "
        f"train_labeled={int(graph_data.train_labeled_mask.sum().item())}, "
        f"eval_labeled={int(graph_data.eval_labeled_mask.sum().item())}"
    )

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
    eval_data = graph_data.get_eval_data()

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
    logger.info(f"模型参数数: {sum(p.numel() for p in model.parameters())}")

    # ── 9. 优化器 + 损失 ───────────────────────────────────
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # ── 10. 训练循环 ───────────────────────────────────────
    epochs = config.get("epochs", 20)
    threshold = config.get("threshold", 0.5)
    best_eval_loss = float("inf")
    best_epoch = 0
    train_log: list[dict] = []

    k_values = [5, 10, 20]
    train_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 全图张量移至设备
    x = train_data["x"].to(device)
    edge_index = train_data["edge_index"].to(device)
    y = train_data["y"].to(device)
    train_mask = train_data["mask"].to(device)
    eval_mask = eval_data["mask"].to(device)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        avg_train_loss = loss.item()

        # --- Eval ---
        model.eval()
        with torch.no_grad():
            eval_logits = model(x, edge_index)
            eval_loss = criterion(eval_logits[eval_mask], y[eval_mask])
            avg_eval_loss = eval_loss.item()

            eval_scores = torch.sigmoid(eval_logits[eval_mask]).cpu().numpy()
            eval_labels = y[eval_mask].cpu().numpy()
            eval_preds = (eval_scores >= threshold).astype(int)

        cls_metrics, _ = compute_classification_metrics(
            eval_labels, eval_scores, eval_preds, threshold
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
            logger.info(f"  -> 保存最佳模型 (eval_loss={avg_eval_loss:.4f}, epoch={epoch+1})")

    train_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 11. 保存训练日志 ────────────────────────────────────
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(output_dir / "train_log.csv", index=False)
    logger.info(f"训练日志已保存: {output_dir / 'train_log.csv'}")

    # ── 12. 保存特征配置 ────────────────────────────────────
    feature_config = graph_data.get_feature_config()
    feature_config["hidden_dim"] = config.get("hidden_dim", 64)
    feature_config["num_layers"] = config.get("num_layers", 2)
    feature_config["graph_backend"] = "torch_geometric" if torch_geometric_available else "manual_graphsage"
    feature_config["torch_geometric_available"] = torch_geometric_available
    feature_config["fallback_used"] = fallback_used
    feature_config["excluded_label_value"] = -1
    feature_config["label_col"] = "label"
    feature_config["warnings"] = [
        "当前图数据基于 sample0427 构建，仅用于流程级验证。",
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

    # ── 13. 保存 run_meta ──────────────────────────────────
    run_meta = {
        "model_name": "graphsage",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "train_started_at": train_started_at,
        "train_finished_at": train_finished_at,
        "config_path": args.config,
        "graph_data_dir": str(graph_data_dir),
        "graph_backend": "torch_geometric" if torch_geometric_available else "manual_graphsage",
        "torch_geometric_available": torch_geometric_available,
        "fallback_used": fallback_used,
        "best_epoch": best_epoch,
        "best_eval_loss": round(best_eval_loss, 6),
        "device": str(device),
        "feature_normalization_enabled": normalization_applied,
        "notes": [
            "当前 GraphSAGE 基于 sample0427 图数据训练，仅用于流程级验证。",
        ],
    }
    if normalization_applied:
        run_meta["previous_run_id"] = "202604291703"
        run_meta["reason_for_rerun"] = "feature normalization for numeric stability"

    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 记录最新 run_id
    with open(output_root / "latest_run.txt", "w") as f:
        f.write(run_id)

    logger.info(f"训练完成！输出目录: {output_dir}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"最佳 epoch: {best_epoch}, 最佳 eval_loss: {best_eval_loss:.6f}")


if __name__ == "__main__":
    main()
