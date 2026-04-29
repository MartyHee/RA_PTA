"""GraphSAGE 图数据加载"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch


class GraphData:
    """加载并管理图数据，提供 PyTorch 张量格式的输出。"""

    def __init__(
        self,
        node_features_path: str,
        labels_path: str,
        train_mask_path: str,
        eval_mask_path: str,
        edge_path: str,
        node_path: str,
        graph_meta_path: str | None = None,
    ):
        self.node_features_path = node_features_path
        self.labels_path = labels_path
        self.train_mask_path = train_mask_path
        self.eval_mask_path = eval_mask_path
        self.edge_path = edge_path
        self.node_path = node_path
        self.graph_meta_path = graph_meta_path

        # 加载
        self.node_features: torch.Tensor = self._load_npy(node_features_path, "node_features")
        self.labels: torch.Tensor = self._load_npy(labels_path, "labels")
        self.train_mask: torch.Tensor = self._load_npy(train_mask_path, "train_mask")
        self.eval_mask: torch.Tensor = self._load_npy(eval_mask_path, "eval_mask")
        self.edge_index: torch.Tensor = self._load_edge_index(edge_path)
        self.nodes_df: pd.DataFrame = self._load_nodes_csv(node_path)

        self.num_nodes = self.node_features.shape[0]
        self.feature_dim = self.node_features.shape[1]

        # 图元信息
        self.graph_meta: dict | None = None
        if graph_meta_path and os.path.exists(graph_meta_path):
            with open(graph_meta_path, "r", encoding="utf-8") as f:
                self.graph_meta = json.load(f)

        # 监督训练 mask：train_mask 且 label in {0,1}
        self.train_labeled_mask = self.train_mask & (self.labels >= 0)
        self.eval_labeled_mask = self.eval_mask & (self.labels >= 0)

        # 校验
        self._validate()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @staticmethod
    def _load_npy(path: str, name: str) -> torch.Tensor:
        arr = np.load(path)
        if name in ("train_mask", "eval_mask"):
            return torch.tensor(arr, dtype=torch.bool)
        if name == "labels":
            return torch.tensor(arr, dtype=torch.float32)
        return torch.tensor(arr, dtype=torch.float32)

    @staticmethod
    def _load_edge_index(path: str) -> torch.Tensor:
        df = pd.read_csv(path)
        src = df["source_node_id"].values.astype(np.int64)
        dst = df["target_node_id"].values.astype(np.int64)
        return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    @staticmethod
    def _load_nodes_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        N = self.num_nodes
        assert self.labels.shape == (N,), f"labels shape mismatch: {self.labels.shape}"
        assert self.train_mask.shape == (N,), f"train_mask shape mismatch: {self.train_mask.shape}"
        assert self.eval_mask.shape == (N,), f"eval_mask shape mismatch: {self.eval_mask.shape}"

        # edge_index 不越界
        max_node = int(self.edge_index.max().item())
        assert max_node < N, f"edge_index 中最大节点 id {max_node} >= num_nodes {N}"

        # train/eval 不重叠
        overlap = self.train_mask & self.eval_mask
        assert overlap.sum().item() == 0, f"train/eval mask 重叠 {overlap.sum().item()} 个节点"

        # train/eval 有标签节点 label 只包含 0/1
        for mask, name in [(self.train_labeled_mask, "train"), (self.eval_labeled_mask, "eval")]:
            lbl = self.labels[mask]
            if lbl.numel() > 0:
                uniq = lbl.unique().tolist()
                for v in uniq:
                    assert v in (0.0, 1.0), f"{name} labeled 节点 label 包含 {v}（期望 0/1）"

    # ------------------------------------------------------------------
    # 获取训练/评估数据
    # ------------------------------------------------------------------
    def get_train_data(self) -> dict:
        return {
            "x": self.node_features,
            "edge_index": self.edge_index,
            "y": self.labels,
            "mask": self.train_labeled_mask,
        }

    def get_eval_data(self) -> dict:
        return {
            "x": self.node_features,
            "edge_index": self.edge_index,
            "y": self.labels,
            "mask": self.eval_labeled_mask,
        }

    def get_feature_config(self) -> dict:
        meta = self.graph_meta or {}
        return {
            "graph_data_dir": os.path.dirname(self.node_features_path),
            "num_nodes": self.num_nodes,
            "num_edges": self.edge_index.shape[1],
            "feature_dim": self.feature_dim,
            "feature_columns": meta.get("feature_columns", []),
            "node_type_counts": meta.get("node_type_counts", {}),
            "edge_type_counts": meta.get("edge_type_counts", {}),
            "train_node_count": int(self.train_mask.sum().item()),
            "eval_node_count": int(self.eval_mask.sum().item()),
            "labeled_node_count": int((self.labels >= 0).sum().item()),
            "unlabeled_node_count": int((self.labels < 0).sum().item()),
            "train_mask_count": int(self.train_labeled_mask.sum().item()),
            "eval_mask_count": int(self.eval_labeled_mask.sum().item()),
            "model_input_dim": self.feature_dim,
        }

    def to_pyg_data(self) -> "torch_geometric.data.Data":
        """转为 torch_geometric Data 对象。"""
        from torch_geometric.data import Data

        return Data(
            x=self.node_features,
            edge_index=self.edge_index,
            y=self.labels.long(),
            train_mask=self.train_mask,
            eval_mask=self.eval_mask,
        )