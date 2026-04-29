"""DNN 数据加载与预处理"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DNNDataProcessor:
    """处理 DNN 模型的数据：拟合 scaler/vocab，转换数据为张量。"""

    def __init__(
        self,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        id_cols: list[str] | None = None,
        label_col: str | None = "label",
    ):
        self.numeric_cols = list(numeric_cols) if numeric_cols else []
        self.categorical_cols = list(categorical_cols) if categorical_cols else []
        self.id_cols = list(id_cols) if id_cols else []
        self.label_col = label_col

        # 拟合后填充的状态
        self.medians: dict[str, float] = {}
        self.scaler: StandardScaler | None = None
        self.cat_vocabs: dict[str, dict] = {}
        self.cat_embed_dims: list[tuple[int, int]] = []

    def fit(self, df: pd.DataFrame) -> "DNNDataProcessor":
        """在训练集上拟合数值 scaler 和类别 vocabulary。"""
        # --- 数值特征 ---
        if self.numeric_cols:
            self.medians = df[self.numeric_cols].median().to_dict()
            filled = df[self.numeric_cols].fillna(self.medians)
            self.scaler = StandardScaler()
            self.scaler.fit(filled)

        # --- 类别特征 ---
        for col in self.categorical_cols:
            raw = df[col].fillna("__MISSING__")
            unique_vals = sorted(raw.unique().tolist())
            vocab: dict = {"__UNK__": 0}
            for i, v in enumerate(unique_vals):
                vocab[v] = i + 1  # 0 保留给 UNK
            self.cat_vocabs[col] = vocab
            embed_dim = min(16, max(4, int(len(vocab) ** 0.5) + 1))
            self.cat_embed_dims.append((len(vocab), embed_dim))

        return self

    def transform(self, df: pd.DataFrame) -> dict:
        """将 DataFrame 转换为模型输入张量。"""
        result: dict = {}

        # --- 数值特征 ---
        if self.numeric_cols:
            numeric_data = df[self.numeric_cols].fillna(self.medians).values
            numeric_data = self.scaler.transform(numeric_data)
            result["numeric"] = torch.tensor(numeric_data, dtype=torch.float32)
        else:
            result["numeric"] = torch.empty((len(df), 0))

        # --- 类别特征 ---
        if self.categorical_cols:
            cat_indices_list = []
            for col in self.categorical_cols:
                raw = df[col].fillna("__MISSING__").values
                indices = [self.cat_vocabs[col].get(v, 0) for v in raw]
                cat_indices_list.append(indices)
            # (num_cat, N) -> (N, num_cat)
            cat_tensor = torch.tensor(cat_indices_list, dtype=torch.long).t()
            result["categorical"] = cat_tensor
        else:
            result["categorical"] = torch.empty((len(df), 0), dtype=torch.long)

        # --- 标签 ---
        if self.label_col and self.label_col in df.columns:
            result["labels"] = torch.tensor(
                df[self.label_col].values, dtype=torch.float32
            )
        else:
            result["labels"] = None

        # --- ID 列 ---
        id_cols_present = [c for c in self.id_cols if c in df.columns]
        if id_cols_present:
            result["ids"] = df[id_cols_present].copy()
        else:
            result["ids"] = None

        return result

    def get_config(self) -> dict:
        """返回可序列化的配置字典（用于保存 feature_config_used.json）。"""
        config = {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "id_cols": self.id_cols,
            "label_col": self.label_col,
            "medians": self.medians,
            "cat_vocabs": self.cat_vocabs,
            "cat_embed_dims": self.cat_embed_dims,
        }
        if self.scaler is not None:
            config["scaler_mean"] = self.scaler.mean_.tolist()
            config["scaler_scale"] = self.scaler.scale_.tolist()
        else:
            config["scaler_mean"] = []
            config["scaler_scale"] = []
        return config

    @classmethod
    def from_config(cls, config: dict) -> "DNNDataProcessor":
        """从配置字典恢复处理器（用于 evaluate 加载）。"""
        processor = cls(
            numeric_cols=config.get("numeric_cols", []),
            categorical_cols=config.get("categorical_cols", []),
            id_cols=config.get("id_cols", []),
            label_col=config.get("label_col", "label"),
        )
        processor.medians = config.get("medians", {})
        if config.get("scaler_mean"):
            processor.scaler = StandardScaler()
            processor.scaler.mean_ = np.array(config["scaler_mean"])
            processor.scaler.scale_ = np.array(config["scaler_scale"])
        processor.cat_vocabs = config.get("cat_vocabs", {})
        raw_dims = config.get("cat_embed_dims", [])
        processor.cat_embed_dims = [tuple(d) for d in raw_dims]
        return processor


class TabularDataset(Dataset):
    """DNN 训练/评估用 PyTorch Dataset。"""

    def __init__(
        self,
        numeric_tensor: torch.Tensor,
        cat_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
    ):
        self.numeric = numeric_tensor
        self.categorical = cat_tensor
        self.labels = label_tensor

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "numeric": self.numeric[idx],
            "categorical": self.categorical[idx],
            "label": self.labels[idx],
        }


def get_excluded_cols(quality_check: dict) -> set[str]:
    """从质量检查报告中获取所有应排除的字段集合。"""
    excluded: set[str] = set()
    for key in [
        "excluded_all_null_cols",
        "excluded_placeholder_cols",
        "excluded_all_minus_one_cols",
    ]:
        for col in quality_check.get("excluded_fields", {}).get(key, []):
            excluded.add(col)
    # 全零列
    for col in quality_check.get("flagged_fields", {}).get("all_zero_cols", []):
        excluded.add(col)
    # 常量列
    for item in quality_check.get("flagged_fields", {}).get("constant_cols", []):
        excluded.add(item["col"])
    return excluded