"""多模态数据加载器 — 从 npz 文件读取三模态特征"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """多模态 PyTorch Dataset，加载 npz 中的三模态特征。

    Args:
        npz_path: .npz 文件路径
        feature_info: 可选的 feature_info dict, 用于维度校验
    """

    def __init__(
        self,
        npz_path: str | Path,
        feature_info: dict[str, Any] | None = None,
    ) -> None:
        self.npz_path = str(npz_path)
        data = np.load(self.npz_path, allow_pickle=True)

        # ── 提取字段 ──────────────────────────────────────────
        self.sample_id: np.ndarray = data["sample_id"]
        self.video_id: np.ndarray = data["video_id"]
        self.author_id: np.ndarray = data["author_id"]
        self.label: np.ndarray = data["label"].astype(np.float32)
        self.text_features: np.ndarray = data["text_features"].astype(np.float32)
        self.visual_features: np.ndarray = data["visual_features"].astype(np.float32)
        self.structured_features: np.ndarray = data["structured_features"].astype(
            np.float32
        )
        self.split: np.ndarray = data["split"]

        self.n_samples = len(self.label)

        # ── 维度校验 ──────────────────────────────────────────
        if feature_info is not None:
            expected_text_dim = feature_info.get("text_dim", 32)
            expected_visual_dim = feature_info.get("visual_dim", 10)
            expected_structured_dim = feature_info.get("structured_dim", 20)
            if self.text_features.shape[1] != expected_text_dim:
                raise ValueError(
                    f"text_features dim mismatch: "
                    f"expected {expected_text_dim}, got {self.text_features.shape[1]}"
                )
            if self.visual_features.shape[1] != expected_visual_dim:
                raise ValueError(
                    f"visual_features dim mismatch: "
                    f"expected {expected_visual_dim}, got {self.visual_features.shape[1]}"
                )
            if self.structured_features.shape[1] != expected_structured_dim:
                raise ValueError(
                    f"structured_features dim mismatch: "
                    f"expected {expected_structured_dim}, "
                    f"got {self.structured_features.shape[1]}"
                )

        # ── label 只包含 0/1 检查 ──────────────────────────────
        unique_labels = np.unique(self.label)
        if not set(unique_labels).issubset({0.0, 1.0}):
            raise ValueError(
                f"label 包含非 0/1 值: {unique_labels}"
            )

        # ── NaN / inf 检查 ─────────────────────────────────────
        self.warnings: list[str] = []
        for name, arr in [
            ("text_features", self.text_features),
            ("visual_features", self.visual_features),
            ("structured_features", self.structured_features),
        ]:
            if np.any(np.isnan(arr)):
                count = int(np.sum(np.isnan(arr)))
                self.warnings.append(
                    f"{name} 包含 {count} 个 NaN 值，已替换为 0"
                )
                arr[np.isnan(arr)] = 0.0
            if np.any(np.isinf(arr)):
                count = int(np.sum(np.isinf(arr)))
                self.warnings.append(
                    f"{name} 包含 {count} 个 inf 值，已替换为 0"
                )
                arr[np.isinf(arr)] = 0.0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "text": torch.from_numpy(self.text_features[idx]),
            "visual": torch.from_numpy(self.visual_features[idx]),
            "structured": torch.from_numpy(self.structured_features[idx]),
            "label": torch.tensor(self.label[idx], dtype=torch.float32),
            "sample_id": self.sample_id[idx],
            "video_id": self.video_id[idx],
            "author_id": str(self.author_id[idx]),
            "split": str(self.split[idx]),
        }

    def get_all_labels(self) -> np.ndarray:
        """返回全部 label 数组，用于训练循环中的批量指标计算。"""
        return self.label

    def get_all_scores(self, model: torch.nn.Module, device: str) -> np.ndarray:
        """使用给定模型对所有样本做推理，返回 sigmoid 概率数组。"""
        model.eval()
        all_scores: list[float] = []
        with torch.no_grad():
            for i in range(self.n_samples):
                text_t = (
                    torch.from_numpy(self.text_features[i]).unsqueeze(0).to(device)
                )
                visual_t = (
                    torch.from_numpy(self.visual_features[i]).unsqueeze(0).to(device)
                )
                struct_t = (
                    torch.from_numpy(self.structured_features[i])
                    .unsqueeze(0)
                    .to(device)
                )
                logit = model(text_t, visual_t, struct_t)
                score = torch.sigmoid(logit)
                all_scores.append(score.cpu().item())
        return np.array(all_scores, dtype=np.float32)

    def get_info(self) -> dict[str, Any]:
        """返回数据集的元信息。"""
        return {
            "npz_path": self.npz_path,
            "n_samples": self.n_samples,
            "text_dim": self.text_features.shape[1],
            "visual_dim": self.visual_features.shape[1],
            "structured_dim": self.structured_features.shape[1],
            "warnings": self.warnings,
        }