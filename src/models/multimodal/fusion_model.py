"""多模态融合模型 — 三分支 + 融合 MLP

结构:
  - text_branch:       Linear(text_dim → text_hidden_dim) → ReLU → Dropout
  - visual_branch:     Linear(visual_dim → visual_hidden_dim) → ReLU → Dropout
  - structured_branch: Linear(structured_dim → structured_hidden_dim) → ReLU → Dropout
  - fusion:            Concat → Linear(fusion_input → fusion_hidden_dim) → ReLU → Dropout → Linear(fusion_hidden_dim → 1)
  - loss:              BCEWithLogitsLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):
    """轻量多模态融合模型，不依赖大型预训练模型。"""

    def __init__(
        self,
        text_dim: int = 32,
        visual_dim: int = 10,
        structured_dim: int = 20,
        text_hidden_dim: int = 32,
        visual_hidden_dim: int = 16,
        structured_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ── text branch ───────────────────────────────────────
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── visual branch ─────────────────────────────────────
        self.visual_branch = nn.Sequential(
            nn.Linear(visual_dim, visual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── structured branch ─────────────────────────────────
        self.structured_branch = nn.Sequential(
            nn.Linear(structured_dim, structured_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── fusion ────────────────────────────────────────────
        fusion_input_dim = text_hidden_dim + visual_hidden_dim + structured_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        structured_features: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，返回 logit。

        Args:
            text_features:       (batch, text_dim)
            visual_features:     (batch, visual_dim)
            structured_features: (batch, structured_dim)

        Returns:
            logits: (batch, 1)
        """
        text_out = self.text_branch(text_features)
        visual_out = self.visual_branch(visual_features)
        structured_out = self.structured_branch(structured_features)

        fused = torch.cat([text_out, visual_out, structured_out], dim=1)
        logit = self.fusion(fused)
        return logit