"""多模态融合模型 — 支持消融的分支动态禁用

默认全模态融合，通过 enabled_modalities 控制启用分支。
禁用模态的 Linear 层不被创建，不参与 forward，不贡献参数量和梯度。

结构:
  - text_branch (可选):       Linear(text_dim → text_hidden_dim) → ReLU → Dropout
  - visual_branch (可选):     Linear(visual_dim → visual_hidden_dim) → ReLU → Dropout
  - structured_branch (可选): Linear(structured_dim → structured_hidden_dim) → ReLU → Dropout
  - fusion:                   Concat(启用分支输出) → Linear(fusion_input → fusion_hidden_dim) → ReLU → Dropout → Linear(fusion_hidden_dim → 1)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# 所有支持的模态名称
ALL_MODALITIES = {"structured", "text", "media"}


class MultimodalFusionModel(nn.Module):
    """轻量多模态融合模型，支持消融实验的动态分支禁用 + 可选的 categorical embedding。

    当 categorical_enabled=True 且 "structured" 在 enabled_modalities 中时，
    为每个 categorical feature 创建独立的 nn.Embedding，输出 concat 到 structured branch 输出后进入融合层。
    """

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
        enabled_modalities: list[str] | None = None,
        categorical_enabled: bool = False,
        cat_embed_dims: list[list[int]] | None = None,
    ) -> None:
        super().__init__()

        if enabled_modalities is None:
            enabled_modalities = ["structured", "text", "media"]

        # 校验
        self._validate_enabled_modalities(enabled_modalities)
        self.enabled_modalities = list(enabled_modalities)
        self._feature_dims: dict[str, int | None] = {}
        self._disabled_modalities: list[str] = sorted(
            ALL_MODALITIES - set(enabled_modalities)
        )

        # ── Categorical embedding 状态 ─────────────────────────
        # 只在 structured 模态启用时激活 categorical embedding
        self.categorical_enabled = (
            categorical_enabled and "structured" in self.enabled_modalities
        )
        self.cat_embed_dims = cat_embed_dims or []
        self.cat_total_dim = 0

        # 条件创建分支 + 记录 feature_dims
        self._branch_output_dims: list[int] = []

        if "text" in self.enabled_modalities:
            self.text_branch = nn.Sequential(
                nn.Linear(text_dim, text_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self._feature_dims["text"] = text_dim
            self._branch_output_dims.append(text_hidden_dim)
        else:
            self._feature_dims["text"] = None

        if "media" in self.enabled_modalities:
            self.visual_branch = nn.Sequential(
                nn.Linear(visual_dim, visual_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self._feature_dims["media"] = visual_dim
            self._branch_output_dims.append(visual_hidden_dim)
        else:
            self._feature_dims["media"] = None

        if "structured" in self.enabled_modalities:
            self.structured_branch = nn.Sequential(
                nn.Linear(structured_dim, structured_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self._feature_dims["structured"] = structured_dim

            # 计算 structured branch 输出的实际维度
            base_structured_output = structured_hidden_dim
            if self.categorical_enabled:
                self.cat_embeddings = nn.ModuleList()
                for vocab_size, embed_dim in self.cat_embed_dims:
                    self.cat_embeddings.append(
                        nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                    )
                self.cat_total_dim = sum(d for _, d in self.cat_embed_dims)
                structured_output_dim = base_structured_output + self.cat_total_dim
            else:
                structured_output_dim = base_structured_output

            self._branch_output_dims.append(structured_output_dim)
        else:
            self._feature_dims["structured"] = None

        # ── fusion ────────────────────────────────────────────
        fusion_input_dim = sum(self._branch_output_dims)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    @staticmethod
    def _validate_enabled_modalities(modalities: list[str]) -> None:
        """校验 enabled_modalities 参数合法性。"""
        if not isinstance(modalities, (list, tuple)):
            raise ValueError(
                f"enabled_modalities 必须为列表，收到: {type(modalities)}"
            )
        if len(modalities) == 0:
            raise ValueError(
                "enabled_modalities 为空列表——至少需启用一个模态。"
            )
        unknown = set(modalities) - ALL_MODALITIES
        if unknown:
            raise ValueError(
                f"enabled_modalities 包含未知模态: {sorted(unknown)}. "
                f"合法值: {sorted(ALL_MODALITIES)}"
            )

    def get_ablation_info(self) -> dict[str, Any]:
        """返回消融元信息，用于记录到 run_meta 和 feature_config。"""
        info: dict[str, Any] = {
            "enabled_modalities": self.enabled_modalities,
            "disabled_modalities": self._disabled_modalities,
            "feature_dims": self._feature_dims,
            "modality_branch_mode": "dynamic_disable",
            "categorical_enabled": self.categorical_enabled,
            "categorical_active": self.categorical_enabled,
            "cat_total_dim": self.cat_total_dim,
        }
        if self.categorical_enabled:
            # 注意：cat_embed_dims 包含 vocab_size 和 embed_dim
            cat_vocab_sizes = [vd[0] for vd in self.cat_embed_dims]
            cat_embed_dims_only = [vd[1] for vd in self.cat_embed_dims]
            info["categorical_vocab_sizes"] = cat_vocab_sizes
            info["categorical_embedding_dims"] = cat_embed_dims_only
        return info

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        structured_features: torch.Tensor,
        categorical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播，返回 logit。

        禁用模态的输入 tensor 仍然可传入，但不参与计算。
        只有启用模态的分支产生输出并进入融合层。

        Args:
            text_features:          (batch, text_dim) 或任意 tensor（禁用时）
            visual_features:        (batch, visual_dim) 或任意 tensor（禁用时）
            structured_features:    (batch, structured_dim) 或任意 tensor（禁用时）
            categorical_features:   (batch, num_cat_features) 可选，long 类型

        Returns:
            logits: (batch, 1)
        """
        branch_outputs: list[torch.Tensor] = []

        if hasattr(self, "text_branch"):
            branch_outputs.append(self.text_branch(text_features))
        if hasattr(self, "visual_branch"):
            branch_outputs.append(self.visual_branch(visual_features))
        if hasattr(self, "structured_branch"):
            num_repr = self.structured_branch(structured_features)
            if (
                hasattr(self, "cat_embeddings")
                and categorical_features is not None
            ):
                cat_embs = [
                    emb(categorical_features[:, i])
                    for i, emb in enumerate(self.cat_embeddings)
                ]
                cat_flat = torch.cat(cat_embs, dim=1)
                struct_repr = torch.cat([num_repr, cat_flat], dim=1)
                branch_outputs.append(struct_repr)
            else:
                branch_outputs.append(num_repr)

        fused = (
            torch.cat(branch_outputs, dim=1)
            if len(branch_outputs) > 1
            else branch_outputs[0]
        )
        logit = self.fusion(fused)
        return logit