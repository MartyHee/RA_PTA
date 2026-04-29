"""Wide & Deep 二分类模型"""

from __future__ import annotations

import torch
import torch.nn as nn


class WideDeepModel(nn.Module):
    """Wide & Deep 二分类模型。

    结构：
    - Deep 部分：数值特征 + 类别 Embedding → 拼接 → MLP → deep_logit
    - Wide 部分：交叉特征 Embedding（dim=1） → 求和 → wide_logit
    - 融合：final_logit = wide_logit + deep_logit + bias
    """

    def __init__(
        self,
        numeric_dim: int,
        cat_embed_dims: list[tuple[int, int]],
        wide_vocab_sizes: list[int],
        deep_hidden_units: list[int] | None = None,
        dropout: float = 0.3,
        wide_embedding_dim: int = 1,
    ):
        """
        Args:
            numeric_dim: 数值特征维度
            cat_embed_dims: [(vocab_size, embed_dim), ...] 每个类别特征的词表和嵌入维度
            wide_vocab_sizes: [vocab_size, ...] 每个 wide 交叉特征的词表大小
            deep_hidden_units: Deep MLP 隐藏层单元数
            dropout: dropout 概率
            wide_embedding_dim: Wide 部分每个交叉特征的 embedding 维度（默认 1）
        """
        super().__init__()
        if deep_hidden_units is None:
            deep_hidden_units = [64, 32]

        # ── Deep 部分 ──────────────────────────────────────────

        # 类别特征 Embedding
        self.cat_embeddings = nn.ModuleList()
        cat_embed_total = 0
        for vocab_size, embed_dim in cat_embed_dims:
            self.cat_embeddings.append(
                nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            )
            cat_embed_total += embed_dim

        # Deep 输入维度 = 数值 + 所有类别 embedding 摊平
        deep_input_dim = numeric_dim + cat_embed_total

        # Deep MLP
        deep_layers = []
        prev_dim = deep_input_dim
        for hidden_dim in deep_hidden_units:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*deep_layers)
        self.deep_output = nn.Linear(prev_dim, 1)

        # ── Wide 部分 ──────────────────────────────────────────

        # 每个 wide 交叉特征使用一个 Embedding(vocab_size, 1)
        self.wide_embeddings = nn.ModuleList()
        for vocab_size in wide_vocab_sizes:
            self.wide_embeddings.append(
                nn.Embedding(vocab_size, wide_embedding_dim, padding_idx=0)
            )

        # ── Bias ───────────────────────────────────────────────
        self.wide_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        wide: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            numeric: (batch, numeric_dim) 数值特征
            categorical: (batch, num_cat) 类别特征索引
            wide: (batch, num_wide) wide 交叉特征索引

        Returns:
            logits: (batch,) 二分类 logit
        """
        # ── Deep 部分 ──
        if len(self.cat_embeddings) > 0:
            cat_embs = []
            for i, embed_layer in enumerate(self.cat_embeddings):
                cat_embs.append(embed_layer(categorical[:, i].long()))
            cat_flat = torch.cat(cat_embs, dim=1)
            deep_input = torch.cat([numeric, cat_flat], dim=1)
        else:
            deep_input = numeric

        deep_hidden = self.deep_mlp(deep_input)
        deep_logit = self.deep_output(deep_hidden).squeeze(-1)

        # ── Wide 部分 ──
        wide_logit = 0.0
        if len(self.wide_embeddings) > 0:
            for i, embed_layer in enumerate(self.wide_embeddings):
                wide_logit = wide_logit + embed_layer(wide[:, i].long()).squeeze(-1)

        # ── 融合 ──
        final_logit = deep_logit + wide_logit + self.wide_bias
        return final_logit