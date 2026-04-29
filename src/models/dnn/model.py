"""DNN 二分类模型"""

from __future__ import annotations

import torch
import torch.nn as nn


class DNNModel(nn.Module):
    """DNN 二分类模型。

    结构：
    - 数值特征直连分支
    - 类别特征 Embedding 分支
    - 拼接后 MLP
    - 二分类 logit 输出
    """

    def __init__(
        self,
        numeric_dim: int,
        cat_embed_dims: list[tuple[int, int]],
        hidden_units: list[int] | None = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            numeric_dim: 数值特征维度
            cat_embed_dims: [(vocab_size, embed_dim), ...] 每个类别特征的词表和嵌入维度
            hidden_units: MLP 隐藏层单元数
            dropout: dropout 概率
        """
        super().__init__()
        if hidden_units is None:
            hidden_units = [64, 32]

        # 类别特征 Embedding
        self.cat_embeddings = nn.ModuleList()
        cat_embed_total = 0
        for vocab_size, embed_dim in cat_embed_dims:
            self.cat_embeddings.append(
                nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            )
            cat_embed_total += embed_dim

        # 输入维度 = 数值 + 所有类别 embedding 摊平
        input_dim = numeric_dim + cat_embed_total

        # MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, numeric, categorical):
        """前向传播。

        Args:
            numeric: (batch, numeric_dim) 数值特征
            categorical: (batch, num_cat) 类别特征索引

        Returns:
            logits: (batch,) 二分类 logit
        """
        # Embed + flatten categorical features
        if len(self.cat_embeddings) > 0:
            cat_embs = []
            for i, embed_layer in enumerate(self.cat_embeddings):
                cat_embs.append(embed_layer(categorical[:, i].long()))
            cat_flat = torch.cat(cat_embs, dim=1)
            combined = torch.cat([numeric, cat_flat], dim=1)
        else:
            combined = numeric

        # MLP → logit
        x = self.mlp(combined)
        logit = self.output_layer(x).squeeze(-1)
        return logit