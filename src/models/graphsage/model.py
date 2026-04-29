"""GraphSAGE 模型定义（基于 torch_geometric.nn.SAGEConv）"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """多层 GraphSAGE 模型，每层使用 SAGEConv + ReLU + Dropout，最后接线性输出层。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, hidden_dim))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播，返回每个节点的 logit。

        Args:
            x: [num_nodes, in_dim] 节点特征。
            edge_index: [2, num_edges] 边索引。

        Returns:
            logits: [num_nodes]，每个节点一个 logit。
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.output_layer(x).squeeze(-1)  # [num_nodes]
        return logits