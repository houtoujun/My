"""
基于图卷积的基线模型骨架。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class GCNBaseline(nn.Module):
    """简化的 GCN 占位实现。"""

    def __init__(self, config: Dict[str, Any], num_nodes: int) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.num_pred = int(config.get("num_pred", 12))
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        X: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if X.dim() == 4:
            X = X.squeeze(-1)
        if X.dim() != 3:
            raise ValueError("输入 X 需为 [B, P, N] 或 [B, P, N, 1]。")
        batch = X.size(0)
        return X.new_zeros(batch, self.num_pred, self.num_nodes)


__all__ = ["GCNBaseline"]
