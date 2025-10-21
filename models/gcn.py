"""
图卷积 + 序列建模的轻量级基线实现。

思路：
1. 依据地理邻接矩阵（若不存在则退化为全连接）做一次归一化邻域聚合；
2. 将“自身历史”与“邻域聚合历史”拼接后输入 GRU；
3. 线性头输出未来多步预测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


def _load_normalized_adjacency(config: Dict[str, Any], num_nodes: int) -> torch.Tensor:
    path = Path(config.get("geo_mask_path", "My/artifacts/geo_mask.npy"))
    if path.exists():
        mask = np.load(path)
        adjacency = np.asarray(mask, dtype=np.float32)
    else:
        adjacency = np.ones((num_nodes, num_nodes), dtype=np.float32)

    adjacency = adjacency + np.eye(num_nodes, dtype=np.float32)
    degree = adjacency.sum(axis=1, keepdims=True)
    degree[degree == 0.0] = 1.0
    normalized = adjacency / degree
    return torch.from_numpy(normalized.astype(np.float32))


class GCNBaseline(nn.Module):
    """基于图卷积聚合的时间序列预测模型。"""

    def __init__(self, config: Dict[str, Any], num_nodes: int) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_pred = int(config.get("num_pred", 12))
        self.history_len = int(config.get("history_len", 12))

        hidden_size = int(config.get("gcn_hidden_size", 64))
        num_layers = int(config.get("gcn_num_layers", 2))
        dropout = float(config.get("gcn_dropout", 0.1))

        adjacency = _load_normalized_adjacency(config, self.num_nodes)
        self.register_buffer("adjacency", adjacency)

        self.temporal_model = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_pred),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        参数:
            X: [B, history_len, N] 或 [B, history_len, N, 1]
        返回:
            preds: [B, num_pred, N]
        """
        if X.dim() == 4:
            X = X.squeeze(-1)
        if X.dim() != 3:
            raise ValueError("输入 X 必须为 [B, history_len, N] 或 [B, history_len, N, 1]")

        batch, history_len, num_nodes = X.shape
        if history_len != self.history_len:
            raise ValueError(
                f"history_len={history_len} 与模型初始化时设置的 {self.history_len} 不一致"
            )
        if num_nodes != self.num_nodes:
            raise ValueError(f"节点数量 {num_nodes} 与初始化设置的 {self.num_nodes} 不一致")

        adjacency = self.adjacency.to(X.device)
        neighbor_agg = torch.einsum("ij,bpj->bpi", adjacency, X)
        features = torch.stack([X, neighbor_agg], dim=-1)  # [B, history_len, N, 2]
        sequence = features.reshape(batch * num_nodes, history_len, 2)

        _, hidden = self.temporal_model(sequence)
        last_hidden = hidden[-1]  # [B*N, hidden_size]
        preds = self.head(last_hidden).view(batch, num_nodes, self.num_pred).transpose(1, 2)
        return preds.contiguous()


__all__ = ["GCNBaseline"]

