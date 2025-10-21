"""
基于循环神经网络（GRU、LSTM）的多步预测基线模型。

设计要点：
- 对每个站点的时间序列独立建模，但共享同一组 RNN 参数；
- 输入格式遵循数据加载器输出的 [B, history_len, N]；
- 最终输出 [B, num_pred, N]，方便与现有评估流程对接。
"""

from __future__ import annotations

from typing import Any, Dict, Type

import torch
import torch.nn as nn


class _SequenceRNNTrajectory(nn.Module):
    """将单变量序列映射为多步预测的基础 RNN 模型。"""

    def __init__(
        self,
        *,
        rnn_cls: Type[nn.Module],
        config: Dict[str, Any],
        num_nodes: int,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.history_len = int(config.get("history_len", 12))
        self.num_pred = int(config.get("num_pred", 12))

        hidden_size = int(config.get("hidden_size", 64))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1))

        self.rnn_type = rnn_cls.__name__.upper()
        self.rnn = rnn_cls(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
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

        sequence = X.transpose(1, 2).reshape(batch * num_nodes, history_len, 1)
        out, hidden = self.rnn(sequence)

        if isinstance(self.rnn, nn.LSTM):
            last_hidden = hidden[0][-1]  # LSTM 返回 (h, c)
        else:
            last_hidden = hidden[-1]

        preds = self.proj(last_hidden)  # [B*N, num_pred]
        preds = preds.view(batch, num_nodes, self.num_pred).transpose(1, 2)
        return preds.contiguous()


class GRUBaseline(_SequenceRNNTrajectory):
    """使用 GRU 的序列到序列基线模型。"""

    def __init__(self, config: Dict[str, Any], num_nodes: int) -> None:
        super().__init__(rnn_cls=nn.GRU, config=config, num_nodes=num_nodes)


class LSTMBaseline(_SequenceRNNTrajectory):
    """使用 LSTM 的序列到序列基线模型。"""

    def __init__(self, config: Dict[str, Any], num_nodes: int) -> None:
        super().__init__(rnn_cls=nn.LSTM, config=config, num_nodes=num_nodes)


__all__ = ["GRUBaseline", "LSTMBaseline"]

