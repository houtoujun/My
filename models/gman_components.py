"""
GMAN 通用组件。

包含 GMAN_PDFusion 所需的基础模块：conv2d_, FC, STEmbedding,
temporalAttention, gatedFusion, transformAttention。

这些实现移植自 GMAN 官方仓库并结合当前项目对时间步长的改动。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2d_(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_size,
        stride=(1, 1),
        padding="SAME",
        use_bias: bool = True,
        activation=F.relu,
        bn_decay: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.activation = activation
        if padding == "SAME":
            if isinstance(kernel_size, (list, tuple)):
                self.padding_size = [k // 2 for k in kernel_size]
            else:
                self.padding_size = [kernel_size // 2, kernel_size // 2]
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        momentum = bn_decay if bn_decay is not None else 0.1
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=momentum)
        nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)
        pad = [
            self.padding_size[1],
            self.padding_size[1],
            self.padding_size[0],
            self.padding_size[0],
        ]
        x = F.pad(x, pad)
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(
        self,
        input_dims,
        units,
        activations,
        bn_decay: Optional[float],
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert isinstance(units, list)
        self.convs = nn.ModuleList(
            [
                conv2d_(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    padding="VALID",
                    use_bias=use_bias,
                    activation=activation,
                    bn_decay=bn_decay,
                )
                for input_dim, num_unit, activation in zip(input_dims, units, activations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    """
    时空嵌入模块。

    输入:
        SE: [num_vertex, D]
        TE: [batch_size, num_his + num_pred, 2] (day_of_week, time_of_day)
    输出:
        [batch_size, num_his + num_pred, num_vertex, D]
    """

    def __init__(self, D: int, bn_decay: float, time_steps_per_day: int = 288) -> None:
        super().__init__()
        self.time_steps_per_day = time_steps_per_day
        self.FC_se = FC(
            input_dims=[D, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
        )
        te_input_dim = 7 + self.time_steps_per_day
        self.FC_te = FC(
            input_dims=[te_input_dim, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
        )

    def forward(self, SE: torch.Tensor, TE: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        if T is None:
            T = self.time_steps_per_day
        # Spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        # Temporal embedding
        device = TE.device
        day_idx = (TE[..., 0].to(torch.int64) % 7).clamp(min=0)
        time_idx = (TE[..., 1].to(torch.int64) % T).clamp(min=0)
        dayofweek = F.one_hot(day_idx, num_classes=7).to(torch.float32).to(device)
        timeofday = F.one_hot(time_idx, num_classes=T).to(torch.float32).to(device)
        TE_cat = torch.cat((dayofweek, timeofday), dim=-1).unsqueeze(2)
        TE_embed = self.FC_te(TE_cat)
        return SE + TE_embed


class temporalAttention(nn.Module):
    """
    Temporal Attention 机制。
    输入:
        X:   [batch_size, num_step, num_vertex, D]
        STE: [batch_size, num_step, num_vertex, D]
    输出:
        [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool = True) -> None:
        super().__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X: torch.Tensor, STE: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[0]
        X_cat = torch.cat((X, STE), dim=-1)
        query = self.FC_q(X_cat)
        key = self.FC_k(X_cat)
        value = self.FC_v(X_cat)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) / math.sqrt(self.d)
        if self.mask:
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step, device=X.device, dtype=torch.bool).tril()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            attention = attention.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(attention, dim=-1)
        X_ret = torch.matmul(attention, value)
        X_ret = X_ret.permute(0, 2, 1, 3)
        X_ret = torch.cat(torch.split(X_ret, batch_size, dim=0), dim=-1)
        X_ret = self.FC(X_ret)
        return X_ret


class gatedFusion(nn.Module):
    """
    空间 / 时间特征门控融合。
    输入:
        HS, HT: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, D: int, bn_decay: float) -> None:
        super().__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(
            input_dims=[D, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
        )

    def forward(self, HS: torch.Tensor, HT: torch.Tensor) -> torch.Tensor:
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(XS + XT)
        H = self.FC_h(z * HS + (1 - z) * HT)
        return H


class transformAttention(nn.Module):
    """
    Transform Attention 机制。
    输入:
        X:        [batch_size, num_his, num_vertex, D]
        STE_his:  [batch_size, num_his, num_vertex, D]
        STE_pred: [batch_size, num_pred, num_vertex, D]
    输出:
        [batch_size, num_pred, num_vertex, D]
    """

    def __init__(self, K: int, d: int, bn_decay: float) -> None:
        super().__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X: torch.Tensor, STE_his: torch.Tensor, STE_pred: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[0]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) / math.sqrt(self.d)
        attention = torch.softmax(attention, dim=-1)
        X_ret = torch.matmul(attention, value)
        X_ret = X_ret.permute(0, 2, 1, 3)
        X_ret = torch.cat(torch.split(X_ret, batch_size, dim=0), dim=-1)
        X_ret = self.FC(X_ret)
        return X_ret


__all__ = [
    "FC",
    "STEmbedding",
    "temporalAttention",
    "gatedFusion",
    "transformAttention",
]
