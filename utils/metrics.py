"""
常用评估指标骨架。
"""

from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    return (pred - target).abs().mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    return torch.sqrt(((pred - target) ** 2).mean())


def rse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    denom = ((target - target.mean()) ** 2).sum().clamp_min(1e-6)
    num = ((pred - target) ** 2).sum()
    return torch.sqrt(num / denom)


def corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    numerator = (pred_centered * target_centered).sum()
    denom = torch.sqrt(
        (pred_centered ** 2).sum().clamp_min(1e-6)
        * (target_centered ** 2).sum().clamp_min(1e-6)
    )
    return numerator / denom


__all__ = ["mae", "rmse", "rse", "corr"]
