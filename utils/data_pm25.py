"""
PM2.5 数据加载与切分模块。

读取 `data_merged.npz` 及配套的时间、节点信息，完成缺失填补、标准化、
时间特征编码，并构造窗口化数据集供模型训练/验证/测试使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PM25DatasetConfig:
    """数据加载配置。"""

    data_dir: Path
    history_len: int
    pred_len: int
    batch_size: int
    num_workers: int = 0
    feature_idx: int = 0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    shuffle_train: bool = True
    pin_memory: bool = False
    scaler_path: Optional[Path] = None


class StandardScaler:
    """按节点统计的标准化器。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "StandardScaler":
        return cls(state["mean"], state["std"])


def _load_summary(data_dir: Path) -> Dict[str, Any]:
    summary_path = data_dir / "summary_merged.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_raw_pm25_data(
    data_dir: Path,
    feature_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame, Dict[str, Any]]:
    """
    加载融合数据张量及配套信息。

    返回:
        series: [T, N] 的目标特征序列（PM2.5 等）
        mask:   [T, N] 的缺失布尔掩码（True 表示原始缺失）
        times:  pandas 时间索引
        nodes:  节点信息表（按 node_id 排序）
        summary: summary_merged.json 内容（若存在）
    """
    npz_path = data_dir / "data_merged.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {npz_path}")

    merged = np.load(npz_path)
    x = merged["x"]  # [T, N, F]
    mask = merged["mask"]
    if feature_idx >= x.shape[-1]:
        raise IndexError(f"feature_idx {feature_idx} 超出特征维度 {x.shape[-1]}")

    target = x[..., feature_idx]  # [T, N]
    target_mask = mask[..., feature_idx]

    times_path = data_dir / "times.csv"
    times_series = pd.read_csv(times_path, encoding="utf-8-sig")["datetime"]
    times_series = pd.to_datetime(times_series, utc=False)
    if getattr(times_series.dt, "tz", None) is None:
        times_series = times_series.dt.tz_localize("Asia/Shanghai")
    else:
        times_series = times_series.dt.tz_convert("Asia/Shanghai")
    times = pd.DatetimeIndex(times_series)

    nodes_path = data_dir / "nodes.csv"
    nodes = (
        pd.read_csv(nodes_path, encoding="utf-8-sig")
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    summary = _load_summary(data_dir)
    return target, target_mask, times, nodes, summary


def fill_missing_with_nanmean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    使用按节点的均值填补缺失。

    参数:
        values: [T, N] 数值序列。
        mask:   [T, N] 布尔掩码，True 表示缺失。
    """
    filled = values.astype(np.float32).copy()
    filled = np.where(mask, np.nan, filled)

    node_means = np.nanmean(filled, axis=0, keepdims=True)
    node_means = np.nan_to_num(node_means, nan=float(np.nanmean(filled)))
    nan_indices = np.where(np.isnan(filled))
    if nan_indices[0].size > 0:
        filled[nan_indices] = node_means[0, nan_indices[1]]
    filled = np.nan_to_num(filled, nan=0.0)
    return filled


def build_time_features(times: pd.DatetimeIndex) -> Tuple[np.ndarray, int, int]:
    """
    生成 GMAN 兼容的时间特征，并返回日内步数与时间分辨率（分钟）。
    - day_of_week: 0-6
    - time_of_day: 0-(steps_per_day-1)
    """
    if len(times) < 2:
        raise ValueError("时间序列长度不足以推断采样频率�?")
    deltas = times.to_series().diff().dropna().dt.total_seconds() / 60.0
    freq_minutes = int(round(deltas.mode().iloc[0]))
    if freq_minutes <= 0:
        raise ValueError("无法解析有效的时间分辨率�?")
    steps_per_day = int(round(24 * 60 / freq_minutes))
    day_of_week = times.dayofweek.to_numpy(dtype=np.int16)
    minute_of_day = (times.hour * 60 + times.minute).to_numpy(dtype=np.int32)
    time_of_day = (minute_of_day // freq_minutes).astype(np.int16)
    features = np.stack([day_of_week, time_of_day], axis=-1)
    return features, steps_per_day, freq_minutes


def _compute_split_points(
    total_steps: int,
    history: int,
    pred: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test 比例之和需为 1。")
    train_end = int(total_steps * train_ratio)
    val_end = train_end + int(total_steps * val_ratio)
    val_end = min(val_end, total_steps - pred - 1)
    test_end = total_steps
    if train_end <= history + pred:
        raise ValueError("训练样本过少，请调整窗口或比例。")
    return train_end, val_end, test_end


def _window_indices(
    start: int,
    end: int,
    history: int,
    pred: int,
    allow_past: bool = True,
) -> np.ndarray:
    """生成窗口起始索引，保证历史与预测完全落在 [start, end) 内。"""
    if allow_past:
        start_idx = max(start - history, 0)
    else:
        start_idx = start
    last_start = end - history - pred
    if last_start < start_idx:
        return np.array([], dtype=np.int32)
    return np.arange(start_idx, last_start + 1, dtype=np.int32)


class PM25WindowDataset(Dataset):
    """基于滑动窗口的 PM2.5 数据集。"""

    def __init__(
        self,
        series: np.ndarray,
        mask: np.ndarray,
        te: np.ndarray,
        indices: Sequence[int],
        history: int,
        pred: int,
    ) -> None:
        self.series = series.astype(np.float32)
        self.mask = mask.astype(np.bool_)
        self.te = te.astype(np.int64)
        self.history = history
        self.pred = pred
        self.indices = np.asarray(indices, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = int(self.indices[idx])
        his_end = start + self.history
        pred_end = his_end + self.pred

        X = self.series[start:his_end]  # [P, N]
        y = self.series[his_end:pred_end]  # [Q, N]
        X_mask = self.mask[start:his_end]
        y_mask = self.mask[his_end:pred_end]
        TE = self.te[start:pred_end]  # [P+Q, 2]

        batch = {
            "X": torch.from_numpy(X),  # float32
            "y": torch.from_numpy(y),
            "mask_X": torch.from_numpy(X_mask.astype(np.float32)),
            "mask_y": torch.from_numpy(y_mask.astype(np.float32)),
            "TE": torch.from_numpy(TE),  # int64
        }
        return batch


def load_pm25_dataloaders(
    config: PM25DatasetConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    构建训练/验证/测试数据加载器，并返回元信息（含 scaler、时间、节点等）。
    """
    series_raw, mask, times, nodes, summary = load_raw_pm25_data(
        config.data_dir, config.feature_idx
    )
    series_filled = fill_missing_with_nanmean(series_raw, mask)

    train_end, val_end, test_end = _compute_split_points(
        total_steps=series_filled.shape[0],
        history=config.history_len,
        pred=config.pred_len,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    train_series = series_filled[:train_end]
    mean = np.mean(train_series, axis=0, keepdims=True)
    std = np.std(train_series, axis=0, keepdims=True)
    scaler = StandardScaler(mean, std)
    series_scaled = scaler.transform(series_filled)

    te, steps_per_day, freq_minutes = build_time_features(times)
    te = te.astype(np.int64)

    train_indices = _window_indices(0, train_end, config.history_len, config.pred_len)
    val_indices = _window_indices(train_end, val_end, config.history_len, config.pred_len)
    test_indices = _window_indices(val_end, test_end, config.history_len, config.pred_len)

    if train_indices.size == 0 or val_indices.size == 0 or test_indices.size == 0:
        raise RuntimeError("窗口划分失败，请检查时间长度或比例配置。")

    train_dataset = PM25WindowDataset(
        series_scaled, mask, te, train_indices, config.history_len, config.pred_len
    )
    val_dataset = PM25WindowDataset(
        series_scaled, mask, te, val_indices, config.history_len, config.pred_len
    )
    test_dataset = PM25WindowDataset(
        series_scaled, mask, te, test_indices, config.history_len, config.pred_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    if config.scaler_path is not None:
        config.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(config.scaler_path, mean=scaler.mean, std=scaler.std)

    metadata: Dict[str, Any] = {
        "scaler": scaler,
        "times": times,
        "nodes": nodes,
        "summary": summary,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
        "steps_per_day": steps_per_day,
        "freq_minutes": freq_minutes,
    }
    return train_loader, val_loader, test_loader, metadata


__all__ = [
    "PM25DatasetConfig",
    "PM25WindowDataset",
    "StandardScaler",
    "load_raw_pm25_data",
    "load_pm25_dataloaders",
]
