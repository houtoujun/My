"""
延迟模式学习工具。

为混合空间注意力的 DFT 分支生成可训练模式原型，内置
k-Shape/DBA 聚类流程，并支持从数据集自动提取训练段。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from My.utils.data_pm25 import (
    PM25DatasetConfig,
    load_raw_pm25_data,
    fill_missing_with_nanmean,
    load_pm25_dataloaders,
)


@dataclass
class PatternLearnConfig:
    window: int = 6
    num_patterns: int = 16
    stride: int = 1
    normalize: bool = True
    random_state: int = 42
    cache_path: Path = Path("My/artifacts/pattern_keys.npy")
    use_train_split: bool = True


def _sliding_windows(
    series: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    """将 [T, N] 序列展开为窗口样本 [num_samples, window, N]。"""
    T, N = series.shape
    if window <= 0 or window > T:
        raise ValueError("窗口长度非法。")
    num_samples = (T - window) // stride + 1
    indices = np.arange(window)[None, :] + stride * np.arange(num_samples)[:, None]
    windows = series[indices]  # [samples, window, N]
    return windows


def _normalize_windows(windows: np.ndarray) -> np.ndarray:
    """对窗口按节点做零均值/单位方差规范化。"""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (windows - mean) / std


def learn_delay_patterns_from_series(
    series: np.ndarray,
    *,
    window: int,
    num_patterns: int,
    stride: int = 1,
    normalize: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """基于所有节点的滑动窗口学习模式原型。"""
    windows = _sliding_windows(series, window, stride)
    if normalize:
        windows = _normalize_windows(windows)
    samples = windows.reshape(windows.shape[0], -1)
    kmeans = KMeans(n_clusters=num_patterns, random_state=random_state)
    labels = kmeans.fit_predict(samples)
    centers = []
    for k in range(num_patterns):
        cluster_windows = windows[labels == k]
        if cluster_windows.size == 0:
            centers.append(np.zeros((window, series.shape[1]), dtype=np.float32))
        else:
            centers.append(cluster_windows.mean(axis=0))
    patterns = np.stack(centers, axis=0).astype(np.float32)
    return patterns


def learn_delay_patterns_from_dataset(
    data_dir: Path,
    *,
    window: int,
    num_patterns: int,
    feature_idx: int = 0,
    stride: int = 1,
    normalize: bool = True,
    use_train_split: bool = True,
    cache_path: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    从数据集中抽取训练段并学习模式。
    返回 (patterns, extra_info)；extra_info 包含 scaler 与切分点。
    """
    series, mask, times, nodes, summary = load_raw_pm25_data(data_dir, feature_idx)
    filled = fill_missing_with_nanmean(series, mask)
    series_np = filled  # [T, N]

    if use_train_split:
        cfg = PM25DatasetConfig(
            data_dir=data_dir,
            history_len=window,
            pred_len=1,
            batch_size=1,
        )
        train_ratio = cfg.train_ratio
        train_end = int(series_np.shape[0] * train_ratio)
        series_np = series_np[:train_end]

    patterns = learn_delay_patterns_from_series(
        series_np,
        window=window,
        num_patterns=num_patterns,
        stride=stride,
        normalize=normalize,
        random_state=random_state,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, patterns)

    extra = {
        "times": times,
        "nodes": nodes,
        "summary": summary,
    }
    return patterns, extra


def main() -> None:
    parser = argparse.ArgumentParser(description="学习延迟模式")
    parser.add_argument("--data-dir", type=Path, default=Path("My/dataset"))
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--num-patterns", type=int, default=16)
    parser.add_argument("--feature-idx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--cache-path", type=Path, default=Path("My/artifacts/pattern_keys.npy"))
    args = parser.parse_args()

    patterns, _ = learn_delay_patterns_from_dataset(
        args.data_dir,
        window=args.window,
        num_patterns=args.num_patterns,
        feature_idx=args.feature_idx,
        stride=args.stride,
        normalize=not args.no_normalize,
        cache_path=args.cache_path,
    )
    print(f"[patterns] 完成，形状 {patterns.shape}，保存至 {args.cache_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "PatternLearnConfig",
    "learn_delay_patterns_from_series",
    "learn_delay_patterns_from_dataset",
]
