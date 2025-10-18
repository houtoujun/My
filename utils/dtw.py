"""
动态时间规整（DTW）工具集。

提供面向 PM2.5 数据集的 DTW 距离矩阵计算、缓存与加载功能，支持
Sakoe-Chiba 约束、下采样与多特征聚合，并可在命令行下直接生成
`dtw_matrix.npy`。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _haversine_distance(coords: np.ndarray) -> np.ndarray:
    """计算节点间的大圆距离，返回单位为公里的矩阵。"""
    lat = np.deg2rad(coords[:, 0:1])
    lon = np.deg2rad(coords[:, 1:2])
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    earth_radius = 6371.0
    return earth_radius * c


def _validate_sequences(sequences: np.ndarray) -> np.ndarray:
    """确保输入为二维 [num_nodes, time]，多特征情况下进行均值聚合。"""
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    elif arr.ndim != 2:
        raise ValueError("sequences 需为 [num_nodes, time] 或 [num_nodes, time, feat]。")
    if np.isnan(arr).any():
        raise ValueError("sequences 中存在 NaN，请先完成缺失填补。")
    return arr


def _sakoe_chiba_radius(length: int, radius: Optional[int], radius_ratio: float) -> Optional[int]:
    if radius is not None:
        return max(radius, 0)
    if radius_ratio <= 0:
        return None
    return max(int(math.ceil(length * radius_ratio)), 1)


def _dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    radius: Optional[int] = None,
) -> float:
    """
    计算两条序列的 DTW 距离（平均每步成本）。

    参数:
        seq_a/seq_b: 形状 [T] 的序列。
        radius: Sakoe-Chiba 半径（允许的时间偏移窗宽）。为 None 时使用完整矩阵。
    """
    len_a = seq_a.shape[0]
    len_b = seq_b.shape[0]
    if radius is not None:
        radius = max(radius, abs(len_a - len_b))
    else:
        radius = max(len_a, len_b)

    inf = np.float32(np.inf)
    dtw = np.full((len_a + 1, len_b + 1), inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, len_a + 1):
        j_start = max(1, i - radius)
        j_end = min(len_b, i + radius)
        for j in range(j_start, j_end + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    distance = dtw[len_a, len_b]
    norm = (len_a + len_b) / 2.0
    return float(distance / norm)


def compute_dtw_matrix(
    sequences: np.ndarray,
    *,
    radius: Optional[int] = None,
    radius_ratio: float = 0.1,
    downsample: int = 1,
    cache_path: Optional[Path] = None,
    reuse_if_exists: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    计算节点之间的 DTW 距离矩阵。

    参数:
        sequences: [num_nodes, time] 或 [num_nodes, time, feat]，缺失需事先填补。
        radius:    Sakoe-Chiba 半径；若为 None，则按 radius_ratio 自动估计。
        radius_ratio: 以时间长度的比例确定半径，0 表示使用全窗口。
        downsample: 下采样步长（>1 时按时间轴取子序列）。
        cache_path: 若提供，则保存/加载缓存 `.npy` 文件。
        reuse_if_exists: 为 True 且 cache_path 存在时直接加载缓存。
        verbose:   是否打印进度信息。
    """
    sequences = _validate_sequences(sequences)
    if downsample > 1:
        sequences = sequences[:, ::downsample]
    num_nodes, time_len = sequences.shape

    if cache_path is not None and reuse_if_exists and cache_path.exists():
        if verbose:
            print(f"[DTW] 载入缓存: {cache_path}")
        return np.load(cache_path)

    radius_val = _sakoe_chiba_radius(time_len, radius, radius_ratio)
    if verbose:
        print(
            f"[DTW] 计算距离矩阵: nodes={num_nodes}, len={time_len}, "
            f"radius={radius_val}, downsample={downsample}"
        )

    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        matrix[i, i] = 0.0
        seq_i = sequences[i]
        for j in range(i + 1, num_nodes):
            dist = _dtw_distance(seq_i, sequences[j], radius=radius_val)
            matrix[i, j] = matrix[j, i] = dist
        if verbose and (i + 1) % max(1, num_nodes // 10) == 0:
            print(f"[DTW] progress: {i + 1}/{num_nodes}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
    return matrix


def compute_dtw_from_dataset(
    data_dir: Path,
    *,
    feature_idx: int = 0,
    radius: Optional[int] = None,
    radius_ratio: float = 0.1,
    downsample: int = 1,
    cache_path: Optional[Path] = None,
    reuse_if_exists: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取数据集并计算指定特征的 DTW。

    返回:
        dtw_matrix: [N, N] 的距离矩阵。
        sequences:  用于计算的序列（下采样后）。
    """
    from My.utils.data_pm25 import load_raw_pm25_data, fill_missing_with_nanmean

    series, mask, _, _, _ = load_raw_pm25_data(data_dir, feature_idx=feature_idx)
    filled = fill_missing_with_nanmean(series, mask)
    sequences = filled.T  # [N, T]
    if downsample > 1:
        sequences = sequences[:, ::downsample]
    matrix = compute_dtw_matrix(
        sequences,
        radius=radius,
        radius_ratio=radius_ratio,
        downsample=1,
        cache_path=cache_path,
        reuse_if_exists=reuse_if_exists,
        verbose=verbose,
    )
    return matrix, sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 DTW 距离矩阵")
    parser.add_argument("--data-dir", type=Path, default=Path("My/dataset"))
    parser.add_argument("--feature-idx", type=int, default=0, help="使用的特征索引")
    parser.add_argument("--radius", type=int, default=None, help="Sakoe-Chiba 半径")
    parser.add_argument("--radius-ratio", type=float, default=0.1, help="半径比例")
    parser.add_argument("--downsample", type=int, default=6, help="时间下采样步长")
    parser.add_argument("--cache-path", type=Path, default=Path("My/artifacts/dtw_matrix.npy"))
    args = parser.parse_args()

    matrix, _ = compute_dtw_from_dataset(
        args.data_dir,
        feature_idx=args.feature_idx,
        radius=args.radius,
        radius_ratio=args.radius_ratio,
        downsample=args.downsample,
        cache_path=args.cache_path,
        reuse_if_exists=False,
        verbose=True,
    )
    print(f"[DTW] 完成，矩阵形状 {matrix.shape}，已保存至 {args.cache_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "compute_dtw_matrix",
    "compute_dtw_from_dataset",
    "_haversine_distance",
]
