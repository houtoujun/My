"""
地理与语义邻域工具。

用于根据经纬度、DTW 距离等信息生成稀疏注意力所需的掩码或邻居索引。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .dtw import _haversine_distance


def build_geographic_mask(
    *,
    coords: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
    hop_matrix: Optional[np.ndarray] = None,
    threshold: float = 200.0,
    use_hop: bool = False,
    include_self: bool = True,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    构建地理邻接掩码。

    参数:
        coords: [N, 2] 经纬度，若未提供 distance_matrix 则必填。
        distance_matrix: 预先计算的距离矩阵（公里）。
        hop_matrix: 节点跳数矩阵，use_hop=True 时使用。
        threshold: 距离（千米）或跳数上限。
        use_hop: True 则使用 hop_matrix，False 使用距离。
        include_self: 是否保留节点自身。
        cache_path: 若提供，则保存掩码到该路径。
    """
    if use_hop:
        if hop_matrix is None:
            raise ValueError("use_hop=True 时需提供 hop_matrix。")
        mask = np.asarray(hop_matrix <= float(threshold), dtype=bool)
    else:
        if distance_matrix is None:
            if coords is None:
                raise ValueError("构建距离掩码时需提供 coords 或 distance_matrix。")
            distance_matrix = _haversine_distance(coords.astype(np.float64))
        mask = np.asarray(distance_matrix <= float(threshold), dtype=bool)

    if include_self:
        np.fill_diagonal(mask, True)
    else:
        np.fill_diagonal(mask, False)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mask)
    return mask


def build_semantic_mask(
    dtw_matrix: np.ndarray,
    *,
    topk: int = 8,
    include_self: bool = True,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    基于 DTW 距离构建语义邻居掩码（选取距离最小的若干节点）。
    """
    distances = np.asarray(dtw_matrix, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("dtw_matrix 需为 [N, N] 方阵。")
    num_nodes = distances.shape[0]
    mask = np.zeros_like(distances, dtype=bool)
    if topk >= num_nodes:
        mask[:] = True
    else:
        indices = np.argpartition(distances, kth=topk, axis=1)[:, :topk]
        row_indices = np.arange(num_nodes)[:, None]
        mask[row_indices, indices] = True

    if include_self:
        np.fill_diagonal(mask, True)
    else:
        np.fill_diagonal(mask, False)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mask)
    return mask


def mask_to_neighbor_indices(
    mask: np.ndarray,
    *,
    max_neighbors: Optional[int] = None,
    include_self: bool = True,
    sort_matrix: Optional[np.ndarray] = None,
    descending: bool = False,
    fill_value: int = -1,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将布尔掩码转为邻居索引及有效标记。

    参数:
        mask: [N, N] 布尔掩码。
        max_neighbors: 限制每个节点的最大邻居数；缺省时取所有节点的最大邻居数。
        include_self: 是否确保自身在邻居列表中。
        sort_matrix: [N, N] 排序参考（如距离/相似度），按升序排序；若 descending=True，按降序。
        descending: 是否按降序排列排序结果。
        fill_value: 填充无效位置的值（默认 -1）。
        cache_path: 若提供，则保存 npz 文件（indices、valid）。

    返回:
        indices: [N, max_neighbors]，邻居索引（不足填 fill_value）。
        valid:   同形状布尔矩阵，表示对应位置是否为有效邻居。
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask 必须为 [N, N] 方阵。")
    num_nodes = mask.shape[0]

    if sort_matrix is not None:
        sort_matrix = np.asarray(sort_matrix)
        if sort_matrix.shape != mask.shape:
            raise ValueError("sort_matrix 需与 mask 形状一致。")

    neighbor_lists = []
    max_len = 0
    for i in range(num_nodes):
        neighbors = np.nonzero(mask[i])[0]
        if not include_self:
            neighbors = neighbors[neighbors != i]
        elif neighbors.size == 0:
            neighbors = np.array([i], dtype=np.int64)

        if sort_matrix is not None and neighbors.size > 0:
            scores = sort_matrix[i, neighbors]
            order = np.argsort(scores)
            if descending:
                order = order[::-1]
            neighbors = neighbors[order]

        neighbor_lists.append(neighbors.astype(np.int64, copy=False))
        max_len = max(max_len, neighbors.size)

    if max_neighbors is None:
        max_neighbors = max_len

    indices = np.full((num_nodes, max_neighbors), fill_value, dtype=np.int64)
    valid = np.zeros((num_nodes, max_neighbors), dtype=bool)
    for i, neighbors in enumerate(neighbor_lists):
        length = min(neighbors.size, max_neighbors)
        if length > 0:
            indices[i, :length] = neighbors[:length]
            valid[i, :length] = True

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, indices=indices, valid=valid)

    return indices, valid


__all__ = [
    "build_geographic_mask",
    "build_semantic_mask",
    "mask_to_neighbor_indices",
]
