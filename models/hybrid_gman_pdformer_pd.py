"""
GMAN_PDFusion：融合地理/语义注意力与 PDFormer 风格 DFT 的 GMAN 变体。

保持原 GMAN 输入输出接口：
- forward(X, TE) 接收 [B, P, N]（或 [B,P,N,1]）与时间嵌入 [B, P+Q, 2]
- 输出 [B, Q, N]

核心改动：
1. SpatialAttention 替换为 HybridSpatialAttention，按 geo_ratio 拆分头数，
   地理分支叠加 DTW 掩码与 DFT 模式记忆，语义分支按 DTW Top-K 掩码。
2. DFT 分支使用训练得到的 pattern_keys（My/utils/patterns.py），在地理头
   上显式建模传播延迟。
3. 自动加载/回退 geo_mask、sem_mask、pattern_keys，便于在训练脚手架中
   直接实例化。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gman_components import FC, STEmbedding, temporalAttention, gatedFusion, transformAttention


def _init_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _load_npy(path: Optional[str | Path], fallback: torch.Tensor) -> torch.Tensor:
    if path is None:
        return fallback
    p = Path(path)
    if not p.exists():
        return fallback
    array = np.load(p)
    return torch.from_numpy(array).to(fallback.dtype)


def _extract_scalar_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """从 [B, T, N] 序列提取滑窗，返回 [B, T, N, window]。"""
    if window <= 1:
        return x.unsqueeze(-1)
    pad = torch.zeros(x.size(0), window - 1, x.size(2), device=x.device, dtype=x.dtype)
    padded = torch.cat([pad, x], dim=1)
    windows = padded.unfold(dimension=1, size=window, step=1)
    return windows


class HybridSpatialAttention(nn.Module):
    """混合空间注意力，含地理/语义并行分支与 DFT 模块。"""

    def __init__(
        self,
        K: int,
        d: int,
        bn_decay: float,
        *,
        geo_ratio: float,
        window: int,
        pattern_keys: Optional[torch.Tensor] = None,
        geo_mask: Optional[torch.Tensor] = None,
        sem_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.K_total = K
        self.d = d
        self.D = K * d
        self.window = window
        self.geo_ratio = float(geo_ratio)
        self.K_geo = max(1, int(round(self.K_total * self.geo_ratio))) if self.K_total > 0 else 0
        self.K_geo = min(self.K_geo, self.K_total)
        self.K_sem = self.K_total - self.K_geo
        self.dft_dim = self.K_geo * self.d
        self.chunk_size = max(1, int(chunk_size))

        self.FC_q = FC(input_dims=2 * self.D, units=self.D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * self.D, units=self.D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * self.D, units=self.D, activations=F.relu, bn_decay=bn_decay)
        self.FC_out = FC(input_dims=self.D, units=self.D, activations=F.relu, bn_decay=bn_decay)

        if self.dft_dim > 0:
            self.scalar_proj = nn.Linear(self.D, 1, bias=False)
            _init_linear(self.scalar_proj)
            self.window_query = nn.Linear(self.window, self.dft_dim, bias=False)
            _init_linear(self.window_query)
            self.pattern_memory = nn.Linear(self.window, self.dft_dim, bias=False)
            _init_linear(self.pattern_memory)
            self.pattern_value = nn.Linear(self.window, self.dft_dim, bias=False)
            _init_linear(self.pattern_value)
        else:
            self.scalar_proj = None
            self.window_query = None
            self.pattern_memory = None
            self.pattern_value = None

        self.register_buffer("pattern_keys", None, persistent=False)
        self.register_buffer("geo_mask", None, persistent=False)
        self.register_buffer("sem_mask", None, persistent=False)
        self.register_buffer("geo_neighbors_indices", None, persistent=False)
        self.register_buffer("geo_neighbors_valid", None, persistent=False)
        self.register_buffer("sem_neighbors_indices", None, persistent=False)
        self.register_buffer("sem_neighbors_valid", None, persistent=False)
        self.update_pattern_keys(pattern_keys)
        self.update_geo_mask(geo_mask)
        self.update_semantic_mask(sem_mask)

        self.last_geo_output: Optional[torch.Tensor] = None
        self.last_sem_output: Optional[torch.Tensor] = None

    def update_pattern_keys(self, pattern_keys: Optional[torch.Tensor]) -> None:
        if pattern_keys is None:
            self.pattern_keys = None
        else:
            self.pattern_keys = pattern_keys.detach().clone()

    def update_geo_mask(self, geo_mask: Optional[torch.Tensor]) -> None:
        if geo_mask is None:
            self.geo_mask = None
        else:
            gm = geo_mask.to(dtype=torch.bool)
            while gm.dim() > 2 and gm.shape[0] == 1:
                gm = gm.squeeze(0)
            if gm.dim() != 2:
                raise ValueError("geo_mask 必须最终为二维 [N, N]")
            self.geo_mask = gm

    def update_semantic_mask(self, sem_mask: Optional[torch.Tensor]) -> None:
        if sem_mask is None:
            self.sem_mask = None
        else:
            sm = sem_mask.to(dtype=torch.bool)
            while sm.dim() > 2 and sm.shape[0] == 1:
                sm = sm.squeeze(0)
            if sm.dim() != 2:
                raise ValueError("sem_mask 必须最终为二维 [N, N]")
            self.sem_mask = sm

    def update_geo_neighbors(
        self,
        indices: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> None:
        if indices is None or valid is None:
            self.geo_neighbors_indices = None
            self.geo_neighbors_valid = None
            return
        if indices.dim() != 2 or valid.shape != indices.shape:
            raise ValueError("地理邻居索引与有效标记需为 [N, M] 且形状一致。")
        self.geo_neighbors_indices = indices.long()
        self.geo_neighbors_valid = valid.to(dtype=torch.bool)

    def update_semantic_neighbors(
        self,
        indices: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> None:
        if indices is None or valid is None:
            self.sem_neighbors_indices = None
            self.sem_neighbors_valid = None
            return
        if indices.dim() != 2 or valid.shape != indices.shape:
            raise ValueError("语义邻居索引与有效标记需为 [N, M] 且形状一致。")
        self.sem_neighbors_indices = indices.long()
        self.sem_neighbors_valid = valid.to(dtype=torch.bool)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = tensor.shape
        return tensor.view(B, T, N, self.K_total, self.d).permute(0, 3, 1, 2, 4).contiguous()

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        B, K, T, N, d = tensor.shape
        return tensor.permute(0, 2, 3, 1, 4).contiguous().view(B, T, N, K * d)

    def _attention_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, K_head, T, N, d = query.shape
        chunk = min(self.chunk_size, N)
        outputs = []
        key_t = key.transpose(-1, -2)  # [B,K,T,d,N]
        if mask is not None:
            mask = mask.to(query.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            mask = mask.to(torch.bool)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            q_chunk = query[..., start:end, :]  # [B,K,T,C,d]
            scores = torch.matmul(q_chunk, key_t) / math.sqrt(d)  # [B,K,T,C,N]
            if mask is not None:
                mask_chunk = mask[..., start:end, :]  # [1,1,C,N]
                mask_chunk = mask_chunk.expand(B, K_head, T, end - start, N)
                scores = scores.masked_fill(~mask_chunk, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            out_chunk = torch.matmul(attn, value)  # [B,K,T,C,d]
            outputs.append(out_chunk)
        return torch.cat(outputs, dim=3)

    def _attention_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        B, K_head, T, N, d = query.shape
        device = query.device
        indices = indices.to(device=device, dtype=torch.long)
        valid = valid.to(device=device, dtype=torch.bool)
        max_neighbors = indices.shape[-1]

        idx = indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, K_head, T, N, max_neighbors)
        mask = valid.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(idx)

        idx_clamped = idx.clamp(min=0)
        gather_idx = idx_clamped.unsqueeze(-1).expand(B, K_head, T, N, max_neighbors, d)

        key_neighbors = torch.gather(
            key.unsqueeze(4).expand(-1, -1, -1, -1, max_neighbors, -1),
            3,
            gather_idx,
        )
        value_neighbors = torch.gather(
            value.unsqueeze(4).expand(-1, -1, -1, -1, max_neighbors, -1),
            3,
            gather_idx,
        )

        scores = (query.unsqueeze(-2) * key_neighbors).sum(-1) / math.sqrt(d)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = attn.masked_fill(~mask, 0.0)

        output = torch.sum(attn.unsqueeze(-1) * value_neighbors, dim=-2)
        return output

    def forward(
        self,
        X: torch.Tensor,
        STE: torch.Tensor,
        *,
        geo_mask: Optional[torch.Tensor] = None,
        sem_mask: Optional[torch.Tensor] = None,
        pattern_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, N, _ = X.shape
        X_cat = torch.cat((X, STE), dim=-1)
        query = self._split_heads(self.FC_q(X_cat))
        key = self._split_heads(self.FC_k(X_cat))
        value = self._split_heads(self.FC_v(X_cat))

        geo_outputs: Optional[torch.Tensor] = None
        sem_outputs: Optional[torch.Tensor] = None

        active_geo_mask = geo_mask if geo_mask is not None else self.geo_mask
        active_sem_mask = sem_mask if sem_mask is not None else self.sem_mask
        active_patterns = pattern_keys if pattern_keys is not None else self.pattern_keys

        if self.K_geo > 0:
            key_geo = key[:, : self.K_geo]
            query_geo = query[:, : self.K_geo]
            value_geo = value[:, : self.K_geo]
            geo_mask_ready = None
            if active_geo_mask is not None:
                geo_mask_ready = active_geo_mask.to(query_geo.device)
                while geo_mask_ready.dim() > 2 and geo_mask_ready.shape[0] == 1:
                    geo_mask_ready = geo_mask_ready.squeeze(0)
                if geo_mask_ready.dim() != 2 or geo_mask_ready.shape[-2:] != (N, N):
                    raise ValueError("geo_mask 必须为 [..., N, N]")
                geo_mask_ready = geo_mask_ready.to(torch.bool)

            if self.dft_dim > 0 and active_patterns is not None:
                patterns = active_patterns.to(query_geo.device)
                patterns = patterns.permute(2, 0, 1).contiguous()  # [N, num_patterns, window]
                pattern_memory = self.pattern_memory(patterns)
                pattern_value = self.pattern_value(patterns)

                scalar = self.scalar_proj(X).squeeze(-1)
                windows = _extract_scalar_windows(scalar, self.window)
                u_proj = self.window_query(windows)
                scores = torch.einsum("btnf,ncf->btnc", u_proj, pattern_memory)
                weights = torch.softmax(scores, dim=-1)
                weights = torch.nan_to_num(weights, nan=0.0)
                df_flat = torch.einsum("btnc,ncf->btnf", weights, pattern_value)
                df_context = df_flat.view(B, T, N, self.K_geo, self.d)
                df_context = df_context.permute(0, 3, 1, 2, 4).contiguous()
                key_geo = key_geo + df_context

            if (
                self.geo_neighbors_indices is not None
                and self.geo_neighbors_valid is not None
            ):
                geo_context = self._attention_sparse(
                    query_geo,
                    key_geo,
                    value_geo,
                    self.geo_neighbors_indices,
                    self.geo_neighbors_valid,
                )
            else:
                geo_context = self._attention_dense(query_geo, key_geo, value_geo, geo_mask_ready)
            self.last_geo_output = geo_context.detach()
            geo_outputs = geo_context
        else:
            self.last_geo_output = None

        if self.K_sem > 0:
            query_sem = query[:, -self.K_sem :]
            key_sem = key[:, -self.K_sem :]
            value_sem = value[:, -self.K_sem :]
            sem_mask_ready = None
            if active_sem_mask is not None:
                sem_mask_ready = active_sem_mask.to(query_sem.device)
                while sem_mask_ready.dim() > 2 and sem_mask_ready.shape[0] == 1:
                    sem_mask_ready = sem_mask_ready.squeeze(0)
                if sem_mask_ready.dim() != 2 or sem_mask_ready.shape[-2:] != (N, N):
                    raise ValueError("sem_mask 必须为 [..., N, N]")
                sem_mask_ready = sem_mask_ready.to(torch.bool)
            if (
                self.sem_neighbors_indices is not None
                and self.sem_neighbors_valid is not None
            ):
                sem_context = self._attention_sparse(
                    query_sem,
                    key_sem,
                    value_sem,
                    self.sem_neighbors_indices,
                    self.sem_neighbors_valid,
                )
            else:
                sem_context = self._attention_dense(query_sem, key_sem, value_sem, sem_mask_ready)
            self.last_sem_output = sem_context.detach()
            sem_outputs = sem_context
        else:
            self.last_sem_output = None

        outputs = []
        if geo_outputs is not None:
            outputs.append(self._merge_heads(geo_outputs))
        if sem_outputs is not None:
            outputs.append(self._merge_heads(sem_outputs))
        spatial = torch.cat(outputs, dim=-1)
        return self.FC_out(spatial)


class HybridSTAttBlock(nn.Module):
    def __init__(
        self,
        K: int,
        d: int,
        bn_decay: float,
        *,
        geo_ratio: float,
        window: int,
        pattern_keys: Optional[torch.Tensor] = None,
        geo_mask: Optional[torch.Tensor] = None,
        sem_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        chunk_size: int = 64,
        geo_neighbors: Optional[torch.Tensor] = None,
        geo_neighbors_valid: Optional[torch.Tensor] = None,
        sem_neighbors: Optional[torch.Tensor] = None,
        sem_neighbors_valid: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.spatialAttention = HybridSpatialAttention(
            K,
            d,
            bn_decay,
            geo_ratio=geo_ratio,
            window=window,
            pattern_keys=pattern_keys,
            geo_mask=geo_mask,
            sem_mask=sem_mask,
            chunk_size=chunk_size,
        )
        self.spatialAttention.update_geo_neighbors(geo_neighbors, geo_neighbors_valid)
        self.spatialAttention.update_semantic_neighbors(sem_neighbors, sem_neighbors_valid)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)
        self.last_gate_z: Optional[torch.Tensor] = None

    def forward(
        self,
        X: torch.Tensor,
        STE: torch.Tensor,
        *,
        geo_mask: Optional[torch.Tensor] = None,
        sem_mask: Optional[torch.Tensor] = None,
        pattern_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        HS = self.spatialAttention(
            X,
            STE,
            geo_mask=geo_mask,
            sem_mask=sem_mask,
            pattern_keys=pattern_keys,
        )
        HT = self.temporalAttention(X, STE)
        XS = self.gatedFusion.FC_xs(HS)
        XT = self.gatedFusion.FC_xt(HT)
        z = torch.sigmoid(XS + XT)
        H = self.gatedFusion.FC_h(z * HS + (1 - z) * HT)
        self.last_gate_z = z.detach()
        return X + H


class GMANPDFusion(nn.Module):
    def __init__(self, config: Dict[str, Any], num_nodes: int) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.num_pred = int(config.get("num_pred", 12))
        self.num_his = int(config.get("history_len", config.get("num_his", 12)))

        self.L = int(config.get("L", 3))
        self.K = int(config.get("K", 8))
        self.d = int(config.get("d", 8))
        self.D = self.K * self.d
        self.bn_decay = float(config.get("bn_decay", 0.1))
        self.geo_ratio = float(config.get("geo_ratio", 0.6))
        self.window = int(config.get("S", 6))
        self.num_patterns = int(config.get("num_patterns", 16))
        self.time_steps_per_day = int(config.get("time_steps_per_day", 24))
        self.spatial_chunk_size = int(config.get("spatial_chunk_size", 64))

        pattern_path = config.get("pattern_path", "My/artifacts/pattern_keys.npy")
        geo_mask_path = config.get("geo_mask_path")
        sem_mask_path = config.get("sem_mask_path")
        geo_neighbors_path = config.get("geo_neighbors_path")
        sem_neighbors_path = config.get("sem_neighbors_path")

        default_patterns = torch.zeros(self.num_patterns, self.window, num_nodes, dtype=torch.float32)
        patterns = _load_npy(pattern_path, default_patterns)
        if patterns.shape != (self.num_patterns, self.window, num_nodes):
            raise ValueError("pattern_keys 形状与配置不匹配")
        self.register_buffer("pattern_keys", patterns, persistent=False)

        default_geo = torch.eye(num_nodes, dtype=torch.bool)
        geo_mask = _load_npy(geo_mask_path, default_geo.to(torch.float32)).to(torch.bool)
        self.register_buffer("geo_mask", geo_mask, persistent=False)

        default_sem = torch.eye(num_nodes, dtype=torch.bool)
        sem_mask = _load_npy(sem_mask_path, default_sem.to(torch.float32)).to(torch.bool)
        self.register_buffer("sem_mask", sem_mask, persistent=False)

        if geo_neighbors_path is not None and Path(geo_neighbors_path).exists():
            geo_neighbors = np.load(geo_neighbors_path)
            geo_idx = torch.from_numpy(geo_neighbors["indices"].astype(np.int64))
            geo_valid = torch.from_numpy(geo_neighbors["valid"].astype(bool))
            self.register_buffer("geo_neighbor_indices", geo_idx, persistent=False)
            self.register_buffer("geo_neighbor_valid", geo_valid, persistent=False)
        else:
            self.geo_neighbor_indices = None
            self.geo_neighbor_valid = None

        if sem_neighbors_path is not None and Path(sem_neighbors_path).exists():
            sem_neighbors = np.load(sem_neighbors_path)
            sem_idx = torch.from_numpy(sem_neighbors["indices"].astype(np.int64))
            sem_valid = torch.from_numpy(sem_neighbors["valid"].astype(bool))
            self.register_buffer("sem_neighbor_indices", sem_idx, persistent=False)
            self.register_buffer("sem_neighbor_valid", sem_valid, persistent=False)
        else:
            self.sem_neighbor_indices = None
            self.sem_neighbor_valid = None

        self.SE = nn.Parameter(torch.randn(num_nodes, self.D))
        nn.init.xavier_uniform_(self.SE)

        self.STEmbedding = STEmbedding(self.D, self.bn_decay, time_steps_per_day=self.time_steps_per_day)
        self.encoder_blocks = nn.ModuleList(
            [
                HybridSTAttBlock(
                    self.K,
                    self.d,
                    self.bn_decay,
                    geo_ratio=self.geo_ratio,
                    window=self.window,
                    pattern_keys=self.pattern_keys,
                    chunk_size=self.spatial_chunk_size,
                    geo_neighbors=getattr(self, "geo_neighbor_indices", None),
                    geo_neighbors_valid=getattr(self, "geo_neighbor_valid", None),
                    sem_neighbors=getattr(self, "sem_neighbor_indices", None),
                    sem_neighbors_valid=getattr(self, "sem_neighbor_valid", None),
                )
                for _ in range(self.L)
            ]
        )
        self.transformAttention = transformAttention(self.K, self.d, self.bn_decay)
        self.decoder_blocks = nn.ModuleList(
            [
                HybridSTAttBlock(
                    self.K,
                    self.d,
                    self.bn_decay,
                    geo_ratio=self.geo_ratio,
                    window=self.window,
                    pattern_keys=self.pattern_keys,
                    chunk_size=self.spatial_chunk_size,
                    geo_neighbors=getattr(self, "geo_neighbor_indices", None),
                    geo_neighbors_valid=getattr(self, "geo_neighbor_valid", None),
                    sem_neighbors=getattr(self, "sem_neighbor_indices", None),
                    sem_neighbors_valid=getattr(self, "sem_neighbor_valid", None),
                )
                for _ in range(self.L)
            ]
        )
        self.FC_1 = FC(input_dims=[1, self.D], units=[self.D, self.D], activations=[F.relu, None], bn_decay=self.bn_decay)
        self.FC_2 = FC(input_dims=[self.D, self.D], units=[self.D, 1], activations=[F.relu, None], bn_decay=self.bn_decay)

    def forward(
        self,
        X: torch.Tensor,
        TE: torch.Tensor,
        *,
        geo_mask: Optional[torch.Tensor] = None,
        sem_mask: Optional[torch.Tensor] = None,
        pattern_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if X.dim() == 3:
            X = X.unsqueeze(-1)
        elif X.dim() != 4 or X.size(-1) != 1:
            raise ValueError("输入 X 需为 [B, P, N] 或 [B, P, N, 1]")

        pattern_tensor = pattern_keys if pattern_keys is not None else self.pattern_keys
        geo_mask_tensor = geo_mask if geo_mask is not None else self.geo_mask
        sem_mask_tensor = sem_mask if sem_mask is not None else self.sem_mask

        X = self.FC_1(X)
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, : self.num_his]
        STE_pred = STE[:, self.num_his :]

        for block in self.encoder_blocks:
            X = block(X, STE_his, geo_mask=geo_mask_tensor, sem_mask=sem_mask_tensor, pattern_keys=pattern_tensor)

        X = self.transformAttention(X, STE_his, STE_pred)

        for block in self.decoder_blocks:
            X = block(X, STE_pred, geo_mask=geo_mask_tensor, sem_mask=sem_mask_tensor, pattern_keys=pattern_tensor)

        X = self.FC_2(X)
        return X.squeeze(-1)


__all__ = ["GMANPDFusion"]
