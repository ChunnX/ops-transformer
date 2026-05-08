from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _ifa_ref_bnsd_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    actual_kv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """BNSD 下单步 decode（S_q=1）的参考实现：按 actual_kv 截断 KV，其余位置 mask 为 -inf。"""
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    B, N, _, D = q.shape
    _, _, S2, _ = k.shape
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    for b in range(B):
        L = int(actual_kv[b].item())
        if L < S2:
            scores[b, :, :, L:] = float("-inf")
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v32).to(dtype=q.dtype)
    return out


class Model(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, scale: float, layout: str):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.layout = layout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_out: torch.Tensor,
        actual_seq: torch.Tensor,
    ) -> torch.Tensor:
        del attention_out
        return _ifa_ref_bnsd_fp16(query, key, value, actual_seq, self.scale)


class ModelNew(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, scale: float, layout: str):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.layout = layout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_out: torch.Tensor,
        actual_seq: torch.Tensor,
    ) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.incre_flash_attention(
            query,
            key,
            value,
            attention_out,
            actual_seq,
            int(self.num_heads),
            float(self.scale),
            self.layout,
            int(self.num_kv_heads),
        )
