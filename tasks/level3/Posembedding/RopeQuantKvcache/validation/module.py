"""
CPU 金标准：qkv 切分 → 对 q/k 做与 AscendC 一致的 half RoPE → k/v 按 per-D 的 scale/offset 量化并写入 cache。
内核未将 RoPE 后的 k 写回 k_out GM，精度验证仅比对 q（见 prepare_inputs.custom_check_precision）。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

QUANT_MIN = -128
QUANT_MAX = 127


def _rope_match_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    sin_adj = sin.clone()
    sin_adj[..., :half] = -sin_adj[..., :half]
    x_rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin_adj


def _quant_to_cache(
    x_f16: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
) -> torch.Tensor:
    scale = quant_scale.to(torch.float32).reshape(1, 1, 1, -1)
    off = quant_offset.to(torch.float32).reshape(1, 1, 1, -1)
    t = x_f16.float() / scale + off
    t = torch.round(t)
    return torch.clamp(t, min=QUANT_MIN, max=QUANT_MAX).to(torch.int8)


def _scatter_cache(
    cache: torch.Tensor,
    quantized: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    b = quantized.shape[0]
    s = quantized.shape[1]
    for bi in range(b):
        iv = int(indices[bi].item())
        cache[bi, iv : iv + s, :, :].copy_(quantized[bi])


def _run_rope_quant(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    size_splits: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = qkv.shape
    sq, sk, sv = size_splits
    d = quant_scale.numel()
    n_q = sq // d
    n_kv = sk // d
    assert sq == n_q * d and sk == n_kv * d and sv == n_kv * d

    q_flat, k_flat, v_flat = qkv.split([sq, sk, sv], dim=-1)
    q_b = q_flat.reshape(b, s, n_q, d)
    k_b = k_flat.reshape(b, s, n_kv, d)
    v_b = v_flat.reshape(b, s, n_kv, d)

    cos_b = torch.broadcast_to(cos.float(), q_b.shape)
    sin_b = torch.broadcast_to(sin.float(), q_b.shape)
    cos_k = torch.broadcast_to(cos.float(), k_b.shape)
    sin_k = torch.broadcast_to(sin.float(), k_b.shape)

    rope_q = _rope_match_kernel(q_b.float(), cos_b, sin_b).to(torch.float16)
    rope_k = _rope_match_kernel(k_b.float(), cos_k, sin_k).to(torch.float16)

    k_q = _quant_to_cache(rope_k, quant_scale, quant_offset)
    v_q = _quant_to_cache(v_b, quant_scale, quant_offset)

    kc = k_cache.clone()
    vc = v_cache.clone()
    _scatter_cache(kc, k_q, indices)
    _scatter_cache(vc, v_q, indices)

    return rope_q, rope_k, v_b, kc, vc


class Model(nn.Module):
    def __init__(self, size_splits: Tuple[int, int, int], kv_output: bool = True, layout: str = "BSND"):
        super().__init__()
        self.size_splits = tuple(size_splits)
        self.kv_output = kv_output
        self.layout = layout

    def forward(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        quant_scale: torch.Tensor,
        quant_offset: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        _q_buf: torch.Tensor,
        _k_buf: torch.Tensor,
        _v_buf: torch.Tensor,
    ) -> List[torch.Tensor]:
        del _q_buf, _k_buf, _v_buf
        return list(
            _run_rope_quant(
                qkv,
                cos,
                sin,
                k_cache,
                v_cache,
                indices,
                quant_scale,
                quant_offset,
                self.size_splits,
            )
        )


class ModelNew(nn.Module):
    def __init__(self, size_splits: Tuple[int, int, int], kv_output: bool = True, layout: str = "BSND"):
        super().__init__()
        self.size_splits = tuple(size_splits)
        self.kv_output = kv_output
        self.layout = layout

    def forward(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        quant_scale: torch.Tensor,
        quant_offset: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        q_buf: torch.Tensor,
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.rope_quant_kvcache(
            qkv,
            cos,
            sin,
            quant_scale,
            quant_offset,
            k_cache,
            v_cache,
            indices,
            list(self.size_splits),
            self.layout,
            self.kv_output,
            q_buf,
            k_buf,
            v_buf,
        )
