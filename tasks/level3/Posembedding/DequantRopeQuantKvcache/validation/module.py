"""
CPU 金标准与 ops-transformer `executor_aclnnDequantRopeQuantKvcache` 中 drqk 逻辑对齐：
dequant(可选) → split → RoPE（与内核一致：对 sin 前半取负后与 half-swap 组合）→ 量化并 scatter 到 k/v cache。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

QUANT_MIN = -128
QUANT_MAX = 127


def _rope_match_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """与 AscendC 实现一致：对 sin 前半轴取负，尾维后半与前半交换后与 sin 组合，再与 x*cos 相加（见 op_kernel 中 RoPE 段）。"""
    d = x.shape[-1]
    half = d // 2
    sin_adj = sin.clone()
    sin_adj[..., :half] = -sin_adj[..., :half]
    x_rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin_adj


def _dequant_like_executor(
    inp: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    activation_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    if weight_scale is None:
        return inp
    ws = weight_scale.reshape(1, 1, -1)
    if activation_scale is not None:
        as_cpu = activation_scale.reshape(inp.shape[0], inp.shape[1], 1)
    else:
        as_cpu = None
    if bias is not None:
        bias_cpu = bias.reshape(1, 1, -1)
        if bias.dtype == torch.int32:
            t = torch.add(inp, bias_cpu)
            t = torch.mul(t, ws)
            if as_cpu is not None:
                t = torch.mul(t, as_cpu)
        else:
            t = torch.mul(inp, ws)
            if as_cpu is not None:
                t = torch.mul(t, as_cpu)
            t = torch.add(t, bias_cpu)
    else:
        t = torch.mul(inp, ws)
        if as_cpu is not None:
            t = torch.mul(t, as_cpu)
    return t


def _quant_update_scatter(
    key_cache: torch.Tensor,
    key: torch.Tensor,
    inv_scale: torch.Tensor,
    indice: torch.Tensor,
    offset: Optional[torch.Tensor],
    page_mode: bool,
) -> None:
    scale = inv_scale.reshape(-1, key.shape[-1])
    off = offset.reshape(-1, key.shape[-1]) if offset is not None else None
    if off is not None:
        quant_out = key.float() * scale + off
    else:
        quant_out = key.float() * scale
    quant_out = torch.round(quant_out)
    quant_out1 = torch.clamp(torch.round(quant_out.float()), min=QUANT_MIN, max=QUANT_MAX).to(torch.int8)
    if page_mode:
        d0, d1, d2, d3 = key_cache.shape
        key_cache_pa = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
        quant_out2 = quant_out1.reshape(-1, quant_out1.shape[-2], quant_out1.shape[-1])
        for b in range(indice.shape[0]):
            iv = int(indice[b].item())
            key_cache_pa[iv] = quant_out2[b]
        key_cache.copy_(key_cache_pa.reshape(d0, d1, d2, d3))
    else:
        s_len = quant_out1.shape[1]
        for b in range(indice.shape[0]):
            iv = int(indice[b].item())
            key_cache[b, iv : iv + s_len, :, :].copy_(quant_out1[b])


def _run_drqk(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    size_splits: Sequence[int],
    offset_k: Optional[torch.Tensor],
    offset_v: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    activation_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    cache_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    page_mode = str(cache_mode).lower() == "page"
    x_work = x
    if x.dtype == torch.int32:
        x_work = _dequant_like_executor(x.float(), bias, weight_scale, activation_scale)
        x_work = x_work.to(cos.dtype)
    else:
        x_work = x
    if x_work.dim() == 2:
        x_work = x_work.unsqueeze(1)
    cos_u, sin_u = cos, sin
    if cos_u.dim() == 2:
        cos_u = cos_u.unsqueeze(1).unsqueeze(2)
        sin_u = sin_u.unsqueeze(1).unsqueeze(2)

    b, s, _ = x_work.shape
    h = k_cache.shape[-1]
    q, kt, vt = x_work.split(tuple(size_splits), dim=-1)
    q1 = q.reshape(b, s, -1, h)
    k1 = kt.reshape(b, s, -1, h)
    v1 = vt.reshape(b, s, -1, h)

    ropek = _rope_match_kernel(k1, cos_u, sin_u)
    ropeq = _rope_match_kernel(q1, cos_u, sin_u)

    inv_k = (1.0 / scale_k.to(torch.float32)).to(torch.float32)
    inv_v = (1.0 / scale_v.to(torch.float32)).to(torch.float32)
    kc = k_cache.clone()
    vc = v_cache.clone()
    _quant_update_scatter(kc, ropek, inv_k, indices, offset_k, page_mode)
    _quant_update_scatter(vc, v1, inv_v, indices, offset_v, page_mode)
    return ropeq, ropek, v1, kc, vc


class Model(nn.Module):
    def __init__(
        self,
        size_splits: Tuple[int, int, int],
        kv_output: bool = True,
        quant_mode: str = "static",
        layout: str = "BSND",
        cache_mode: str = "contiguous",
    ):
        super().__init__()
        self.size_splits = size_splits
        self.kv_output = kv_output
        self.quant_mode = quant_mode
        self.layout = layout
        self.cache_mode = cache_mode

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        scale_k: torch.Tensor,
        scale_v: torch.Tensor,
        offset_k: Optional[torch.Tensor],
        offset_v: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor],
        activation_scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        _q_buf: torch.Tensor,
        _k_buf: torch.Tensor,
        _v_buf: torch.Tensor,
    ) -> List[torch.Tensor]:
        del _q_buf, _k_buf, _v_buf
        ropeq, ropek, v1, kco, vco = _run_drqk(
            x,
            cos,
            sin,
            k_cache,
            v_cache,
            indices,
            scale_k,
            scale_v,
            self.size_splits,
            offset_k,
            offset_v,
            weight_scale,
            activation_scale,
            bias,
            self.cache_mode,
        )
        return [ropeq, ropek, v1, kco, vco]


class ModelNew(nn.Module):
    def __init__(
        self,
        size_splits: Tuple[int, int, int],
        kv_output: bool = True,
        quant_mode: str = "static",
        layout: str = "BSND",
        cache_mode: str = "contiguous",
    ):
        super().__init__()
        self.size_splits = size_splits
        self.kv_output = kv_output
        self.quant_mode = quant_mode
        self.layout = layout
        self.cache_mode = cache_mode

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        scale_k: torch.Tensor,
        scale_v: torch.Tensor,
        offset_k: Optional[torch.Tensor],
        offset_v: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor],
        activation_scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        q_buf: torch.Tensor,
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.dequant_rope_quant_kvcache(
            x,
            cos,
            sin,
            k_cache,
            v_cache,
            indices,
            scale_k,
            scale_v,
            offset_k,
            offset_v,
            weight_scale,
            activation_scale,
            bias,
            list(self.size_splits),
            self.quant_mode,
            self.layout,
            self.kv_output,
            self.cache_mode,
            q_buf,
            k_buf,
            v_buf,
        )
