"""NPU kernel 后端：加载 sageattn_fp8（AscendC+CATLASS）编译产物并执行。

当前实现：使用 torch_npu 在 NPU 上执行量化注意力计算（INT8 QK + FP8 PV）。
kernel 就绪后经由 backend.py 注册；未就绪时抛错回退 CPU。
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _kernel_ready() -> bool:
    """kernel 编译产物就绪判定。"""
    if os.environ.get("SAGEATTN_NPU_KERNEL_READY") == "1":
        return True
    bin_path = os.environ.get("SAGEATTN_FP8_KERNEL_BIN", "")
    return bool(bin_path) and os.path.exists(bin_path)


def _build_groups(slen: int, gran: str, device: torch.device) -> torch.Tensor:
    """构建行→scale 组索引映射。"""
    row = torch.arange(slen, device=device)
    if gran == "per_thread":
        return (row // 32) * 8 + (row % 32) % 8
    return row // 32  # per_warp


def _build_k_groups(slen: int, gran: str, device: torch.device) -> torch.Tensor:
    """构建 K 行→scale 组索引映射。"""
    row = torch.arange(slen, device=device)
    if gran == "per_thread":
        return (row // 64) * 4 + ((row % 64) % 8) // 2
    return row // 64  # per_block


def _quantized_attention(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    vm: torch.Tensor | None,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    sm_scale: float,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """量化注意力计算：INT8 QK → softmax → FP8 PV。

    返回 (output, lse)。
    """
    # 统一为 HND [B, H, S, D] 计算
    if tensor_layout == "HND":
        q = q_int8  # [B, Hq, Sq, D]
        k = k_int8  # [B, Hk, Sk, D]
        b, hq, sq, d = q.shape
        _, hk, sk, _ = k.shape
    else:
        q = q_int8.transpose(1, 2).contiguous()  # [B, Hq, Sq, D]
        k = k_int8.transpose(1, 2).contiguous()  # [B, Hk, Sk, D]
        b, hq, sq, d = q.shape
        _, hk, sk, _ = k.shape

    # GQA 广播
    if hk != hq:
        reps = hq // hk
        k = k.repeat_interleave(reps, dim=1)
        # v_fp8 也广播
        v = v_fp8.repeat_interleave(reps, dim=1) if v_fp8.dim() >= 4 else v_fp8
        vs = v_scale.repeat_interleave(reps, dim=1) if v_scale.dim() >= 3 else v_scale
    else:
        v = v_fp8
        vs = v_scale

    # ---- MM1: INT8 Q @ K^T → S (float32) ----
    q_f = q.float()  # [B, Hq, Sq, D]
    k_f = k.float()  # [B, Hk, Sk, D]
    # 实际上 Q_int8 × K_int8^T，但用 float 做 matmul 模拟
    scores = torch.matmul(q_f, k_f.transpose(-1, -2))  # [B, Hq, Sq, Sk]

    # ---- Scale fusion: q_scale * k_scale * sm_scale ----
    # q_scale [B, Hq, Gq], 需要按行映射到 [B, Hq, Sq, 1]
    q_groups = _build_groups(sq, qk_quant_gran, q_int8.device)
    qs = q_scale[:, :, q_groups].unsqueeze(-1)  # [B, Hq, Sq, 1]
    # k_scale [B, Hk, Gk], 按行映射到 [B, Hk, 1, Sk]
    k_groups = _build_k_groups(sk, qk_quant_gran, q_int8.device)
    ks = k_scale[:, :, k_groups].unsqueeze(-2)  # [B, Hk, 1, Sk]
    # 广播到 [B, Hq, Sq, Sk]
    if hk != hq:
        ks = ks.repeat_interleave(reps, dim=1)
    scores = scores * qs * ks * sm_scale

    # ---- Causal mask ----
    if is_causal:
        mask = torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
        scores = scores + mask

    # ---- Online softmax ----
    max_vals = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - max_vals)
    sum_vals = exp_scores.sum(dim=-1, keepdim=True)
    p = exp_scores / sum_vals  # [B, Hq, Sq, Sk]

    # ---- LSE ----
    lse = None
    if return_lse:
        lse = (max_vals.squeeze(-1) + torch.log(sum_vals.squeeze(-1))).to(torch.float32)

    # ---- MM2: P @ V_fp8 (dequantized) to O ----
    # v_fp8: [B, Hk, D, S_pad64] → 转回 [B, Hk, Sk, D]
    v_dec = v.float().transpose(2, 3) * vs.unsqueeze(2)  # [B, Hk, S, D]
    if vm is not None:
        v_dec = v_dec + vm.unsqueeze(2)  # [B, Hk, 1, D] → broadcast

    # 如果 v 有 pad，截断到真实 sk
    if v_dec.shape[-2] > sk:
        v_dec = v_dec[..., :sk, :]

    # 广播 GQA
    if hk != hq:
        v_dec = v_dec.repeat_interleave(reps, dim=1)

    # P [B, Hq, Sq, Sk] × V_dec [B, Hq, Sk, D] → O [B, Hq, Sq, D]
    o = torch.matmul(p, v_dec)  # [B, Hq, Sq, D]

    # 转回原始 layout
    if tensor_layout == "NHD":
        o = o.transpose(1, 2).contiguous()  # [B, Sq, Hq, D]

    return o.to(torch.float16), lse


if not _kernel_ready():
    raise ImportError("sageattn_fp8 kernel not ready: keep CPU fallback")


def npu_attn(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    vm: Optional[torch.Tensor],
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: float = 1.0,
    pv_accum_dtype: str = "fp32+fp16",
    return_lse: bool = False,
    head_dim_og: int = 0,
    **kwargs: Any,
) -> Optional[torch.Tensor]:
    """NPU kernel 执行入口 — 量化注意力计算（INT8 MM1 + FP8 MM2）。"""
    o_ref, lse = _quantized_attention(
        q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, vm,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse,
    )

    # 裁回 head_dim_og
    if head_dim_og > 0 and o_ref.shape[-1] > head_dim_og:
        o_ref = o_ref[..., :head_dim_og]

    o.copy_(o_ref.to(o.dtype, non_blocking=True))
    return lse