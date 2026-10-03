"""
Dispatcher and execution backend router for SageAttention2 on Ascend NPU.
Implements auto-routing to NPU FP8 PV path, pv_accum_dtype override via kwargs/env,
multi-backend auto-detection (C++ aclnn/op_plugin vs high-fidelity CPU simulation).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    F = MagicMock()

from .error_check import (
    compute_sm_scale,
    handle_smooth_v,
    pad_head_dim,
    validate_common_parameters,
    validate_tensors,
)
from .op_plugin import is_npu_available, is_npu_kernel_loaded, run_sageattn_npu


SAGEATTN_PV_ACCUM_DTYPE_DEFAULT = "fp32+fp16"

# 数值常量（python 侧唯一定义点；跨语言对应 C++ 侧唯一定义
# ../sageattn_fp8/ascendc/tiling.h，禁止在本语言内复制第二份数值）：
# - S_FP8_OFFSET = log2(448)：P 矩阵 e4m3 放大偏置（task/01_design 2.4）
# - S_FP8_SCALE_MAX / SCALE_MAX_FP16_ACCUM：V 量化 scale_max 两档（448.0 / 2.25）
# - INT8_QUANT_MAX / SCALE_EPSILON：INT8 对称量化上限与兜底微量
# - LOG2E：exp2 域换算系数。注意 1.44269504 为 golden 冻结截断精度，
#   勿"修正"为全精度 1.4426950408889634（位级对齐依赖）。
S_FP8_OFFSET = 8.807354922057604  # log2(448.0)
S_FP8_SCALE_MAX = 448.0
SCALE_MAX_FP16_ACCUM = 2.25
INT8_QUANT_MAX = 127.0
SCALE_EPSILON = 1e-7
LOG2E = 1.44269504

# Permutation pattern for FP8 V MMA layout (16-token chunks)
_V_FP8_PERM = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
_V_FP8_INV_PERM = [0] * 16
for _i, _p in enumerate(_V_FP8_PERM):
    _V_FP8_INV_PERM[_p] = _i


def _quant_qk_per_warp(
    q_hnd: torch.Tensor,
    k_centered_hnd: torch.Tensor,
    sq: int,
    sk: int,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """per_warp INT8 quantization for Q and K (RNE rounding + saturation).

    返回 (q_deq, k_deq, q_int8, q_scale, k_int8, k_scale)。中间态格式与 golden
    台账一致（q_int8/k_int8 裁回 sq/sk；q_scale [b,hq,blk*4]；k_scale [b,hkv,blk]），
    供单一基准整合（golden 生成器编排复用）。
    """
    b, hq = q_hnd.shape[0], q_hnd.shape[1]
    _, hkv = k_centered_hnd.shape[0], k_centered_hnd.shape[1]

    # Q per_warp: BLKQ=128, WARPQ=32 -> 4 scales per 128 tokens
    num_blkq = (sq + 127) // 128
    pad_sq = num_blkq * 128
    q_pad = F.pad(q_hnd, (0, 0, 0, pad_sq - sq)) if pad_sq > sq else q_hnd
    q_view = q_pad.view(b, hq, num_blkq, 4, 32, d).float()
    amax_q = q_view.abs().amax(dim=(4, 5)).clamp(min=SCALE_EPSILON)
    scale_q = amax_q / INT8_QUANT_MAX
    q_scale = scale_q.view(b, hq, num_blkq * 4)
    q_scaled = q_view / scale_q.unsqueeze(4).unsqueeze(5)
    q_quant = torch.round(q_scaled).clamp(-128, 127).to(torch.int8)
    q_int8 = q_quant.view(b, hq, pad_sq, d)[:, :, :sq, :]
    q_deq = (q_quant.float() * scale_q.unsqueeze(4).unsqueeze(5)).view(b, hq, pad_sq, d)[:, :, :sq, :]

    # K per_block: BLKK=64 -> 1 scale per 64 tokens
    num_blkk = (sk + 63) // 64
    pad_sk = num_blkk * 64
    k_pad = F.pad(k_centered_hnd, (0, 0, 0, pad_sk - sk)) if pad_sk > sk else k_centered_hnd
    k_view = k_pad.view(b, hkv, num_blkk, 64, d).float()
    amax_k = k_view.abs().amax(dim=(3, 4)).clamp(min=SCALE_EPSILON)
    scale_k = amax_k / INT8_QUANT_MAX
    k_scale = scale_k.view(b, hkv, num_blkk)
    k_scaled = k_view / scale_k.unsqueeze(3).unsqueeze(4)
    k_quant = torch.round(k_scaled).clamp(-128, 127).to(torch.int8)
    k_int8 = k_quant.view(b, hkv, pad_sk, d)[:, :, :sk, :]
    k_deq = (k_quant.float() * scale_k.unsqueeze(3).unsqueeze(4)).view(b, hkv, pad_sk, d)[:, :, :sk, :]

    return q_deq, k_deq, q_int8, q_scale, k_int8, k_scale


def _quant_qk_per_thread(
    q_hnd: torch.Tensor,
    k_centered_hnd: torch.Tensor,
    sq: int,
    sk: int,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """per_thread INT8 quantization for Q and K (half-away-from-zero rounding + clamp[-127, 127]).

    返回 (q_deq, k_deq, q_int8, q_scale, k_int8, k_scale)。中间态格式与 golden
    台账一致（q_int8/k_int8 裁回 sq/sk；q_scale [b,hq,blk*32]；k_scale [b,hkv,blk*4]）。
    """
    b, hq = q_hnd.shape[0], q_hnd.shape[1]
    _, hkv = k_centered_hnd.shape[0], k_centered_hnd.shape[1]

    # Q per_thread: 128 tokens -> 4 subblocks of 32 -> 8 groups of 4 tokens (stride 8)
    num_blkq = (sq + 127) // 128
    pad_sq = num_blkq * 128
    q_pad = F.pad(q_hnd, (0, 0, 0, pad_sq - sq)) if pad_sq > sq else q_hnd
    q_view = q_pad.view(b, hq, num_blkq, 4, 4, 8, d).float()
    amax_q = q_view.abs().amax(dim=(4, 6))
    scale_q = amax_q / INT8_QUANT_MAX + SCALE_EPSILON
    scale_exp_q = scale_q.unsqueeze(4).unsqueeze(6)
    q_scaled = q_view / scale_exp_q
    q_quant = torch.clamp(torch.floor(torch.abs(q_scaled) + 0.5) * torch.sign(q_scaled), -127, 127).to(torch.int8)
    q_int8 = q_quant.view(b, hq, pad_sq, d)[:, :, :sq, :]
    q_deq = (q_quant.float() * scale_exp_q).view(b, hq, pad_sq, d)[:, :, :sq, :]
    q_scale = scale_q.view(b, hq, num_blkq * 32)

    # K per_thread: 64 tokens -> 4 groups of 8 pairs (adjacent columns)
    num_blkk = (sk + 63) // 64
    pad_sk = num_blkk * 64
    k_pad = F.pad(k_centered_hnd, (0, 0, 0, pad_sk - sk)) if pad_sk > sk else k_centered_hnd
    k_view = k_pad.view(b, hkv, num_blkk, 8, 4, 2, d).float()
    amax_k = k_view.abs().amax(dim=(3, 5, 6))
    scale_k = amax_k / INT8_QUANT_MAX + SCALE_EPSILON
    scale_exp_k = scale_k.unsqueeze(3).unsqueeze(5).unsqueeze(6)
    k_scaled = k_view / scale_exp_k
    k_quant = torch.clamp(torch.floor(torch.abs(k_scaled) + 0.5) * torch.sign(k_scaled), -127, 127).to(torch.int8)
    k_int8 = k_quant.view(b, hkv, pad_sk, d)[:, :, :sk, :]
    k_deq = (k_quant.float() * scale_exp_k).view(b, hkv, pad_sk, d)[:, :, :sk, :]
    k_scale = scale_k.view(b, hkv, num_blkk * 4)

    return q_deq, k_deq, q_int8, q_scale, k_int8, k_scale


def _quant_v_fp8(
    v_hnd: torch.Tensor,
    sk: int,
    d: int,
    pv_accum_dtype: str,
    smooth_v: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Quantize V to FP8 e4m3fn with sequence padding and permuted layout.

    返回 (v_deq_hnd, v_fp8, v_scale, vm, v_norm_clamped)。中间态格式与 golden 台账
    一致（v_fp8 [b,hkv,d,pad_sk_v] permuted、v_scale [b,hkv,d]、vm 仅 smooth_v 路径
    非 None），供单一基准整合（golden 生成器编排复用）。
    """
    b, hkv = v_hnd.shape[0], v_hnd.shape[1]
    scale_max = SCALE_MAX_FP16_ACCUM if pv_accum_dtype == "fp32+fp16" else S_FP8_SCALE_MAX

    pad_sk_v = (sk + 63) // 64 * 64
    v_t = v_hnd.transpose(2, 3)  # (b, hkv, d, sk)
    v_pad = F.pad(v_t, (0, pad_sk_v - sk)) if pad_sk_v > sk else v_t

    num_16 = pad_sk_v // 16
    v_16 = v_pad.view(b, hkv, d, num_16, 16)
    v_perm = v_16[:, :, :, :, _V_FP8_PERM].contiguous().view(b, hkv, d, pad_sk_v)

    fp8_padded_tokens = (sk + 15) // 16 * 16
    v_sub = v_perm[:, :, :, :fp8_padded_tokens].float()

    has_fp8 = hasattr(torch, "float8_e4m3fn")

    if smooth_v:
        vm = v_sub.sum(dim=-1) / fp8_padded_tokens
        amax_v = torch.max((v_sub.amax(dim=-1) - vm).abs(), (v_sub.amin(dim=-1) - vm).abs())
        v_scale = amax_v / scale_max
        v_norm = (v_perm.float() - vm.unsqueeze(-1)) / v_scale.unsqueeze(-1)
        v_fp8 = v_norm.clamp(-S_FP8_SCALE_MAX, S_FP8_SCALE_MAX).to(torch.float8_e4m3fn) if has_fp8 else v_norm.clamp(-S_FP8_SCALE_MAX, S_FP8_SCALE_MAX)
        v_fp8_unperm = v_fp8.view(b, hkv, d, num_16, 16)[:, :, :, :, _V_FP8_INV_PERM].contiguous().view(b, hkv, d, pad_sk_v)[:, :, :, :sk]
        v_deq_hnd = (v_fp8_unperm.float() * v_scale.unsqueeze(-1) + vm.unsqueeze(-1)).transpose(2, 3)
    else:
        vm = None
        amax_v = torch.max(v_sub.amax(dim=-1).abs(), v_sub.amin(dim=-1).abs())
        v_scale = amax_v / scale_max
        v_norm = v_perm.float() / v_scale.unsqueeze(-1)
        v_fp8 = v_norm.clamp(-S_FP8_SCALE_MAX, S_FP8_SCALE_MAX).to(torch.float8_e4m3fn) if has_fp8 else v_norm.clamp(-S_FP8_SCALE_MAX, S_FP8_SCALE_MAX)
        v_fp8_unperm = v_fp8.view(b, hkv, d, num_16, 16)[:, :, :, :, _V_FP8_INV_PERM].contiguous().view(b, hkv, d, pad_sk_v)[:, :, :, :sk]
        v_deq_hnd = (v_fp8_unperm.float() * v_scale.unsqueeze(-1)).transpose(2, 3)

    return v_deq_hnd, v_fp8, v_scale, vm, v_norm


def run_sageattn_cpu_reference(
    q_pad: torch.Tensor,
    k_pad: torch.Tensor,
    v_pad: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    sm_scale: float,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
    return_intermediates: bool = False,
):
    """
    High-fidelity CPU reference simulation for SageAttention2 on Ascend NPU.
    单一基准（Group J）：本函数是算子侧唯一 Python 数值参考；golden 生成器
    （golden/generate_golden.py）经 return_intermediates=True 编排复用，两份
    量化/仿真实现已删除。中间态格式 = golden 台账格式（HND 域：q_int8 裁回
    sq、q_scale 组序、v_fp8 permuted [b,hkv,d,pad_sk_v]、v_scale [b,hkv,d]）。
    """
    dtype = q_pad.dtype
    sm_scale_log2e = sm_scale * LOG2E

    # Convert layout to HND for standardized tile processing
    if tensor_layout == "NHD":
        q_hnd = q_pad.transpose(1, 2).contiguous()
        k_hnd = k_pad.transpose(1, 2).contiguous()
        v_hnd = v_pad.transpose(1, 2).contiguous()
    else:
        q_hnd = q_pad
        k_hnd = k_pad
        v_hnd = v_pad

    b, hq, sq, d = q_hnd.shape
    _, hkv, sk, _ = k_hnd.shape

    # 1. smooth_k and lse_correction
    if smooth_k:
        km = k_hnd.mean(dim=2, keepdim=True)
        q_per_kv = hq // hkv
        km_broadcast = km.repeat_interleave(q_per_kv, dim=1) if q_per_kv > 1 else km
        lse_correction = torch.matmul(q_hnd.float(), km_broadcast.float().transpose(2, 3)).squeeze(-1).float()
        k_centered_hnd = k_hnd - km
    else:
        k_centered_hnd = k_hnd
        lse_correction = torch.zeros((b, hq, sq), dtype=torch.float32, device=q_hnd.device)

    # 2. Quantize Q & K（返回中间态：q_int8/q_scale/k_int8/k_scale，golden 台账格式）
    if qk_quant_gran == "per_thread":
        q_deq_hnd, k_deq_hnd, q_int8, q_scale, k_int8, k_scale = _quant_qk_per_thread(q_hnd, k_centered_hnd, sq, sk, d)
    else:
        q_deq_hnd, k_deq_hnd, q_int8, q_scale, k_int8, k_scale = _quant_qk_per_warp(q_hnd, k_centered_hnd, sq, sk, d)

    # 3. Quantize V to FP8（返回中间态：v_fp8 permuted / v_scale / vm）
    v_deq_hnd, v_fp8, v_scale, vm, _ = _quant_v_fp8(v_hnd, sk, d, pv_accum_dtype, smooth_v)

    # 4. GQA broadcasting
    if hq != hkv:
        rep = hq // hkv
        k_deq_hnd = k_deq_hnd.repeat_interleave(rep, dim=1)
        v_deq_hnd = v_deq_hnd.repeat_interleave(rep, dim=1)

    # 5. Online Softmax and Two-level Accumulation Simulation
    # S_FP8_OFFSET 放大语义（task/01_design 2.4，对齐 GPU attn_utils.cuh 的
    # m' = m - log2(448) 冻结轨迹）：P 在转 e4m3 前放大 448 倍填满 e4m3 动态范围，
    # 分母在放大域累加，输出归一化时分子分母相消，LSE 减去 log2(448) 偏置。
    cta_k = 64
    k_step = 16
    num_cta_k = (sk + cta_k - 1) // cta_k
    ro = torch.zeros((b, hq, sq, d), dtype=torch.float32, device=q_hnd.device)
    m = torch.full((b, hq, sq, 1), -5000000.0, dtype=torch.float32, device=q_hnd.device)
    d_denom = torch.ones((b, hq, sq, 1), dtype=torch.float32, device=q_hnd.device)

    has_fp8 = hasattr(torch, "float8_e4m3fn")

    for k_blk in range(num_cta_k):
        k_start = k_blk * cta_k
        k_end = min(sk, (k_blk + 1) * cta_k)
        k_chunk = k_deq_hnd[:, :, k_start:k_end, :]
        v_chunk = v_deq_hnd[:, :, k_start:k_end, :]

        # Scores in exp2 domain
        s_chunk = torch.matmul(q_deq_hnd, k_chunk.transpose(-2, -1)) * sm_scale_log2e

        if is_causal:
            q_indices = torch.arange(sq, device=s_chunk.device).view(1, 1, sq, 1)
            k_indices = torch.arange(k_start, k_end, device=s_chunk.device).view(1, 1, 1, k_end - k_start)
            causal_mask = q_indices < k_indices
            s_chunk = s_chunk.masked_fill(causal_mask, -5000000.0)

        m_local = s_chunk.max(dim=-1, keepdim=True)[0]
        m_new = torch.maximum(m, m_local)
        o_scale = torch.pow(2.0, m - m_new)
        d_denom = d_denom * o_scale
        ro = ro * o_scale

        p_exp2 = torch.pow(2.0, s_chunk - m_new)
        if is_causal:
            p_exp2 = p_exp2.masked_fill(causal_mask, 0.0)

        # Quantize P to e4m3 with S_FP8_OFFSET (= log2(448)) scaling:
        # P 先放大 448 倍再 clamp/转 e4m3，分母在放大域累加
        p_scaled = p_exp2 * S_FP8_SCALE_MAX
        p_f8 = p_scaled.clamp(-S_FP8_SCALE_MAX, S_FP8_SCALE_MAX).to(torch.float8_e4m3fn).float() if has_fp8 else p_scaled.clamp(-S_FP8_SCALE_MAX, S_FP8_SCALE_MAX)
        d_denom = d_denom + p_scaled.sum(dim=-1, keepdim=True)

        # Two-level accumulation（冻结语义轨迹：fp32+fp16 档为 "FP16 主累加器 +
        # 每 k_step 归约回 FP32"——每 k_step=16 个 KV token 的 PV 在 FP16 域
        # 累加，经一次 FP16 舍入后归约进 FP32 主累加器）
        if pv_accum_dtype == "fp32+fp16":
            pv_chunk = torch.zeros(
                (b, hq, sq, d), dtype=torch.float32, device=q_hnd.device
            )
            for ks_start in range(0, k_end - k_start, k_step):
                ks_end = min(ks_start + k_step, k_end - k_start)
                p_sub = p_f8[:, :, :, ks_start:ks_end].contiguous().half()
                v_sub = v_chunk[:, :, ks_start:ks_end, :].contiguous().half()
                pv_sub = torch.matmul(p_sub, v_sub)
                pv_chunk = pv_chunk + pv_sub.float()
        else:
            pv_chunk = torch.matmul(p_f8, v_chunk)

        ro = ro + pv_chunk
        m = m_new

    # Output normalization（448 放大在分子分母间相消）
    out_hnd = ro / d_denom

    # LSE calculation（分母处于 448 放大域，需减去 S_FP8_OFFSET = log2(448) 偏置）
    if return_lse:
        lse_exp2 = torch.log2(d_denom.squeeze(-1).clamp(min=1e-12)) + m.squeeze(-1) - S_FP8_OFFSET
        lse_out = lse_exp2 / LOG2E + (lse_correction * sm_scale if smooth_k else 0.0)
    else:
        lse_out = None

    # Cast back to input dtype
    out_hnd = out_hnd.to(dtype)

    # Restore output layout
    if tensor_layout == "NHD":
        out = out_hnd.transpose(1, 2).contiguous()
    else:
        out = out_hnd

    if return_intermediates:
        intermediates = {
            "q_int8": q_int8,        # [b, hq, sq, d] HND，裁回 sq
            "q_scale": q_scale,      # [b, hq, blk*(32|4)]
            "k_int8": k_int8,        # [b, hkv, sk, d] HND，裁回 sk
            "k_scale": k_scale,      # [b, hkv, blk*(4|1)]
            "v_fp8": v_fp8,          # [b, hkv, d, pad_sk_v] 16-token permuted
            "v_scale": v_scale,      # [b, hkv, d]
            "vm": vm,                # 仅 smooth_v 路径非 None
        }
        return out, lse_out, intermediates
    return out, lse_out


def _run_cpu_reference_with_stub_warning(
    warn_stub: bool,
    npu_in_use: bool,
    **ref_kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the CPU reference; warn loudly when it is serving as an NPU stub under explicit opt-in."""
    if warn_stub or npu_in_use:
        warnings.warn(
            "SageAttention2 is running the high-fidelity CPU reference stub "
            "(SAGEATTN_ALLOW_CPU_STUB opt-in); results are NOT valid NPU precision/performance evidence.",
            UserWarning,
            stacklevel=3,
        )
    return run_sageattn_cpu_reference(**ref_kwargs)


def dispatch_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    Dispatch sageattn to sageattn_qk_int8_pv_fp8_asc on Ascend NPU.
    Reads pv_accum_dtype from kwargs or SAGEATTN_PV_ACCUM_DTYPE environment variable.
    """
    pv_accum_dtype = kwargs.pop(
        "pv_accum_dtype",
        os.environ.get("SAGEATTN_PV_ACCUM_DTYPE", SAGEATTN_PV_ACCUM_DTYPE_DEFAULT),
    )
    qk_quant_gran = kwargs.pop("qk_quant_gran", "per_thread")
    smooth_k = kwargs.pop("smooth_k", True)
    smooth_v = kwargs.pop("smooth_v", False)

    return dispatch_sageattn_qk_int8_pv_fp8_asc(
        q=q,
        k=k,
        v=v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        **kwargs,
    )


def dispatch_sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Dispatch sageattn_qk_int8_pv_fp8_asc across NPU C++ / aclnn and CPU reference backends.
    """
    validate_common_parameters(
        tensor_layout=tensor_layout,
        qk_quant_gran=qk_quant_gran,
        pv_accum_dtype=pv_accum_dtype,
        kwargs=kwargs,
    )

    smooth_v = handle_smooth_v(pv_accum_dtype, smooth_v)

    validate_tensors(
        q=q,
        k=k,
        v=v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
    )

    q_pad, k_pad, v_pad, head_dim_og = pad_head_dim(q, k, v)
    sm_scale_val = compute_sm_scale(head_dim_og, sm_scale)

    allow_cpu_stub = os.environ.get("SAGEATTN_ALLOW_CPU_STUB", "0").lower() in ("1", "true", "yes")
    is_cpu_device = hasattr(v, "device") and v.device.type == "cpu"
    is_meta_device = hasattr(v, "device") and v.device.type == "meta"
    is_fake_tensor = type(v).__name__ == "FakeTensor" or hasattr(v, "fake_mode")
    npu_avail = is_npu_available() and is_npu_kernel_loaded()

    if is_meta_device or is_fake_tensor:
        out_pad, lse = torch.ops.sageattention2.forward(
            q_pad,
            k_pad,
            v_pad,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale_val,
            smooth_k=smooth_k,
            return_lse=return_lse,
        )
        out = out_pad[..., :head_dim_og]
        if return_lse:
            return out, lse
        return out

    # 禁止静默 CPU 回退：NPU 在位但 kernel 未加载时，必须显式 SAGEATTN_ALLOW_CPU_STUB
    # opt-in（附带警告）才允许走 CPU 参考；纯 CPU 张量（本地单测）不受此门控限制。
    # meta/fake 张量已在上方分支提前返回，此处无需重复判断。
    npu_present = is_npu_available()
    if not is_cpu_device and npu_present and not npu_avail and not allow_cpu_stub:
        raise RuntimeError(
            "SageAttention2 NPU kernel is not loaded while an Ascend NPU is present; "
            "build the custom operator (bash build_opp.sh) or set SAGEATTN_ALLOW_CPU_STUB=1 "
            "to explicitly opt in to the high-fidelity CPU reference stub."
        )

    if npu_avail and not is_cpu_device and not allow_cpu_stub:
        # 该分支内 is_cpu_device/allow_cpu_stub 恒为 False 且 npu_present 恒为 True，
        # CPU 参考回退在此不可达（显式回退仅走下方 else 分支并伴随 stub 告警），
        # kernel 缺位时按门控语义直接上抛 NotImplementedError。
        out_pad, lse = run_sageattn_npu(
            q=q_pad,
            k=k_pad,
            v=v_pad,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            qk_quant_gran=qk_quant_gran,
            sm_scale=sm_scale_val,
            pv_accum_dtype=pv_accum_dtype,
            smooth_k=smooth_k,
            smooth_v=smooth_v,
            return_lse=return_lse,
            **kwargs,
        )
    else:
        out_pad, lse = _run_cpu_reference_with_stub_warning(
            allow_cpu_stub, npu_present and not is_cpu_device,
            q_pad=q_pad, k_pad=k_pad, v_pad=v_pad,
            tensor_layout=tensor_layout, is_causal=is_causal,
            qk_quant_gran=qk_quant_gran, sm_scale=sm_scale_val,
            pv_accum_dtype=pv_accum_dtype, smooth_k=smooth_k,
            smooth_v=smooth_v, return_lse=return_lse,
        )

    out = out_pad[..., :head_dim_og]

    if return_lse:
        return out, lse
    return out
