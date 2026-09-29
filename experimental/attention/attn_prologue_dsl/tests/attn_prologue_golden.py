# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 CPU 参考实现（golden）—— `attn_prologue`。

设计依据：`attn_prologue_plan/03_golden设计.md`。

五条原则：
  P1 独立      —— 不 import cannbotdsl，不复用 kernel 侧任何布局变换/常量
  P2 确定性    —— 只用显式 Generator，不碰全局 RNG
  P3 分段可观测 —— 返回 10 个中间量而不只是 3 个公开输出
  P4 自证      —— 同一份实现可跑 fp64 影子（compute_dtype=float64）
  P5 不物化    —— 按行分块；trace 可关

融合算子的三个"必须建模对"的顺序（03 文档 §2）：
  1. QB 的 A 操作数是**量化后**的 qr，不是 qr_fp32
  2. Q 支路 RoPE 在量化之后；KV 支路 RoPE 在量化之前
  3. kv_cache 是 bitcast(uint8)，不是数值 cast
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Iterable, Iterator

import torch

MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
E8M0_BIAS = 127
E4M3FN_MAX = 448.0
# E4M3 量化的共享指数偏置：给 3 位尾数留动态范围（01 文档 §3）
QUANT_EXP_HEADROOM = 8
TYPICAL_BLOCK_SIZE = 128

SMALL_PROFILE = (64, 64, 2, 64, 16)
PRODUCTION_COMPAT_PROFILE = (7168, 1536, 32, 512, 64)
TARGET_PROFILE = (5120, 1280, 64, 512, 64)


def ceil_div(value: int, divisor: int) -> int:
    """正整数向上整除。"""
    if value < 0 or divisor <= 0:
        raise ValueError("value must be non-negative and divisor positive")
    return (value + divisor - 1) // divisor


def paired_scale_shape(outer_size: int, reduction_size: int) -> tuple[int, int, int]:
    """公开接口的 paired-E8M0 scale shape：(outer, ceil(K/64), 2)。"""
    return outer_size, ceil_div(reduction_size, 64), MX_SCALE_PAIR


# --------------------------------------------------------------------------
# MXFP8 编解码
# --------------------------------------------------------------------------

def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """把 E8M0 / int8 / uint8 载体统一看成无符号字节。"""
    if scale.device.type != "cpu":
        scale = scale.cpu()
    if scale.dtype in (torch.uint8, torch.int8, torch.float8_e8m0fnu):
        return scale.contiguous().view(torch.uint8)
    raise TypeError("paired scales must use uint8 / int8 / float8_e8m0fnu storage")


def _pow2(exponents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """精确构造 2**exponents。

    用 ``ldexp`` 而不是 ``pow(2.0, e)``：前者是位操作，对 e=-127 这种
    落进 fp32 次正规区的指数也精确（E8M0 字节 0 就是 2^-127）。
    """
    ones = torch.ones(exponents.shape, dtype=dtype)
    return torch.ldexp(ones, exponents.to(torch.int32))


def decode_paired_e8m0(
    scale: torch.Tensor,
    outer_size: int,
    reduction_size: int,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """把 (outer, ceil(K/64), 2) 解码成"每 32 个元素一个"的 FP 倍数。"""
    expected = paired_scale_shape(outer_size, reduction_size)
    if tuple(scale.shape) != expected:
        raise ValueError(f"expected paired scale shape {expected}, got {tuple(scale.shape)}")
    # MX 是每 32 值一组；K 轴共 ceil(K/32) 个真实组，pair 末尾可能是 padding。
    valid_groups = ceil_div(reduction_size, MX_GROUP_SIZE)
    # pair 内存连续 ⇒ 展平后就是按组号升序的 E8M0 字节。
    exponent_bytes = _scale_bytes(scale).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid_groups].to(torch.int16) - E8M0_BIAS
    return _pow2(exponents, compute_dtype)


def expand_paired_e8m0(
    scale: torch.Tensor,
    outer_size: int,
    reduction_size: int,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """把每组一个的 scale 广播成每个元素一个。"""
    decoded = decode_paired_e8m0(scale, outer_size, reduction_size, compute_dtype)
    return decoded.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :reduction_size]


def dequantize_mxfp8(
    payload: torch.Tensor,
    scale: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """用公开 paired E8M0 元数据反量化 rank-2 E4M3FN payload。"""
    if payload.ndim != 2 or payload.dtype != torch.float8_e4m3fn:
        raise TypeError("payload must be a rank-2 torch.float8_e4m3fn tensor")
    outer_size, reduction_size = payload.shape
    expanded = expand_paired_e8m0(scale, outer_size, reduction_size, compute_dtype)
    return payload.to(compute_dtype) * expanded


def _encode_e8m0_exponents(exponents: torch.Tensor) -> torch.Tensor:
    """无偏指数 → E8M0 字节，以 int8 载体存储（与公开 descale dtype 一致）。"""
    return (exponents.to(torch.int16) + E8M0_BIAS).to(torch.uint8).view(torch.int8)


def _floor_log2(values: torch.Tensor) -> torch.Tensor:
    """精确的 floor(log2(x))，x > 0。

    **不用** ``floor(log2(x))``：对略小于 2 的幂的输入，``log2`` 可能向上
    round 到整数，``floor`` 就会多给 1。``frexp`` 返回 x = m·2^e 且
    m ∈ [0.5, 1)，所以 floor(log2(x)) == e - 1，是位精确的。
    这也和 kernel 侧"直接取 FP32 指数字段"的做法等价。
    """
    return torch.frexp(values).exponent.to(torch.int16) - 1


def quantize_mxfp8(
    values: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """沿最后一维做动态 MXFP8 量化，每 32 值共享一个 E8M0 指数。

    共享指数 ``e = clamp(floor(log2(amax)) - 8, -127, 127)``。
    零块输出 payload 全 0、scale 字节 0。

    返回 ``(payload, paired_scale, normalized)``。``normalized`` 是**实际
    被 cast 的值**（已除以组 scale、已饱和），只服务于 03 文档 §4.3 的
    E4M3 边界归因，不是公开输出。
    """
    if values.ndim != 2 or not values.is_floating_point():
        raise TypeError("values must be a rank-2 floating-point tensor")
    if values.device.type != "cpu":
        raise ValueError("the independent golden only accepts CPU tensors")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("MXFP8 quantization requires finite values")

    rows, width = values.shape
    groups = ceil_div(width, MX_GROUP_SIZE)
    pair_groups = ceil_div(groups, MX_SCALE_PAIR) * MX_SCALE_PAIR
    payload = torch.empty((rows, width), dtype=torch.float8_e4m3fn)
    normalized_all = torch.empty((rows, width), dtype=compute_dtype)
    scale_flat = torch.zeros((rows, pair_groups), dtype=torch.int8)
    work = values.to(compute_dtype)

    for group_index in range(groups):
        start = group_index * MX_GROUP_SIZE
        end = min(start + MX_GROUP_SIZE, width)
        block = work[:, start:end]
        amax = block.abs().amax(dim=1)
        zero = amax == 0
        # log2(0) 无定义：零块借 1.0 参与运算，稍后整组清零。
        safe_amax = torch.where(zero, torch.ones_like(amax), amax)
        exponents = (_floor_log2(safe_amax) - QUANT_EXP_HEADROOM).clamp(-127, 127)
        scales = _pow2(exponents, compute_dtype)
        normalized = (block / scales[:, None]).clamp(-E4M3FN_MAX, E4M3FN_MAX)
        normalized = torch.where(zero[:, None], torch.zeros_like(normalized), normalized)
        normalized_all[:, start:end] = normalized
        payload[:, start:end] = normalized.to(torch.float8_e4m3fn)
        encoded = _encode_e8m0_exponents(exponents)
        # 零块按公开约定输出 scale 字节 0，不保留借用值。
        scale_flat[:, group_index] = torch.where(zero, torch.zeros_like(encoded), encoded)

    paired = scale_flat.reshape(rows, pair_groups // MX_SCALE_PAIR, MX_SCALE_PAIR)
    return payload, paired.contiguous(), normalized_all


# --------------------------------------------------------------------------
# RMSNorm / RoPE
# --------------------------------------------------------------------------

def rms_norm(
    values: torch.Tensor,
    weight: torch.Tensor,
    norm_eps: float,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """整行 RMSNorm。QA 沿 R 归约，KV 沿 D 归约。"""
    if values.ndim != 2 or weight.ndim != 1:
        raise ValueError("RMSNorm expects rank-2 values and rank-1 weight")
    if values.shape[1] != weight.numel():
        raise ValueError("RMSNorm weight length must equal the row width")
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    work = values.to(compute_dtype)
    variance = work.square().mean(dim=-1, keepdim=True)
    return work * torch.rsqrt(variance + norm_eps) * weight.to(compute_dtype)


class RopeConvention(str, enum.Enum):
    """`rotary` 的配对/落位约定。

    流程图与 issue #7 **都没有定义**这件事（01 文档 §5，决策 D2）。
    默认值是前一版基线冻结的 INTERLEAVE_HALF，已由用户确认沿用。

    ⚠️ **INTERLEAVE_HALF 在标准 RoPE 表下不是旋转。** 它的逐对 2×2 矩阵是
    ``[[cos[i], sin[i]], [sin[half+i], cos[half+i]]]``——注意**没有负号**，
    正交条件化简为 ``sin(θ_p + θ_q) = 0``（``p = i//2``，``q = (half+i)//2``），
    即角度表的后半必须是前半的相反数。用 ``make_rope_tables`` 那种单调频率表
    时范数不守恒（实测 0.979 / 1.106）。GPTJ_INTERLEAVED 与 NEOX_HALF_SPLIT
    自带负号，在任何表下都是旋转。
    见 ``test_interleave_half_is_orthogonal_only_with_antisymmetric_tables``。
    """

    INTERLEAVE_HALF = "interleave_half"
    GPTJ_INTERLEAVED = "gptj_interleaved"
    NEOX_HALF_SPLIT = "neox_half_split"


def apply_tail_rope(
    values: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    convention: RopeConvention = RopeConvention.INTERLEAVE_HALF,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """只改写末尾 Dr 维；前 D-Dr 维**逐位原样透传**。

    ``values`` 可以是 [T,D] 或 [T,N,D]；后者在 N 上广播 sin/cos
    （流程图 Q 支路的 RoPE 在 reshape 之后，作用于 (b,s,n,rd)）。
    """
    if values.ndim not in (2, 3):
        raise ValueError("RoPE values must have shape [T,D] or [T,N,D]")
    if rope_sin.ndim != 2 or tuple(rope_sin.shape) != tuple(rope_cos.shape):
        raise ValueError("rope_sin and rope_cos must have the same rank-2 shape")
    token_count, rope_width = rope_sin.shape
    if values.shape[0] != token_count or rope_width > values.shape[-1]:
        raise ValueError("RoPE shape is incompatible with values")
    if rope_width % 2:
        raise ValueError("RoPE width must be even")

    result = values.to(compute_dtype).clone()
    tail = result[..., -rope_width:]
    # [T,Dr] → [T,1,...,Dr]，兼容 [T,D] 与 [T,N,D]
    table_shape = (token_count,) + (1,) * (values.ndim - 2) + (rope_width,)
    sin = rope_sin.to(compute_dtype).reshape(table_shape)
    cos = rope_cos.to(compute_dtype).reshape(table_shape)
    half = rope_width // 2

    if convention is RopeConvention.INTERLEAVE_HALF:
        # 相邻输入对 (2i, 2i+1) → 两个连续输出半区。
        # 没有负号：符号约定由调用方提供的 rope_sin 表承载。
        even = tail[..., 0::2].clone()
        odd = tail[..., 1::2].clone()
        tail[..., :half] = even * cos[..., :half] + odd * sin[..., :half]
        tail[..., half:] = even * sin[..., half:] + odd * cos[..., half:]
    elif convention is RopeConvention.GPTJ_INTERLEAVED:
        even = tail[..., 0::2].clone()
        odd = tail[..., 1::2].clone()
        se, ce = sin[..., 0::2], cos[..., 0::2]
        tail[..., 0::2] = even * ce - odd * se
        tail[..., 1::2] = even * se + odd * ce
    elif convention is RopeConvention.NEOX_HALF_SPLIT:
        lo = tail[..., :half].clone()
        hi = tail[..., half:].clone()
        sl, cl = sin[..., :half], cos[..., :half]
        tail[..., :half] = lo * cl - hi * sl
        tail[..., half:] = hi * cl + lo * sl
    else:  # pragma: no cover - enum 已穷举
        raise ValueError(f"unknown RoPE convention: {convention!r}")
    return result


# --------------------------------------------------------------------------
# 输入 / trace 容器
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AttnPrologueInputs:
    """公开输入（issue #7 的 14 项 INPUT）。"""

    x: torch.Tensor
    wqa: torch.Tensor
    wqb: torch.Tensor
    wkv: torch.Tensor
    descale_x: torch.Tensor
    descale_wqa: torch.Tensor
    descale_wqb: torch.Tensor
    descale_wkv: torch.Tensor
    norm_weight_qr: torch.Tensor
    norm_weight_kv: torch.Tensor
    rope_sin: torch.Tensor
    rope_cos: torch.Tensor
    cache_index: torch.Tensor
    kv_cache: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AttnPrologueTrace:
    """全部阶段产物；★ 标记的是公开输出。

    每个字段对应 kernel 侧一个可 dump 的落点，见 03 文档 §3 的对照表。
    ``trace=False`` 时非 ★ 字段为 None。
    """

    # 公开输出
    q: torch.Tensor                     # ★ [T,N,D] bf16
    qr: torch.Tensor                    # ★ [T,R]   e4m3
    descale_qr: torch.Tensor            # ★ [T,G(R),2] e8m0(int8)
    kv_cache: torch.Tensor              # ★ [P,BS,1,D] uint8（就地更新结果）
    # 中间量
    x_deq: torch.Tensor | None = None           # [T,H]
    qa_fp32: torch.Tensor | None = None         # [T,R]
    qr_fp32: torch.Tensor | None = None         # [T,R]  RMSNorm 后、量化前
    qr_normalized: torch.Tensor | None = None   # [T,R]  实际被 cast 的值
    q_proj_fp32: torch.Tensor | None = None     # [T,N*D]
    q_roped_fp32: torch.Tensor | None = None    # [T,N,D] RoPE 后、BF16 前
    kv_fp32: torch.Tensor | None = None         # [T,D]
    kv_normed: torch.Tensor | None = None       # [T,D]
    kv_roped: torch.Tensor | None = None        # [T,D]  RoPE 后、量化前
    kv_bytes: torch.Tensor | None = None        # [T,D]  bitcast 后、scatter 前


def _validate_inputs(inputs: AttnPrologueInputs) -> tuple[int, int, int, int, int]:
    """校验冻结契约，返回 (T, H, R, N, D)。"""
    if inputs.x.ndim != 2:
        raise ValueError("x must have shape [T,H]")
    token_count, hidden_size = inputs.x.shape
    if inputs.wqa.ndim != 2 or inputs.wqa.shape[1] != hidden_size:
        raise ValueError("wqa must have shape [R,H]")
    rank_size = inputs.wqa.shape[0]
    if inputs.wkv.ndim != 2 or inputs.wkv.shape[1] != hidden_size:
        raise ValueError("wkv must have shape [D,H]")
    head_size = inputs.wkv.shape[0]
    if inputs.wqb.ndim != 2 or inputs.wqb.shape[1] != rank_size:
        raise ValueError("wqb must have shape [N*D,R]")
    if inputs.wqb.shape[0] % head_size:
        raise ValueError("wqb output width must be divisible by D")
    head_count = inputs.wqb.shape[0] // head_size

    for name, tensor in (
        ("x", inputs.x), ("wqa", inputs.wqa),
        ("wqb", inputs.wqb), ("wkv", inputs.wkv),
    ):
        if tensor.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name} must use torch.float8_e4m3fn")

    for name, tensor, shape in (
        ("descale_x", inputs.descale_x, paired_scale_shape(token_count, hidden_size)),
        ("descale_wqa", inputs.descale_wqa, paired_scale_shape(rank_size, hidden_size)),
        ("descale_wqb", inputs.descale_wqb,
         paired_scale_shape(head_count * head_size, rank_size)),
        ("descale_wkv", inputs.descale_wkv, paired_scale_shape(head_size, hidden_size)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")

    if tuple(inputs.cache_index.shape) != (token_count,):
        raise ValueError("cache_index must have shape [T]")
    if inputs.cache_index.dtype != torch.int64:
        raise TypeError("cache_index must use int64")
    if inputs.kv_cache.ndim != 4 or tuple(inputs.kv_cache.shape[2:]) != (1, head_size):
        raise ValueError("kv_cache must have shape [blocks, block_size, 1, D]")
    if inputs.kv_cache.dtype != torch.uint8:
        raise TypeError("kv_cache must use uint8 storage")
    if tuple(inputs.norm_weight_qr.shape) != (rank_size,):
        raise ValueError("norm_weight_qr must have shape [R]")
    if tuple(inputs.norm_weight_kv.shape) != (head_size,):
        raise ValueError("norm_weight_kv must have shape [D]")
    if tuple(inputs.rope_sin.shape) != tuple(inputs.rope_cos.shape):
        raise ValueError("rope_sin and rope_cos shapes must match")
    if inputs.rope_sin.shape[0] != token_count:
        raise ValueError("RoPE tables must have T rows")
    if inputs.rope_sin.shape[1] > head_size or inputs.rope_sin.shape[1] % 2:
        raise ValueError("Dr must be even and at most D")

    for dimension, name in ((hidden_size, "H"), (rank_size, "R"), (head_size, "D")):
        if dimension % 64:
            raise ValueError(f"{name} must be a multiple of 64")

    # cache_index 域：0 <= index < P*BS，且唯一（重复值语义未定义，决策 D3）
    slot_count = inputs.kv_cache.shape[0] * inputs.kv_cache.shape[1]
    if bool((inputs.cache_index < 0).any()) or bool((inputs.cache_index >= slot_count).any()):
        raise IndexError("cache_index contains an invalid slot")
    if inputs.cache_index.unique().numel() != token_count:
        raise ValueError("cache_index values must be unique")

    return token_count, hidden_size, rank_size, head_count, head_size


# --------------------------------------------------------------------------
# 主 golden
# --------------------------------------------------------------------------

def attn_prologue_golden(
    inputs: AttnPrologueInputs,
    norm_eps: float,
    *,
    convention: RopeConvention = RopeConvention.INTERLEAVE_HALF,
    trace: bool = True,
    row_chunk: int = 512,
    compute_dtype: torch.dtype = torch.float32,
) -> AttnPrologueTrace:
    """完整的 attn_prologue CPU 参考。

    ``row_chunk`` 只影响内存占用，**不影响任何数值**：所有阶段都是逐 token
    独立的（RMSNorm 沿特征轴、RoPE 逐 token、scatter 逐行）。
    ``compute_dtype=torch.float64`` 给出 fp64 影子实现（03 文档 §7.1）。
    """
    token_count, _, rank_size, head_count, head_size = _validate_inputs(inputs)
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    # 权重只反量化一次（与 T 无关）
    wqa = dequantize_mxfp8(inputs.wqa, inputs.descale_wqa, compute_dtype)
    wqb = dequantize_mxfp8(inputs.wqb, inputs.descale_wqb, compute_dtype)
    wkv = dequantize_mxfp8(inputs.wkv, inputs.descale_wkv, compute_dtype)
    gamma_qr = inputs.norm_weight_qr
    gamma_kv = inputs.norm_weight_kv

    kv_cache = inputs.kv_cache.clone()
    block_size = int(inputs.kv_cache.shape[1])
    # [P,BS,1,D] → [P*BS,D] 的零拷贝视图，scatter 直接写它
    cache_2d = kv_cache.view(-1, head_size)

    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "q", "qr", "descale_qr", "x_deq", "qa_fp32", "qr_fp32",
            "qr_normalized", "q_proj_fp32", "q_roped_fp32",
            "kv_fp32", "kv_normed", "kv_roped", "kv_bytes",
        )
    }

    for start in range(0, token_count, row_chunk):
        stop = min(start + row_chunk, token_count)
        rows = slice(start, stop)

        # [输入解码] x 的 MXFP8 → FP
        x_deq = dequantize_mxfp8(
            inputs.x[rows],
            inputs.descale_x[rows],
            compute_dtype,
        )
        sin = inputs.rope_sin[rows]
        cos = inputs.rope_cos[rows]

        # ---------------- Q 支路 ----------------
        # [QA] H → R 低秩下投影
        qa = x_deq @ wqa.T
        # [RMSNorm] 沿 R
        qr_fp32 = rms_norm(qa, gamma_qr, norm_eps, compute_dtype)
        # [动态 MXFP8 量化] 同时产出 payload 与 descale
        qr, descale_qr, qr_normalized = quantize_mxfp8(qr_fp32, compute_dtype)
        # [QB] ★顺序关键★ A 操作数是量化后的 qr，不是 qr_fp32（03 文档 §2.1）
        q_proj = dequantize_mxfp8(qr, descale_qr, compute_dtype) @ wqb.T
        # [reshape] N*D → (N,D)，然后尾部 RoPE 在 N 上广播
        q_heads = q_proj.reshape(stop - start, head_count, head_size)
        q_roped = apply_tail_rope(
            q_heads, sin, cos, convention=convention, compute_dtype=compute_dtype
        )
        q = q_roped.to(torch.bfloat16)

        # ---------------- KV 支路 ----------------
        # [KV] 同一份 x，H → D
        kv = x_deq @ wkv.T
        # [RMSNorm] 沿 D
        kv_normed = rms_norm(kv, gamma_kv, norm_eps, compute_dtype)
        # ★顺序关键★ KV 是先 RoPE 再量化（03 文档 §2.2）
        kv_roped = apply_tail_rope(
            kv_normed, sin, cos, convention=convention, compute_dtype=compute_dtype
        )
        # [量化] 隐含 scale = 1，饱和到 ±448；★bitcast 不是数值 cast★
        kv_payload = kv_roped.clamp(-E4M3FN_MAX, E4M3FN_MAX).to(torch.float8_e4m3fn)
        kv_bytes = kv_payload.contiguous().view(torch.uint8)

        # [scatter] 只就地更新 cache_index 命中的行
        cache_2d[inputs.cache_index[rows]] = kv_bytes

        parts["q"].append(q)
        parts["qr"].append(qr)
        parts["descale_qr"].append(descale_qr)
        if trace:
            parts["x_deq"].append(x_deq)
            parts["qa_fp32"].append(qa)
            parts["qr_fp32"].append(qr_fp32)
            parts["qr_normalized"].append(qr_normalized)
            parts["q_proj_fp32"].append(q_proj)
            parts["q_roped_fp32"].append(q_roped)
            parts["kv_fp32"].append(kv)
            parts["kv_normed"].append(kv_normed)
            parts["kv_roped"].append(kv_roped)
            parts["kv_bytes"].append(kv_bytes)

    def joined(name: str) -> torch.Tensor | None:
        chunks = parts[name]
        return torch.cat(chunks, dim=0) if chunks else None

    return AttnPrologueTrace(
        q=joined("q"),
        qr=joined("qr"),
        descale_qr=joined("descale_qr"),
        kv_cache=kv_cache,
        x_deq=joined("x_deq"),
        qa_fp32=joined("qa_fp32"),
        qr_fp32=joined("qr_fp32"),
        qr_normalized=joined("qr_normalized"),
        q_proj_fp32=joined("q_proj_fp32"),
        q_roped_fp32=joined("q_roped_fp32"),
        kv_fp32=joined("kv_fp32"),
        kv_normed=joined("kv_normed"),
        kv_roped=joined("kv_roped"),
        kv_bytes=joined("kv_bytes"),
    )


# --------------------------------------------------------------------------
# E4M3 失配归因（03 文档 §4.3）
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class E4M3Attribution:
    """把 E4M3 字节失配拆成"量化边界 tie-break"与"真失配"。"""

    total: int
    mismatched: int
    tie_break: int
    hard: int
    hard_index: tuple[torch.Tensor, ...]
    max_code_distance: int

    @property
    def tie_break_ratio(self) -> float:
        return self.tie_break / self.total if self.total else 0.0

    def report(self) -> str:
        lines = [
            f"elements={self.total} mismatched={self.mismatched} "
            f"tie_break={self.tie_break} ({self.tie_break_ratio:.3e}) hard={self.hard}",
            f"max_code_distance={self.max_code_distance}",
        ]
        if self.hard:
            head = [tuple(int(axis[i]) for axis in self.hard_index)
                    for i in range(min(8, self.hard))]
            lines.append(f"first hard mismatches at {head}")
        return "\n".join(lines)


def _decode_e4m3_bytes(raw: torch.Tensor) -> torch.Tensor:
    """uint8 字节 → FP32 数值（零拷贝 reinterpret）。"""
    return raw.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def attribute_e4m3_mismatch(
    got_bytes: torch.Tensor,
    expected_bytes: torch.Tensor,
    expected_value: torch.Tensor,
    *,
    boundary_rel: float = 1e-5,
) -> E4M3Attribution:
    """归因 E4M3 payload 的逐字节失配。

    独立 golden **无法**保证 `qr` / `kv_cache` 逐位相等：Cube 按 BK 分块做
    FP32 累加，torch 是单次归约，相对差约 1e-6。这个差要穿过只有 3 位尾数
    的 E4M3 量化，在 round-to-nearest 的判定边界上就会翻到相邻编码。

    一个失配被判为合法 tie-break 需同时满足：
      1. ``expected_value`` 落在两个编码解码值之间（闭区间）——即 got 是
         朝着真值方向的**相邻**编码，不是随机字节；
      2. ``expected_value`` 到两编码中点的相对距离 ≤ ``boundary_rel``——即
         真值本来就贴在判定边界上。

    ``expected_value`` 必须是**实际被 cast 的值**：`qr` 传
    ``trace.qr_normalized``，`kv_cache` 传 ``trace.kv_roped``。

    ``hard_index`` 非空即真 bug，不是浮点问题。
    """
    if got_bytes.shape != expected_bytes.shape:
        raise ValueError("byte tensors must have the same shape")
    if got_bytes.shape != expected_value.shape:
        raise ValueError("expected_value must have the same shape as the bytes")
    got_bytes = got_bytes.cpu().contiguous()
    expected_bytes = expected_bytes.cpu().contiguous()
    reference = expected_value.detach().cpu().to(torch.float32)

    total = int(got_bytes.numel())
    diff_mask = got_bytes != expected_bytes
    mismatched = int(diff_mask.sum())
    if mismatched == 0:
        return E4M3Attribution(total, 0, 0, 0, tuple(), 0)

    index = torch.nonzero(diff_mask, as_tuple=True)
    got_value = _decode_e4m3_bytes(got_bytes)[index]
    exp_value = _decode_e4m3_bytes(expected_bytes)[index]
    truth = reference[index]

    lower = torch.minimum(got_value, exp_value)
    upper = torch.maximum(got_value, exp_value)
    between = (truth >= lower) & (truth <= upper)
    midpoint = (got_value + exp_value) * 0.5
    near = (truth - midpoint).abs() <= boundary_rel * truth.abs().clamp_min(1e-30)
    # NaN 编码（0x7F / 0xFF）永远不是合法 tie-break
    finite = torch.isfinite(got_value) & torch.isfinite(exp_value)
    tie = between & near & finite

    # 编码距离：合法 tie-break 应当是 1；>1 说明差得比"相邻"更远
    code_distance = (got_bytes.to(torch.int16) - expected_bytes.to(torch.int16)).abs()
    max_code_distance = int(code_distance[diff_mask].max())

    hard_mask = ~tie
    hard = int(hard_mask.sum())
    hard_index = tuple(axis[hard_mask] for axis in index)
    return E4M3Attribution(
        total=total,
        mismatched=mismatched,
        tie_break=int(tie.sum()),
        hard=hard,
        hard_index=hard_index,
        max_code_distance=max_code_distance,
    )


# --------------------------------------------------------------------------
# Fixture 构造（03 文档 §6：每条决策都在防一类具体 bug）
# --------------------------------------------------------------------------

def make_paired_scale(
    outer_size: int,
    reduction_size: int,
    generator: torch.Generator,
    *,
    exponent_range: tuple[int, int] = (-2, 3),
    zero_bytes: bool = False,
) -> torch.Tensor:
    """构造确定性、**非单位**的公开 paired E8M0 scale。

    全 1 scale 会同时掩盖"忘记乘 scale"、"pair 展平顺序错"、"组大小当成
    64 而不是 32"三类 bug，所以指数默认取 [-2, 2]。
    ``zero_bytes=True`` 额外把第 0 组的字节置 0——它代表 2^-127 而**不是**
    数值 0，是 kernel 侧需要特殊构造的边界。
    """
    groups = ceil_div(reduction_size, MX_GROUP_SIZE)
    pair_groups = ceil_div(groups, MX_SCALE_PAIR) * MX_SCALE_PAIR
    exponents = torch.randint(
        exponent_range[0], exponent_range[1],
        (outer_size, pair_groups), dtype=torch.int16, generator=generator,
    )
    encoded = _encode_e8m0_exponents(exponents)
    if zero_bytes:
        encoded[:, 0] = 0
    return encoded.reshape(outer_size, pair_groups // MX_SCALE_PAIR, MX_SCALE_PAIR).contiguous()


def make_rope_tables(
    token_count: int,
    rope_width: int,
    start_position: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """连续位置的 interleave-half RoPE 表。"""
    positions = torch.arange(start_position, start_position + token_count)
    return make_rope_tables_for_positions(positions, rope_width)


def make_rope_tables_for_positions(
    positions: torch.Tensor,
    rope_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """给定绝对位置的 RoPE 表；``repeat_interleave(2)`` 与相邻配对约定一致。"""
    if rope_width % 2:
        raise ValueError("rope_width must be even")
    if positions.ndim != 1:
        raise ValueError("positions must be rank 1")
    frequencies = torch.pow(
        10000.0, -torch.arange(0, rope_width, 2).to(torch.float32) / rope_width
    )
    angles = positions.to(torch.float32)[:, None] * frequencies[None, :]
    return (
        angles.sin().repeat_interleave(2, dim=1),
        angles.cos().repeat_interleave(2, dim=1),
    )


def make_golden_inputs(
    token_count: int,
    *,
    hidden_size: int = 64,
    rank_size: int = 64,
    head_count: int = 2,
    head_size: int = 64,
    rope_width: int = 16,
    block_size: int = TYPICAL_BLOCK_SIZE,
    cache_slots: int | None = None,
    cache_index: torch.Tensor | None = None,
    start_position: int = 0,
    rope_positions: torch.Tensor | None = None,
    zero_scale_group: bool = False,
    force_zero_qr_group: bool = False,
    seed: int = 2026,
) -> AttnPrologueInputs:
    """确定性 fixture。

    ``cache_index`` 默认是**跨 block 的非连续置换**而不是 ``arange(T)``：
    ``arange`` 会让 ``slot // BS`` 恒为 0，完全测不到分页寻址。
    ``kv_cache`` 初值**随机填满**而不是全 0，否则"未命中 slot 被污染"和
    "多写了一行"都测不出来。
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cache_slots = max(token_count * 3, block_size * 3) if cache_slots is None else cache_slots
    block_count = ceil_div(cache_slots, block_size)
    slot_count = block_count * block_size
    if cache_index is None:
        if token_count > slot_count:
            raise ValueError("cache cannot hold token_count distinct slots")
        cache_index = torch.randperm(slot_count, generator=generator)[:token_count]
        cache_index = cache_index.to(torch.int64)

    def fp8(shape: tuple[int, ...]) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.125).to(torch.float8_e4m3fn)

    x = fp8((token_count, hidden_size))
    if force_zero_qr_group:
        # 让第 0 行整行为 0 ⇒ QA 第 0 行全 0 ⇒ 每个量化组都是零块，
        # 覆盖 log2(0) 陷阱与"零块 scale 字节必须为 0"的约定。
        x[0] = torch.zeros(hidden_size).to(torch.float8_e4m3fn)

    if rope_positions is None:
        rope_sin, rope_cos = make_rope_tables(token_count, rope_width, start_position)
    else:
        if tuple(rope_positions.shape) != (token_count,):
            raise ValueError("rope_positions must have shape [T]")
        rope_sin, rope_cos = make_rope_tables_for_positions(rope_positions, rope_width)

    return AttnPrologueInputs(
        x=x,
        wqa=fp8((rank_size, hidden_size)),
        wqb=fp8((head_count * head_size, rank_size)),
        wkv=fp8((head_size, hidden_size)),
        descale_x=make_paired_scale(token_count, hidden_size, generator,
                                   zero_bytes=zero_scale_group),
        descale_wqa=make_paired_scale(rank_size, hidden_size, generator),
        descale_wqb=make_paired_scale(head_count * head_size, rank_size, generator),
        descale_wkv=make_paired_scale(head_size, hidden_size, generator),
        # gamma 非全 1：否则"gamma 没乘"和"gamma 乘错轴"都被掩盖
        norm_weight_qr=0.75 + torch.rand(rank_size, generator=generator) * 0.5,
        norm_weight_kv=0.75 + torch.rand(head_size, generator=generator) * 0.5,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        cache_index=cache_index.clone(),
        kv_cache=torch.randint(
            0, 256, (block_count, block_size, 1, head_size),
            dtype=torch.uint8, generator=generator,
        ),
    )


# --------------------------------------------------------------------------
# 生产轴逻辑 case（80 decode + 20 prefill）
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TypicalCase:
    """下游生产轴 case 的轻量描述，不物化张量。"""

    mode: str
    batch: int
    new_tokens: int
    s2: int
    dspark: int | None = None
    block_size: int = TYPICAL_BLOCK_SIZE

    @property
    def token_count(self) -> int:
        """attn_prologue 实际消费的展平行数 T。"""
        return self.batch * self.new_tokens

    @property
    def case_id(self) -> str:
        suffix = "" if self.dspark is None else f"-dspark{self.dspark}"
        return (f"{self.mode}-b{self.batch}-s{self.new_tokens}"
                f"-s2{self.s2}-bs{self.block_size}{suffix}")


def iter_typical_cases() -> Iterator[TypicalCase]:
    """枚举全部 80 个 decode 与 20 个 prefill 逻辑 case。"""
    s2_values = (8192, 16384, 32768, 131072)
    for batch in (1, 4, 8, 16, 32):
        for new_tokens in (1, 6):
            for dspark in (1, 5):
                for s2 in s2_values:
                    yield TypicalCase("decode", batch, new_tokens, s2, dspark)
    for batch in (1, 4, 8, 16, 32):
        for s2 in s2_values:
            yield TypicalCase("prefill", batch, 8192, s2)


def build_typical_cache_index(case: TypicalCase) -> torch.Tensor:
    """为生产轴 case 构造合法的 KV cache 追加索引。

    ``dspark`` 在公开算子里没有映射，这里只当确定性的块置换系数：
    逻辑块乘一个与块数互质的 multiplier 映射到不同物理块，使两个 dspark
    标签产生两套不同的合法 scatter 地址；互质保证置换一一对应、不冲突。
    """
    if case.new_tokens > case.s2:
        raise ValueError("post-write s2 must be at least the number of new tokens")
    block_size = case.block_size
    blocks_per_sequence = ceil_div(case.s2, block_size)
    multiplier = 1 if case.dspark is None else case.dspark
    if math.gcd(multiplier, blocks_per_sequence) != 1:
        raise ValueError("dspark test permutation must be coprime to block count")
    slots: list[int] = []
    for batch_index in range(case.batch):
        for token_index in range(case.new_tokens):
            # 新 token 固定追加在每条序列末尾，逻辑位置 [s2-new, s2)
            logical = case.s2 - case.new_tokens + token_index
            logical_block, offset = divmod(logical, block_size)
            physical_within = (logical_block * multiplier) % blocks_per_sequence
            physical_block = batch_index * blocks_per_sequence + physical_within
            slots.append(physical_block * block_size + offset)
    result = torch.tensor(slots, dtype=torch.int64)
    if result.unique().numel() != result.numel():
        raise AssertionError("internal cache-index construction produced duplicates")
    return result


def build_typical_rope_positions(case: TypicalCase) -> torch.Tensor:
    """按展平顺序返回每个新 token 的绝对位置。"""
    if case.new_tokens > case.s2:
        raise ValueError("post-write s2 must be at least the number of new tokens")
    one_sequence = torch.arange(case.s2 - case.new_tokens, case.s2, dtype=torch.int64)
    return one_sequence.repeat(case.batch)


def distinct_token_counts(cases: Iterable[TypicalCase] | None = None) -> tuple[int, ...]:
    """逻辑 case 去重后的 T 集合（GEMM 形状只由 T 决定）。"""
    cases = iter_typical_cases() if cases is None else cases
    return tuple(sorted({case.token_count for case in cases}))
