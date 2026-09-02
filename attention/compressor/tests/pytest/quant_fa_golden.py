#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
MXFP8 Flash Attention Golden

功能：生成 BNSD 数据 → CPU golden 计算 → layout 转换 → 精度对比
支持：PA / 非PA 场景，rope 分离传入，GQA
量化：Q/K per-token-group (quant_mode=6), V per-channel-group (quant_mode=8)
输出：逐元素表格 + 统计汇总 (PctRlt 通过率，双千分之五标准)

"""

import argparse
import logging
import math
import struct

import numpy as np
import torch
import torch.nn as nn
import torch_npu

import libVALU as valu

op_info = valu.OpInfo()
status = valu.fpOpStatus()

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

try:
    from ml_dtypes import bfloat16 as np_bfloat16
except ImportError:
    np_bfloat16 = np.float32

try:
    from npu_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    _HAS_NPU = True
except Exception as e:
    logging.getLogger(__name__).warning("Failed to import npu_ops_transformer.ops: %s", e)
    _HAS_NPU = False

try:
    from . import result_compare_method
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 7=aclgraph
# 优先从环境变量读取，支持 GRAPH_PATH=7 python3 -m pytest ...
import os as _os

GRAPH_PATH = int(_os.environ.get("GRAPH_PATH", "0"))
B = 1
N_q = 1  # query heads
N_kv = 1  # kv heads
D = 128

ENABLE_ROPE = False
D_rope = 64

CU_SEQLENS_Q = [0, 128]
CU_SEQLENS_KV = [0, 1024]
SEQUSED_Q = [128]
SEQUSED_KV = [1024]
MAX_SEQLEN_Q = 128
MAX_SEQLEN_KV = 1024

# HiFP4 相关配置 (从 pytest_121_hif4/hif4_golden.py 搬运)
ACTUAL_SEQ_Q = [128]
ACTUAL_SEQ_KV = [1024]
HIFP4_GROUP_SIZE = 64

# NPU 算子 quant mode (HiFP4)
QUERY_QUANT_MODE = 6
KEY_QUANT_MODE = 6
VALUE_QUANT_MODE = 7

# .bin 输入路径 (空则随机生成)
BIN_DIR = ""
BIN_CASE_NAME = ""

hif4_values = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
hif4_values_dict = {0:"0", 0.25:"1", 0.5:"2", 0.75:"3", 1:"4", 1.25:"5", 1.5:"6", 1.75:"7", -0.25:"9", -0.5:"a", -0.75:"b", -1:"c", -1.25:"d", -1.5:"e", -1.75:"f"}


def _derive_seqused(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _get_seqused_q():
    return SEQUSED_Q if SEQUSED_Q is not None else _derive_seqused(CU_SEQLENS_Q)


def _get_seqused_kv():
    return SEQUSED_KV if SEQUSED_KV is not None else _derive_seqused(CU_SEQLENS_KV)


ENABLE_PA = False
BLOCK_SIZE = 512

SPARSE_MODE = 0
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 32

# Layout 选择
INPUT_LAYOUT = "BNSD"  # hif4 不支持 TND, 使用 BNSD
Q_SCALE_LAYOUT = "BNSD"  # 支持: TND, N2TGD
P_SCALE = 1.0

SOFTMAX_SCALE = None

# PA KV Cache Layout
# BnNBsD: [BlockNum, N, BlockSize, D]
# PA_NZ: fp8=[Bn,N,D//32,Bs,32], Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]
KV_CACHE_LAYOUT = "BnNBsD"

IS_CONTIGUOUS = True

ENABLE_LSE = False

# e8m0fnu 最小正数: 2^(-127)，用于替换 scale 中的 0 和非有限值
# e8m0fnu 没有 0 值语义，0 的 biased exponent 会被 NPU 解释为 NaN
E8M0_MIN_POSITIVE = 2 ** (-127)

SEED_Q = 54
SEED_K = 3
SEED_V = 4
SEED_QR = 8
SEED_KR = 9

DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0

DEVICE_ID = 0


def _get_npu_fa_kwargs():
    # 每次调用时读取最新全局变量，确保 pytest 修改配置后生效
    return dict(
        return_softmax_lse=ENABLE_LSE,
    )


# ==============================================================================
# 量化 scale 计算
# MXFP8 量化算法:
#   Q/K: per-token-group, 沿 D 维度按 group_size 分组，每组独立计算 shared exponent
#   V:   per-channel-group, 沿 S 维度按 group_size 分组，每组独立计算 shared exponent
#   quant_scale = 2^(floor(log2(max_abs)) - emax)，全零组 scale=1
#   量化: quantized = original / quant_scale
#   反量化: dequantized = quantized * dequant_scale, 其中 dequant_scale = quant_scale
# ==============================================================================

_EMAX_MAP = {
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 15,
}


def _validate_fp8_dtype(fp8_dtype):
    if fp8_dtype not in _EMAX_MAP:
        raise ValueError(
            f"{fp8_dtype} not supported, expected one of {list(_EMAX_MAP.keys())}"
        )


def get_mxfp8_per_token_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Vectorized Q/K per-token-group quant_scale."""
    _validate_fp8_dtype(fp8_dtype)
    emax_elem = _EMAX_MAP[fp8_dtype]

    dim1, dim2, dim3, dim4 = tensor.shape
    num_groups = math.ceil(dim4 / group_size)
    pad_size = num_groups * group_size - dim4
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))

    grouped = tensor.reshape(dim1, dim2, dim3, num_groups, group_size)
    all_zero_mask = torch.all(grouped == 0, dim=-1)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
    return torch.where(all_zero_mask, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )


def get_mxfp8_per_channel_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Vectorized V per-channel-group quant_scale."""
    _validate_fp8_dtype(fp8_dtype)
    emax_elem = _EMAX_MAP[fp8_dtype]

    dim1, dim2, dim3, dim4 = tensor.shape
    num_groups = math.ceil(dim3 / group_size)
    pad_size = num_groups * group_size - dim3
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size))

    grouped = tensor.reshape(dim1, dim2, num_groups, group_size, dim4)
    all_zero_mask = torch.all(grouped == 0, dim=-2)
    max_vals = torch.max(torch.abs(grouped), dim=-2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
    return torch.where(all_zero_mask, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )


def mxfp8_per_token_group_quant(tensor, quant_scale, group_size=32):
    dim4 = tensor.shape[-1]
    scale_expanded = quant_scale.repeat_interleave(group_size, dim=-1)[..., :dim4]
    return (tensor / scale_expanded.to(tensor.dtype)).to(torch.float32)


def mxfp8_per_channel_group_quant(tensor, quant_scale, group_size=32):
    dim3 = tensor.shape[2]
    scale_expanded = quant_scale.repeat_interleave(group_size, dim=2)[:, :, :dim3, :]
    return (tensor / scale_expanded.to(tensor.dtype)).to(torch.float32)


# ==============================================================================
# Layout 转换函数 - 数据 (Q/K/V)
# ==============================================================================


def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    """Q/K/V BNSD → 各种 layout，支持 fp8 tensor
    cu_seqlens: TND layout 时用于偏移量放置，T=cu_seqlens[-1]；None 时按 seq_lens 顺序紧凑排列
    """
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)

    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return (
            tensor[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N * D)
            .contiguous()
        )
    elif layout == "TND":
        if cu_seqlens is not None:
            T = cu_seqlens[-1]
            result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
            for b in range(B):
                act_s = seq_lens[b]
                offset = cu_seqlens[b]
                if act_s <= 0:
                    continue
                for n in range(N):
                    result[offset : offset + act_s, n, :] = tensor[b, n, :act_s, :]
            return result.contiguous()
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t : t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_kv_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    return convert_q_bnsd_to_layout(
        tensor_bnsd, seq_lens, layout, cu_seqlens=cu_seqlens
    )


def fill_tnd_padding(tensor_tnd, seq_lens, cu_seqlens, fill_value=float("inf")):
    """TND layout 中 cu_seqlens padding 区域（seqused[b] ~ cu_diff[b]）填充指定值
    用于 LSE: padding 位置填 inf 以匹配 NPU 行为
    """
    if cu_seqlens is None:
        return tensor_tnd
    B = len(seq_lens)
    for b in range(B):
        act_s = seq_lens[b]
        offset = cu_seqlens[b]
        cu_diff = cu_seqlens[b + 1] - cu_seqlens[b]
        if cu_diff > act_s:
            tensor_tnd[offset + act_s : offset + cu_diff] = fill_value
    return tensor_tnd


def convert_qk_rope_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    return convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout)


# ==============================================================================
# Layout 转换函数 - Scale (Q/K/V)
# ==============================================================================


def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 → e8m0fnu，提取 IEEE 754 biased exponent
    e8m0fnu 格式: 只有指数位，没有尾数，表示 2^(e-127)
    biased exponent = 0xFF 时表示 NaN
    返回 torch.float8_e8m0fnu 以匹配 def.cpp 中 DT_FLOAT8_E8M0 的 dtype 定义
    """
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp.view(torch.float8_e8m0fnu)


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0fnu 没有 0 值语义；非有限值进入 0xFF 会在 NPU 侧变 NaN。"""
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad_mask = ~torch.isfinite(result)
    zero_mask = result == 0
    bad_count = int(bad_mask.sum().item())
    zero_count = int(zero_mask.sum().item())
    if bad_count:
        logger.info(
            "[WARN] %s: replace %d non-finite scale values before e8m0 packing",
            name,
            bad_count,
        )
        result[bad_mask] = E8M0_MIN_POSITIVE
    if zero_count:
        result[zero_mask] = E8M0_MIN_POSITIVE
    return result


def fp32_to_e8m0fnu_safe(scale, name="scale"):
    scale_safe = sanitize_e8m0_scale(scale, name)
    packed = fp32_to_e8m0fnu(scale_safe)
    nan_byte_count = int((packed == 0xFF).sum().item())
    if nan_byte_count:
        raise ValueError(
            f"{name}: {nan_byte_count} values would become e8m0fnu NaN (0xFF)"
        )
    return packed


def canonical_q_scale_layout(layout):
    layout = (layout or "TND").upper()
    if layout not in ("TND", "N2TGD"):
        raise ValueError(f"Unsupported Q scale layout: {layout}")
    return layout


def resolve_q_scale_layout(layout=None):
    resolved = canonical_q_scale_layout(layout or Q_SCALE_LAYOUT)
    if N_kv <= 0 or N_q % N_kv != 0:
        raise ValueError(f"N_q must be divisible by N_kv, got N_q={N_q}, N_kv={N_kv}")
    group = N_q // N_kv
    return resolved, group


def pack_qk_scale_for_npu(scale_flat):
    """Q/K scale packing: (..., D) → (..., D//2, 2)
    NPU 要求 Q/K scale 按 (偶, 奇) 对打包，每两个相邻 scale 值合并为一个 [..., 2]
    """
    orig_shape = scale_flat.shape
    last_dim = orig_shape[-1]
    new_shape = orig_shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)


def pack_v_scale_for_npu(scale_flat):
    """V scale packing: (..., Sg, D) → (..., Sg//2, D, 2)
    NPU 要求 V scale 按行 (偶行, 奇行) 交错打包
    奇数行 pad 用 E8M0_MIN_POSITIVE，避免 0.0 转 e8m0fnu 后变 NaN
    """
    Sg = scale_flat.shape[-2]
    D = scale_flat.shape[-1]
    prefix_shape = scale_flat.shape[:-2]

    # 奇数行 pad 用 E8M0_MIN_POSITIVE，避免 0.0 转 e8m0fnu 后变 NaN
    if Sg % 2 != 0:
        pad_shape = prefix_shape + (1, D)
        pad = torch.full(
            pad_shape,
            E8M0_MIN_POSITIVE,
            dtype=scale_flat.dtype,
            device=scale_flat.device,
        )
        scale_flat = torch.cat([scale_flat, pad], dim=-2)
        Sg += 1

    out_Sg = Sg // 2
    out_shape = prefix_shape + (out_Sg, D, 2)

    result = torch.zeros(out_shape, dtype=scale_flat.dtype, device=scale_flat.device)
    result[..., 0] = scale_flat[..., ::2, :]
    result[..., 1] = scale_flat[..., 1::2, :]
    return result


def _convert_q_scale_bnsd_to_tnd(scale_bnsd, seq_lens, cu_seqlens=None):
    B, N, _, Dg = scale_bnsd.shape
    Dg_half = Dg // 2
    if cu_seqlens is not None:
        T = cu_seqlens[-1]
        result = torch.zeros(
            (T, N, Dg_half, 2), dtype=scale_bnsd.dtype, device=scale_bnsd.device
        )
        for b in range(B):
            act_s = seq_lens[b]
            offset = cu_seqlens[b]
            if act_s <= 0:
                continue
            for n in range(N):
                result[offset : offset + act_s, n, :, :] = scale_bnsd[
                    b, n, :act_s, :
                ].reshape(act_s, Dg_half, 2)
        return result
    T = sum(seq_lens)
    result = torch.zeros(
        (T, N, Dg_half, 2), dtype=scale_bnsd.dtype, device=scale_bnsd.device
    )
    t = 0
    for b in range(B):
        act_s = seq_lens[b]
        for n in range(N):
            result[t : t + act_s, n, :, :] = scale_bnsd[b, n, :act_s, :].reshape(
                act_s, Dg_half, 2
            )
        t += act_s
    return result


def convert_q_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, cu_seqlens=None):
    """Q scale BNSD → 各种 layout (已 packed: D//2, 2)"""
    layout = canonical_q_scale_layout(layout)
    B, N, _, Dg = scale_bnsd.shape
    max_org_s = max(seq_lens)
    Dg_half = Dg // 2

    if layout == "BNSD":
        return scale_bnsd[:, :, :max_org_s, :].reshape(B, N, max_org_s, Dg_half, 2)
    elif layout == "BSND":
        return (
            scale_bnsd[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N, Dg_half, 2)
        )
    elif layout == "BSH":
        return (
            scale_bnsd[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N * Dg_half, 2)
        )
    elif layout == "TND":
        return _convert_q_scale_bnsd_to_tnd(scale_bnsd, seq_lens, cu_seqlens=cu_seqlens)
    elif layout == "N2TGD":
        tnd_result = _convert_q_scale_bnsd_to_tnd(
            scale_bnsd, seq_lens, cu_seqlens=cu_seqlens
        )
        return convert_q_scale_tnd_to_n2tgd_layout(tnd_result, N_kv)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_k_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, cu_seqlens=None):
    return convert_q_scale_bnsd_to_layout(
        scale_bnsd, seq_lens, layout, cu_seqlens=cu_seqlens
    )


def convert_v_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, group_size=32):
    """V scale BNSD → 各种 layout
    V scale 偶奇行交错 packing: result[..., 0] = 偶数行, result[..., 1] = 奇数行
    奇数 Sg 时 pad 一行 E8M0_MIN_POSITIVE
    """
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)

    if actual_Sg % 2 != 0:
        actual_Sg_padded = actual_Sg + 1
    else:
        actual_Sg_padded = actual_Sg

    S_out = actual_Sg_padded // 2

    if layout == "BNSD":
        transposed = scale_bnsd[:, :, :actual_Sg, :]
        if actual_Sg % 2 != 0:
            pad = torch.full(
                (B, N, 1, D),
                E8M0_MIN_POSITIVE,
                dtype=transposed.dtype,
                device=transposed.device,
            )
            transposed = torch.cat([transposed, pad], dim=2)
        result = torch.zeros(
            (B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        result[..., 0] = transposed[..., ::2, :]
        result[..., 1] = transposed[..., 1::2, :]
        return result
    elif layout == "BSND":
        transposed = scale_bnsd[:, :, :actual_Sg, :].permute(0, 2, 1, 3)
        if actual_Sg % 2 != 0:
            pad = torch.full(
                (B, 1, N, D),
                E8M0_MIN_POSITIVE,
                dtype=transposed.dtype,
                device=transposed.device,
            )
            transposed = torch.cat([transposed, pad], dim=1)
        result = torch.zeros(
            (B, S_out, N, D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        result[..., 0] = transposed[:, ::2, :, :]
        result[..., 1] = transposed[:, 1::2, :, :]
        return result
    elif layout == "BSH":
        transposed = (
            scale_bnsd[:, :, :actual_Sg, :]
            .permute(0, 2, 1, 3)
            .reshape(B, actual_Sg, N * D)
        )
        if actual_Sg % 2 != 0:
            pad = torch.full(
                (B, 1, N * D),
                E8M0_MIN_POSITIVE,
                dtype=transposed.dtype,
                device=transposed.device,
            )
            transposed = torch.cat([transposed, pad], dim=1)
        result = torch.zeros(
            (B, S_out, N * D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        result[..., 0] = transposed[:, ::2, :]
        result[..., 1] = transposed[:, 1::2, :]
        return result
    elif layout == "TND":
        Tv = 0
        for seq_len in seq_lens:
            sg = math.ceil(seq_len / group_size)
            sg_padded = sg + (sg % 2)
            Tv += sg_padded // 2

        result = torch.zeros(
            (Tv, N, D, 2), dtype=torch.float32, device=scale_bnsd.device
        )
        t_start = 0
        for b in range(B):
            org_seq = seq_lens[b]
            sg = math.ceil(org_seq / group_size)
            sg_padded = sg + (sg % 2)
            act_s = sg_padded // 2
            t_end = t_start + act_s
            if act_s <= 0:
                continue
            for n in range(N):
                src = scale_bnsd[b, n, :sg, :]
                if sg % 2 != 0:
                    pad = torch.full(
                        (1, D), E8M0_MIN_POSITIVE, dtype=src.dtype, device=src.device
                    )
                    src = torch.cat([src, pad], dim=0)
                result[t_start:t_end, n, :, 0] = src[::2, :]
                result[t_start:t_end, n, :, 1] = src[1::2, :]
            t_start = t_end
        return result
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_q_scale_tnd_to_n2tgd_layout(tensor_tnd, num_kv_heads):
    """
    输入: (T, N_q, D//2, 2)
    输出: (N_kv, T, G, D//2, 2), G = N_q / N_kv
    N2TGD: N_kv 组，每组 G 个 query head 共享同一个 kv head
    """
    T, N, D, _ = tensor_tnd.shape
    G = N // num_kv_heads
    tensor_reshape = tensor_tnd.reshape(T, num_kv_heads, G, D, 2)
    return tensor_reshape.permute(1, 0, 2, 3, 4).contiguous()


def convert_q_scale_tnd_to_n2gtd_layout(tensor_tnd, num_kv_heads):
    return convert_q_scale_tnd_to_n2tgd_layout(tensor_tnd, num_kv_heads)


# ==============================================================================
# PA 格式转换 - mxfp8_pa_preprocessing
# ==============================================================================


def mxfp8_pa_preprocessing(
    tensor_bnsd,
    seq_lens,
    block_size,
    block_table,
    is_vscale=False,
    is_scale=False,
    is_rope=False,
    kv_layout="BnNBsD",
    group_size=32,
):
    """
    MXFP8 PA 预处理: BNSD → PagedAttention KV Cache

    输入: [B, N, S, D]
    输出 (is_scale=False): [BlockNum, N, BlockSize, D] (fp8 K/V)
    输出 (is_scale=True, is_vscale=False): [BlockNum, N, BlockSize, D//64, 2] (K scale)
    输出 (is_scale=True, is_vscale=True): [BlockNum, N, BlockSize//64, D, 2] (V scale)

    kv_layout:
      - BnNBsD: fp8=[Bn,N,Bs,D], Kscale=[Bn,N,Bs,D//64,2], Vscale=[Bn,N,Bs//64,D,2]
      - BnBsND: fp8=[Bn,Bs,N,D], Kscale=[Bn,Bs,N,D//64,2], Vscale=[Bn,Bs//64,N,D,2]
      - PA_NZ: fp8=[Bn,N,D//32,Bs,32], Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]
    """
    tensor_bnsd = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    B, N, S, D = tensor_bnsd.shape

    if is_rope and is_scale:
        raise ValueError("rope preprocessing does not support scale packing")

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_scale_pack_ratio = group_size * 2
            pack_seq_lens = [
                math.ceil(act_s / v_scale_pack_ratio) for act_s in seq_lens
            ]
            pack_block_size = math.ceil(block_size / v_scale_pack_ratio)
            total_block_num = sum(
                math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens
            )
            out_shape = (total_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K scale D must be even for packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (
                sum(math.ceil(act_s / block_size) for act_s in seq_lens),
                N,
                block_size,
                D // 2,
                2,
            )
    else:
        tensor_processed = tensor_bnsd
        pack_seq_lens = seq_lens
        pack_block_size = block_size
        out_shape = (
            sum(math.ceil(act_s / block_size) for act_s in seq_lens),
            N,
            block_size,
            D,
        )

    fill_value = E8M0_MIN_POSITIVE if is_scale else 0

    out_cache = torch.full(
        out_shape,
        fill_value,
        dtype=tensor_processed.dtype,
        device=tensor_processed.device,
    )
    block_num = [math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(block_num[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * pack_block_size
            valid_len = min(pack_block_size, pack_seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[blockid, :, :valid_len] = tensor_processed[
                b, :, block_offset : block_offset + valid_len
            ]

    if kv_layout == "BnNBsD":
        return out_cache
    elif kv_layout == "BnBsND":
        return out_cache.transpose(1, 2).contiguous()
    elif kv_layout == "PA_NZ":
        Bn, KV_N, Bs, KV_D = out_cache.shape[:4]
        if not is_scale:
            inner = 16 if is_rope else 32
            if KV_D % inner != 0:
                raise ValueError(
                    f"PA_NZ rope D must be divisible by {inner}, got {KV_D}"
                )
            reshaped = out_cache.reshape(Bn, KV_N, Bs, KV_D // inner, inner)
            return reshaped.permute(0, 1, 3, 2, 4).contiguous()
        elif not is_vscale:
            reshaped = out_cache.reshape(Bn, KV_N, Bs // 16, 16, KV_D, 2)
            return reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
        else:
            reshaped = out_cache.reshape(Bn, KV_N, Bs, KV_D // 16, 16, 2)
            return reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def convert_v_scale_to_pa(scale_bnsd, seq_lens, group_size=32):
    """
    V scale 偶奇行交错 packing，用于 PA 预处理
    输入: [B, N, Sg, D], Sg = ceil(orgS/32)
    输出: [B, N, Sg//2, D, 2] (奇数行 pad 到偶数)
    """
    scale_bnsd = (
        scale_bnsd
        if isinstance(scale_bnsd, torch.Tensor)
        else torch.as_tensor(scale_bnsd)
    )
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)

    transposed = scale_bnsd[:, :, :actual_Sg, :]

    if actual_Sg % 2 != 0:
        pad = torch.full(
            (B, N, 1, D),
            E8M0_MIN_POSITIVE,
            dtype=transposed.dtype,
            device=transposed.device,
        )
        transposed = torch.cat([transposed, pad], dim=2)
        actual_Sg += 1

    S_out = actual_Sg // 2
    result = torch.zeros(
        (B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device
    )
    result[..., 0] = transposed[..., ::2, :]
    result[..., 1] = transposed[..., 1::2, :]
    return result


def bnsd_to_pa_kv(tensor_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD"):
    return mxfp8_pa_preprocessing(
        tensor_bnsd,
        seq_lens,
        block_size,
        block_table,
        is_scale=False,
        kv_layout=kv_layout,
    )


def bnsd_to_pa_kv_scale_token_group(
    scale_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD", group_size=32
):
    return mxfp8_pa_preprocessing(
        scale_bnsd,
        seq_lens,
        block_size,
        block_table,
        is_scale=True,
        is_vscale=False,
        kv_layout=kv_layout,
        group_size=group_size,
    )


def bnsd_to_pa_v_scale_channel_group(
    scale_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD", group_size=32
):
    return mxfp8_pa_preprocessing(
        scale_bnsd,
        seq_lens,
        block_size,
        block_table,
        is_scale=True,
        is_vscale=True,
        kv_layout=kv_layout,
        group_size=group_size,
    )


def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result


# ==============================================================================
# 数据生成
# ==============================================================================


def generate_data():
    """生成 HiFP4 Q/K/V 与 scale
    返回:
      INPUT_LAYOUT == "BNSD" (默认):
        q_fp4 (B, N_q, Sq, D/2)            uint8
        k_fp4 (B, N_kv, Skv, D/2)          uint8
        v_fp4 (B, N_kv, Skv, D/2)          uint8
        dequant_scale_q (1, B, N_q, Sq, D/G) uint32 (5D, NPU 接口要求)
        k_antiquant_scale (1, B, N_kv, Skv, D/G) uint32
        v_antiquant_scale (1, B, N_kv, D, Skv/G) uint32
      INPUT_LAYOUT == "BSH":
        q_fp4 (B, Sq, N_q * D / 2)         uint8   (H=N_q*D, 物理存为 H/2 packed fp4)
        k_fp4 (B, Skv, N_kv * D / 2)       uint8
        v_fp4 (B, Skv, N_kv * D / 2)       uint8
        dequant_scale_q / k_antiquant_scale / v_antiquant_scale 同 BNSD (5D, N 维语义不变)
    """
    if BIN_DIR and BIN_CASE_NAME:
        return _load_data_from_bin()

    G = HIFP4_GROUP_SIZE
    if D % G != 0:
        raise ValueError(f"D={D} 必须是 G={G} 的整数倍")

    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV)

    # fp4_e1m2 分支
    choices = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
    # choices = [1]

    def gen_fp4_q(b, n, s, d, seed):
        ld = 1
        # np.random.seed(seed)
        input_i_data = np.random.choice(choices, size=(b, n, s, d)).astype(np_bfloat16)
        # for i in range(128//ld):
        #     for j in range(128//ld):
        #         dd = np.random.choice(choices).astype(np_bfloat16)
        #         input_i_data[:,:,i*ld:(i+1)*ld, j*ld:(j+1)*ld] = dd

        return new_trans_np_bfloat16_tensor_to_fp4_e1m2(input_i_data.astype(np_bfloat16))
    
    def gen_fp4_k(b, n, s, d, seed):
        ld = 1
        # np.random.seed(seed)
        input_i_data = np.random.choice(choices, size=(b, n, s, d)).astype(np_bfloat16)
        # for i in range(1024//ld):
        #     for j in range(128//ld):
        #         dd0 = np.random.choice(choices).astype(np_bfloat16)
        #         # dd1 = np.random.choice(choices).astype(np_bfloat16)

        #         input_i_data[:,:,i*ld:(i+1)*ld, j*ld:(j+1)*ld] = dd0
                # for k in range(1024//128):
                #     if k == 0:
                    # input_i_data[:,:,(k*128 + i*ld):(k*128 + (i+1)*ld), j*ld:(j+1)*ld] = dd0
                #     else: 
                #         input_i_data[:,:,(k*128 + i*ld):(k*128 + (i+1)*ld), j*ld:(j+1)*ld] = dd1

        return new_trans_np_bfloat16_tensor_to_fp4_e1m2(input_i_data.astype(np_bfloat16))

    def gen_fp4_v(b, n, s, d, seed):
        ld = 1
        # np.random.seed(seed)
        input_i_data = np.random.choice(choices, size=(b, n, s, d)).astype(np_bfloat16)
        # for i in range(1024//ld):
        #     for j in range(128//ld):
        #         dd = np.random.choice(choices).astype(np_bfloat16)
        #         input_i_data[:,:,i*ld:(i+1)*ld, j*ld:(j+1)*ld] = dd
                # for k in range(1024//128):
                    # input_i_data[:,:,(k*128 + i*ld):(k*128 + (i+1)*ld), j*ld:(j+1)*ld] = dd
        return new_trans_np_bfloat16_tensor_to_fp4_e1m2(input_i_data.astype(np_bfloat16))

    q_fp4 = gen_fp4_q(B, N_q, max_sq, D, SEED_Q)
    k_fp4 = gen_fp4_k(B, N_kv, max_skv, D, SEED_K)
    v_fp4 = gen_fp4_v(B, N_kv, max_skv, D, SEED_V)

    # hif4_scale 分支
    # dequant_scale_q = hifp4_scale_generate(np.zeros((1, B, N_q, max_sq, D // G), dtype=np.uint32), scale_range=(192,193))
    # k_antiquant_scale = hifp4_scale_generate(np.zeros((1, B, N_kv, max_skv, D // G), dtype=np.uint32), scale_range=(192,193))
    # v_antiquant_scale = hifp4_scale_generate(np.zeros((1, B, N_kv, D, math.ceil(max_skv / G)), dtype=np.uint32), scale_range=(192,193))
    dequant_scale_q = hifp4_scale_generate(np.zeros((1, B, N_q, max_sq, D // G), dtype=np.uint32), scale_range=(192,193))
    k_antiquant_scale = hifp4_scale_generate(np.zeros((1, B, N_kv, max_skv, D // G), dtype=np.uint32), scale_range=(187,198))
    v_antiquant_scale = hifp4_scale_generate(np.zeros((1, B, N_kv, D, math.ceil(max_skv / G)), dtype=np.uint32), scale_range=(192,193))

    # BSH 排布: 将 BNSD (B,N,S,H/2) 重排为 BSH (B,S,H/2)
    #   y_all = y_all.transpose(0, 2, 1, 3).reshape(B, S, H/2)
    if INPUT_LAYOUT == "BSH":
        logger.info("[INFO] generate_data: BSH 排布输出 Q/K/V")
        q_fp4 = np.ascontiguousarray(q_fp4.transpose(0, 2, 1, 3).reshape(B, max_sq, N_q * D // 2))
        k_fp4 = np.ascontiguousarray(k_fp4.transpose(0, 2, 1, 3).reshape(B, max_skv, N_kv * D // 2))
        v_fp4 = np.ascontiguousarray(v_fp4.transpose(0, 2, 1, 3).reshape(B, max_skv, N_kv * D // 2))

    return q_fp4, k_fp4, v_fp4, dequant_scale_q, k_antiquant_scale, v_antiquant_scale


def _load_data_from_bin():
    """
    bin 命名规则: {BIN_CASE_NAME}_input_{idx}.bin
      idx=0: Q (uint8, B*N_q*Q_S*D/2)
      idx=1: K (uint8, B*N_kv*KV_S*D/2)
      idx=2: V (uint8, B*N_kv*KV_S*D/2)
      idx=27: dequant_scale_query (fp32/uint32, 5D)
      idx=15: key_antiquant_scale  (fp32/uint32, 5D)
      idx=17: value_antiquant_scale (fp32/uint32, 5D)

    若 INPUT_LAYOUT == "BSH", Q/K/V reshape 为 (B,S,H/2); 否则按 BNSD (B,N,S,D/2)
    """
    base = os.path.join(BIN_DIR, BIN_CASE_NAME)
    G = HIFP4_GROUP_SIZE
    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV)

    if INPUT_LAYOUT == "BSH":
        q_fp4 = result_compare_method.load_hif4_bin_cpu(f"{base}_input_0.bin").reshape(B, max_sq, N_q * D // 2)
        k_fp4 = result_compare_method.load_hif4_bin_cpu(f"{base}_input_1.bin").reshape(B, max_skv, N_kv * D // 2)
        v_fp4 = result_compare_method.load_hif4_bin_cpu(f"{base}_input_2.bin").reshape(B, max_skv, N_kv * D // 2)
    else:
        q_fp4 = result_compare_method.load_hif4_bin_cpu(f"{base}_input_0.bin").reshape(B, N_q, max_sq, D // 2)
        k_fp4 = result_compare_method.load_hif4_bin_cpu(f"{base}_input_1.bin").reshape(B, N_kv, max_skv, D // 2)
        v_fp4 = result_compare_method.load_hif4_bin_cpu(f"{base}_input_2.bin").reshape(B, N_kv, max_skv, D // 2)

    dequant_scale_q = result_compare_method.load_fp32_bin_cpu(f"{base}_input_27.bin").view(np.uint32).reshape(
        1, B, N_q, max_sq, D // G)
    k_antiquant_scale = result_compare_method.load_fp32_bin_cpu(f"{base}_input_15.bin").view(np.uint32).reshape(
        1, B, N_kv, max_skv, D // G)
    v_antiquant_scale = result_compare_method.load_fp32_bin_cpu(f"{base}_input_17.bin").view(np.uint32).reshape(
        1, B, N_kv, D, max_skv // G)

    return q_fp4, k_fp4, v_fp4, dequant_scale_q, k_antiquant_scale, v_antiquant_scale


# ==============================================================================
# CPU Golden (HiFP4)
# 从 pytest_121_hif4/hif4_golden.py 搬运, 计算逻辑保持原样
# ==============================================================================


def E6M2_bits_to_float(code: int) -> float:
    if code == 0xFF:
        return float('nan')

    exp_storage = int(code >> 2)          # 6 位指数存储值 (0~63)
    m_val = code & 0x3                # 2 位尾数编码 (0~3)

    # 实际指数 = 存储指数 - 48
    exp = exp_storage - 48
    # 尾数值 = 1 + m_val / 4
    mantissa = 1.0 + m_val / 4.0

    return mantissa * (2.0 ** exp)


def split_32bit(combined: int):
    # 提取各段
    # e6m2_code = (combined >> 24) & 0xFF
    # bits8_val = (combined >> 16) & 0xFF
    # bits16_val = combined & 0xFFFF

    bits16_val = combined >> 16  & 0xFFFF
    bits8_val =  combined >> 8   & 0xFF
    e6m2_code =  combined        & 0xFF

    # 将 8 位整数转换为高位在前的 0/1 列表
    bits8_list = [(bits8_val >> (7 - i)) & 1 for i in range(8)]
    # # 将 8 位整数转换为高位在前的 0/1 列表
    # bits8_list = [(bits8_val >> i) & 1 for i in range(8)]
    # 将 16 位整数转换为高位在前的 0/1 列表
    bits16_list = [(bits16_val >> (15 - i)) & 1 for i in range(16)]

    # print("-- bits16_list")
    # print(bits16_list)

    # 将 E6M2 编码转换为浮点数
    e6m2_float = E6M2_bits_to_float(e6m2_code)

    return e6m2_float, bits8_list, bits16_list


def HiFP4_To_BF16_reverse_order(in_tensor_hifp4, GHifp4Scale_, G = 64):
    print(f"-- start HiFP4_To_BF16_reverse_order --")
    GHifp4Scale = GHifp4Scale_.view(np.int32)
    a = np.array(in_tensor_hifp4)
    Mi = a.shape[0]
    Ni = a.shape[1]
    Mcnt = np.ceil(Mi/G).astype(int)
    # assert GHifp4Scale.dtype == np.uint32, f"期望GHifp4Scale类型为 np.uint32，但实际为 {GHifp4Scale.dtype}"
    GE6M2  = np.zeros((Mcnt,   Ni))  # level-1 Scale Factor
    GE1_8  = np.zeros((Mcnt*8, Ni))  # level-2 EXP
    GE1_16 = np.zeros((Mcnt*16,Ni))  # level-3 EXP
    print("-- GHifp4Scale:", GHifp4Scale.dtype, GHifp4Scale.shape)
    for i in range(Mcnt):
        for j in range(Ni):
            e6m2, bits8, bits16 = split_32bit(GHifp4Scale[i][j])
            GE6M2 [i,            j] = e6m2
            GE1_8 [i*8:i*8+8,    j] = bits8[::-1]
            GE1_16[i*16:i*16+16, j] = bits16[::-1]
    x_pf16 = np.zeros((Mi, Ni))
    for i in range(Mcnt):
        for j in range(Ni):
            x = a[i*G : min((i+1)*G, Mi), j]
            # GE6M2 中当前组当前列的值（标量）
            E6M2 = GE6M2[i, j]
            E1_8 = GE1_8[i*8 : (i+1)*8, j]
            DE64_8 = np.repeat(E1_8, 8)
            E1_16 = GE1_16[i*16 : (i+1)*16, j]
            DE64_16 = np.repeat(E1_16, 4)
            de64 = DE64_8 + DE64_16
            grp = E6M2 * (2.0 ** de64[:len(x)]) * x
            # 存入结果矩阵
            x_pf16[i*G : min((i+1)*G, Mi), j] = grp
    x_pf16 = x_pf16.astype(np_bfloat16)
    print(f"-- end HiFP4_To_BF16_reverse_order --")
    return x_pf16


# ==============================================================================
# BSH 排布相关工具函数
# ==============================================================================

def trans_bnsd_to_bsh(tensor, shape):
    B = shape[0]
    N = shape[1]
    S = shape[2]
    D = shape[3]
    H = N * D
    return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)


def np_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
    cpu golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = np.zeros((B, S, N, D), dtype=tensor.dtype)
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        isHif4 = True
        if (isHif4 == True):
            return tensor.reshape(B, S, N, int(D/2)).transpose(0, 2, 1, 3), [B, N, S, D]
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "BSND":
        print("BSND")
        B = bsh_shape[0]
        S = bsh_shape[1]
        N = bsh_shape[2]
        D = bsh_shape[3]
        return tensor.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "TND" or inputLayout == "NTD_TND":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        if inputLayout=="NTD_TND":
            N = bsh_shape[0]
            tensor = tensor.transpose(1, 0, 2)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print(inputLayout)
        return tensor, bsh_shape


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def IsRoundOne(sign, man, truncLen):
    roundingTruncLen = 64
    # mask0 = (truncLen >= roundingTruncLen) ? 0 : 0x1ul << truncLen;
    if truncLen >= roundingTruncLen:
        mask0 = 0
    else:
        mask0 = 0x1 << truncLen
    # mask1 = (truncLen > roundingTruncLen) ? 0 : 0x1ul << (truncLen - 1);
    if (truncLen > roundingTruncLen):
        mask1 = 0
    else:
        mask1 = 0x1 << (truncLen - 1)

    mask2 = mask1 - 1

    # ROUND_TO_NEAREST
    lastBit = (man & mask0) > 0  # Last bit after conversion
    truncHighBit = (man & mask1) > 0  # Highest bit in the truncated part
    truncLeft = (man & mask2) > 0  # Truncated left part (except for the highest bit)
    return truncHighBit and (truncLeft or lastBit)


def cvt_bfloat16_to_fp4_e1m2(x):
    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    x = eval(float_to_hex(x_abs))
    x = x >> 16
    # print("x=========",x)

    ef = x >> 7 & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 2  #
    maxExp = 1  # max E encoding value of e1m2 is 3
    expBias = 1  # Exponent Bias value of e2m1/e1m2 is 1

    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0):
        eNorm = ef - 127 + 1  # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127  # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta)  # determine if need to carry
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
    else:
        eRet = (eNorm + expBias)
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
        if (((mRet & 0b100) != 0) and (needRound)):
            eRet += 1
            mRet = 0

    if (eRet >= 1):
        eRet = 1
    elif (eRet == 0 and mRet == 0b100):
        eRet += 1
        mRet = 0

    # print("eRet=========",eRet)
    # print("mRet=========",mRet)
    return (((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3))


def cvt_fp4_e1m2_to_bfloat16_new(x):
    Fp4e1m2ToBf16 = {'0': 0x0, '1': 0x3E80, '2': 0x3F00, '3':0x3F40,
                     '4': 0x3F80, '5': 0x3FA0, '6': 0x3FC0, '7':0x3FE0,
                     '8': 0x8000, '9': 0xBE80, '10': 0xBF00, '11':0xBF40,
                     '12': 0xBF80, '13': 0xBFA0, '14': 0xBFC0, '15':0xBFE0}

    x = int(x)
    first_fp4val = x & 0x0f
    second_fp4val = (x >> 4 )& 0x0f
    first_fp4str = str(first_fp4val)
    second_fp4str = str(second_fp4val)

    return Fp4e1m2ToBf16[first_fp4str], Fp4e1m2ToBf16[second_fp4str]


def new_trans_np_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    new_in_tensor = in_tensor.reshape(multi_shape)
    out_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e1m2(new_in_tensor[i])
    out_tensor = out_tensor.astype(np.uint8)

    # 每两个fp4拼成一个uint8保存
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape//2).astype(np.uint8)
    for i in range(multi_shape // 2):
        # fp4_tensor[i] = (out_tensor[i*2] << 4) | out_tensor[i*2+1] # 按常规顺序保存b4
        fp4_tensor[i] = (out_tensor[i*2+1] << 4) | out_tensor[i*2] # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100
    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor


def new_trans_np_fp4_e1m2_tensor_to_bfloat16_new(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)

    # 1个uint8包含两个fp4, 先拆成两个uint8
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape  * 2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i*2], bfloat16_tensor[i*2+1] = cvt_fp4_e1m2_to_bfloat16_new(in_tensor[i])
        fp32_tensor[i*2] = struct.unpack('!f',struct.pack('!I',bfloat16_tensor[i*2].item()<<16))[0]
        fp32_tensor[i*2+1] = struct.unpack('!f',struct.pack('!I',bfloat16_tensor[i*2+1].item()<<16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


# ==============================================================================
# HiFP4 scale 生成
# ==============================================================================

def hifp4_scale_generate(input_scale, params=None, scale_range=(190, 220)):
    """
    入参兼容: 若传入 params (dict, 含 range_input[3]) 则按原逻辑, 否则用 scale_range
    """
    if params is not None and 'range_input' in params:
        scale_data_range = params['range_input'][3]
        print(f"hif4 scale generated: shape = {input_scale.shape}, data_range = {scale_data_range}")
        pertoken_scale = np.random.randint(*scale_data_range, size=input_scale.shape, dtype=np.uint32)
    else:
        pertoken_scale = np.random.randint(scale_range[0], scale_range[1],
                                           size=input_scale.shape, dtype=np.uint32)

    # random_bits = np.random.randint(0, 2**24-0, size=input_scale.shape, dtype=np.uint32)
    # random_upper = random_bits << 8
    lower_bits_pertoken_scale = pertoken_scale & 0xFF
    pertoken_scale[:] = lower_bits_pertoken_scale
    return pertoken_scale


# ==============================================================================
# CPU Golden
# ==============================================================================
def fp32_to_fp16_manual(x: float) -> str:
    """
    fp32 -> fp16 大端4位大写十六进制
    IEEE RTNE 就近偶数舍入，处理尾数进位溢出（解决误差0.5/1.0问题）
    """
    fp32_bits = struct.unpack("<I", struct.pack("<f", x))[0]
    s = (fp32_bits >> 31) & 1
    e = (fp32_bits >> 23) & 0xFF
    m = fp32_bits & 0x7FFFFF
    BIAS_FP32 = 127
    BIAS_FP16 = 15

    if e == 0xFF:
        # Inf / NaN
        new_e = 0x1F
        new_m = m >> 13
    elif e == 0:
        # FP32 次正规，直接归零
        new_e = 0
        new_m = 0
    else:
        new_e = e - BIAS_FP32 + BIAS_FP16
        round_bit = 1 << 12  # 舍入half bit
        rounded_m = m + round_bit
        new_m = rounded_m >> 13

        # ========== 修复关键：尾数进位溢出，指数+1 ==========
        if new_m > 0x3FF:
            new_m = 0
            new_e += 1

        if new_e >= 0x1F:
            # 上溢：无穷大
            new_e = 0x1F
            new_m = 0
        elif new_e <= 0:
            # FP16 次正规处理（带舍入）
            shift = 1 - new_e
            m_norm = m | 0x800000
            rounded_sub = m_norm + (1 << (shift - 1))
            m_sub = rounded_sub >> shift
            new_m = m_sub >> 13
            new_e = 0

    fp16_bits = (s << 15) | ((new_e & 0x1F) << 10) | (new_m & 0x3FF)
    return f"{fp16_bits:04X}"

def hex4_to_fp16(hex_str: str, big_endian: bool = True) -> float:
    """
    4位十六进制串解析FP16
    :param hex_str: 4个十六进制字符，例如 "4000"
    :param big_endian: True=大端，False=小端（内存dump常用）
    :return: fp16对应的浮点数
    """
    raw = bytes.fromhex(hex_str)
    fmt = ">e" if big_endian else "<e"
    return struct.unpack(fmt, raw)[0]

def _np_pfaattention_act_fp4_new(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, pfa_param,
                             dequantScale_query_cur, k_antiquant_scale_cur, v_antiquant_scale_cur, dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    """
    CPU HiFP4 Flash Attention Golden

    =======================================================================
    1. [修改-流水结构] 原: 单轮 C1V1C2V1 逐 K 块循环, S1=全Sq, S2=128
                       → 新: 8轮C1V1 + 1轮C2V2 (参照 mxfp8 C1V1C1V1C2V2 扩展为 8 块)
       - Q 按 Q_BLOCK_SIZE=128 分块 (S1=128)
       - 每轮 C1V1: S1=128, S2=128 (K_BLOCK_SIZE=128)
       - 8 轮 C1V1 结果合并为大的基本块, 1 轮 C2V2: S1=128, S2=1024 (V_BLOCK_SIZE=8*128=1024)
    2. [修改-P量化]   原: HiFP4 量化 (To_HiFP4_reverse_order_new + HiFP4_To_BF16_P_reverse_order)
                       → 新: mxfp8 量化 (FP8_E4M3, 静态 P_SCALE=1.0, 参照 mxfp8_golden.py)
    3. [修改-softmax] 原: findMax + softmax_flashv2_amla 逻辑
                       → 新: 参照 mxfp8 _online_softmax_update, m 对齐到 ln2 整数倍 (ceil)
    4. [修改-C2V2]    原: 单块 P 与 V matmul
                       → 新: 参照 mxfp8 多块 P rescale 逻辑, 8 个 P 块 (S1=128,S2=128)
                             分别 rescale 后合并为大基本块 (S1=128,S2=1024) PV matmul
    5. [删除-mask]    hif4 不需要 mask (参照 mxfp8 _online_softmax_update 删除 mask_j 填 -inf 逻辑)
    6. [删除-尾块]    S2 不考虑尾块, 要求 Skv = 8 * K_BLOCK_SIZE = 1024 的整数倍;
                             S1 按 Q_BLOCK_SIZE 分块, 尾块用 min 处理
    =======================================================================
    """
    print("****************************** start _np_pfaattention_act_fp4 ******************************")
    G = 64  # HiFP4 分组大小

    # ====== 修改: mxfp8 P 量化参数 (静态量化, 参照 mxfp8_golden.py) ======
    FP8_DTYPE = torch.float8_e4m3fn   # mxfp8 P 输出数据类型 (参照 mxfp8_golden.py FP8_DTYPE)
    LN2 = math.log(2.0)               # ln2, 用于 m 对齐 (参照 mxfp8_golden.py LN2)
    P_SCALE = 1.0                     # P 静态量化 scale (参照 mxfp8_golden.py P_SCALE)
    ln_p_scale = math.log(P_SCALE)    # ln(p_scale) (参照 mxfp8_golden.py ln_p_scale)
    EPSILON = 1e-20

    # ====== 修改: 块大小, 8轮C1V1 + 1轮C2V2, C1V1 基本块 S1=128 S2=128, C2V2 大基本块 S1=128 S2=1024 ======
    Q_BLOCK_SIZE = 128    # S1=128, Q 分块大小 (参照 mxfp8 Q_BLOCK_SIZE=128)
    K_BLOCK_SIZE = 128    # S2=128, 每次 C1V1 处理 128 个 K token
    V_BLOCK_SIZE = 1024   # S2=1024 = 8 * K_BLOCK_SIZE, 每次 C2V2 处理 1 个大 V block
    NUM_C1V1 = 8          # 8 轮 C1V1 (8 * 128 = 1024)

    qs_begin = 0
    qs_end = q_tensor.shape[2]  # s1 = Sq
    kvs_begin = 0
    kvs_end = k_tensor.shape[2]  # s2 = Skv

    print("vscale:")
    print(v_antiquant_scale_cur[0,0])

    vscale_nz = np.zeros((32, 16))
    for f in range(4):
        print("vscale nz:")
        vscale_nz = np.zeros((32, 16))
        for i in range(128 // 16):
            for j in range(256 // 64):
                # print(f"[{i}][{j}]")
                vscale_nz[4*i+j] = v_antiquant_scale_cur[0,0,i*16:(i+1)*16,(4*f+j):(4*f+j+1)].reshape(1, 16)
        for k in range(32):
            print(f"[{' ,'.join([str(x) for x in vscale_nz[k]])}],")

    # HiFP4 → BF16 反量化 (Q/K/V 数据反量化保持原 hif4 逻辑不变)
    q_tensor = new_trans_np_fp4_e1m2_tensor_to_bfloat16_new(q_tensor)
    k_tensor = new_trans_np_fp4_e1m2_tensor_to_bfloat16_new(k_tensor)
    v_tensor = new_trans_np_fp4_e1m2_tensor_to_bfloat16_new(v_tensor)
    v_antiquant_scale_cur = v_antiquant_scale_cur.transpose(0, 1, 3, 2)

    S = kvs_end - kvs_begin   # Skv
    qs = q_tensor.shape[2]    # Sq
    qd = q_tensor.shape[3]    # D
    kd = k_tensor.shape[3]    # D
    vd = v_tensor.shape[3]    # D

    TILES_Q = (qs + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE          # Sq / 128
    TILES_KV = (S + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE          # Skv / 128
    TILES_V = (S + V_BLOCK_SIZE - 1) // V_BLOCK_SIZE           # Skv / 1024

    # ====== 修改: S2 不考虑尾块, 要求 Skv = NUM_C1V1 * K_BLOCK_SIZE = 1024 的整数倍 ======
    assert S % V_BLOCK_SIZE == 0, \
        f"hif4 S2 不考虑尾块, 要求 Skv({S}) = V_BLOCK_SIZE({V_BLOCK_SIZE}) 的整数倍"

    # ====== 修改: Q 反量化只做一次 (移到循环外, 全 Sq 一次性反量化) ======
    q_tensor[0, 0, :, :] = (HiFP4_To_BF16_reverse_order(
        q_tensor[0, 0, :, :].copy().T, dequantScale_query_cur[0, 0, :, :].copy().T)).T

    # ====== 修改: 输出累加器 (参照 mxfp8_golden.py 的 out, 归一化在最后做) ======
    out = np.zeros((qs, qd), dtype=np.float32)

    print(f"[CPU Golden] 8轮C1V1+1轮C2V2, TILES_Q={TILES_Q}, TILES_KV={TILES_KV}, TILES_V={TILES_V}")
    print(f"[CPU Golden] C1V1 基本块 S1={Q_BLOCK_SIZE} S2={K_BLOCK_SIZE}, "
          f"C2V2 大基本块 S1={Q_BLOCK_SIZE} S2={V_BLOCK_SIZE}")

    # ====== 修改: 外层循环按 Q block (S1=128) 步进 (参照 mxfp8: for i in range(TILES_Q)) ======
    for i in range(TILES_Q):
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, qs)
        Qi = q_tensor[0, 0, Sq_start:Sq_end, :].astype(np.float32)  # (S1=128, qd)
        print(f"Qi: {Qi}")
        # print(f"qi nz:")
        # for j in range(2):
        #     for i in range(8):
        #         print(f"[{i}][{j}]:")
        #         frac = Qi[i*16:(i+1)*16,j*64:(j+1)*64]
        #         # frac_res = []
        #         for m in range(16):
        #             frac_slice = []
        #             for n in range(8):
        #                 slice = frac[m, 8*n:8*(n+1)]
        #                 frac_slice.append("".join([hif4_values_dict.get(s) for s in slice]))
        #             # frac_res.append(frac_slice)
        #             print(frac_slice)

        # ====== 修改: 每个 Q block 独立的 online softmax 状态 (参照 mxfp8 o_BLOCKS/s_BLOCKS/m_BLOCKS) ======
        o_i = np.zeros((Sq_end - Sq_start, qd), dtype=np.float32)
        sum_i = np.zeros((Sq_end - Sq_start, 1), dtype=np.float32)
        # V1 阶段 max/softmax 使用 fp16 精度 (mi 为 running max, 保持 fp16)
        m_i = np.full((Sq_end - Sq_start, 1), -np.inf, dtype=np.float16)

        # ====== 修改: 中层循环按 V block (S2=1024) 步进, 每个大基本块 8轮C1V1+1轮C2V2 ======
        # (参照 mxfp8: for j in range(0, TILES_KV, 2) → 改为 for v_idx in range(TILES_V))
        for v_idx in range(TILES_V):
            v_start = v_idx * V_BLOCK_SIZE
            v_end = min(v_start + V_BLOCK_SIZE, S)

            # --- V 反量化 (HiFP4 → BF16), 大基本块 S2=1024 一次性反量化 ---
            # (参照 mxfp8: Vj_dequant = Vj * deq_scale_v_j)
            vi = v_tensor[:, :, v_start:v_end, :]  # (1, 1, 1024, vd)
            v_dequant_groups = (v_end - v_start) // G  # 1024/64 = 16 groups
            v_scale_start = v_start // G
            v_scale_end = v_scale_start + v_dequant_groups
            v_antiquant_i = v_antiquant_scale_cur[:, :, v_scale_start:v_scale_end, :]  # (1, 1, 16, D)
            vi_dequant = HiFP4_To_BF16_reverse_order(vi[0, 0].copy(), v_antiquant_i[0, 0].copy())
            vi_dequant = vi_dequant.astype(np.float32)  # (1024, vd)
            print(f"vi_dequant: {vi_dequant}")
            # ========== 8 轮 C1V1, 每轮基本块 S1=128 S2=128 (参照 mxfp8 2 轮 C1V1 扩展为 8 轮) ==========
            P_drops = []    # 存储 mxfp8 反量化后的 P, 每块 (S1=128, S2=128)
            m_blocks = []   # 存储每轮 m_block (running max)
            # sum 在 C2 阶段对大基本块 128*1024 统一计算, 不在每轮 C1V1 中计算

            mi = m_i.copy()  # 前一个大基本块的 running max (参照 mxfp8 mi)

            for k in range(NUM_C1V1):
                k_idx = v_idx * NUM_C1V1 + k
                if k_idx >= TILES_KV:
                    break

                # --- C1: QK matmul, 基本块 S1=128 S2=128 (保持原 hif4 C1 反量化逻辑) ---
                k_start = k_idx * K_BLOCK_SIZE
                k_end = min(k_start + K_BLOCK_SIZE, S)
                ki = k_tensor[:, :, k_start:k_end, :]  # (1, 1, 128, kd)



                # K 反量化 (HiFP4 → BF16, 保持原逻辑: 沿 D 轴分组, 转置后反量化再转回)
                k_antiquant_i = k_antiquant_scale_cur[:, :, k_start:k_end, :]  # (1, 1, 128, D/G)
                ki[0, 0] = (HiFP4_To_BF16_reverse_order(
                    ki[0, 0].copy().T, k_antiquant_i[0, 0].copy().T)).T
                print(f"ki: {ki}")
                # print(f"ki nz:")
                # for j in range(2):
                #     for i in range(8):
                #         print(f"[{i}][{j}]:")
                #         frac = ki[0,0,i*16:(i+1)*16,j*64:(j+1)*64]
                #         # frac_res = []
                #         for m in range(16):
                #             frac_slice = []
                #             for n in range(8):
                #                 slice = frac[m, 8*n:8*(n+1)]
                #                 frac_slice.append("".join([hif4_values_dict.get(s) for s in slice]))
                #             # frac_res.append(frac_slice)
                #             print(frac_slice)
                                
                # S = Q @ K^T, 基本块 (S1=128, S2=128) (参照 mxfp8 _compute_s_block)
                S_ik = np.matmul(Qi, ki[0, 0].astype(np.float32).T)  # (128, 128)

                print("s_ik nz before:")
                s_ik_transpose = S_ik.transpose()
                for j in range(8):
                    for i in range(8):
                        print(f"[{i}][{j}]:")
                        frac = s_ik_transpose[i*16:(i+1)*16,j*16:(j+1)*16]
                        # frac_res = []
                        for m in range(16):
                            print(f"[{' ,'.join([str(x) for x in frac[m]])}],")
                            # frac_res.append(frac_slice)


                S_ik = S_ik * np.float32(0.08837890625) # *根号d

                print(f"S_ik transpose fp32: {S_ik.transpose()}")

                print(f"++++++++++ C1V1[{k_idx}] S1={Sq_end-Sq_start} S2={k_end-k_start} matmul1 ++++++++++")

                # ====== 修改: V1 online softmax + P 量化为 mxfp8 (max/softmax 用 fp16 计算) ======
                # (参照 mxfp8 _online_softmax_update, 删除 mask 填 -inf 逻辑)
                # 求 block 内 max (fp16 精度)
                S_ik_fp16 = S_ik.astype(np.float16)
                # S_ik_fp16 = S_ik.copy()
                # for i in range(128):
                #     for j in range(128):
                #         S_ik_fp16[i][j] = hex4_to_fp16(fp32_to_fp16_manual(S_ik[i][j]))
                print(f"[{k}] S_ik transpose:")
                s_ik_transpose = S_ik_fp16.transpose()
                for i in range(128):
                    x = [str(float(m)) for m in s_ik_transpose[i]]
                    print(f"[{', '.join(x)}],")

                m_block_k = np.max(S_ik_fp16, axis=-1, keepdims=True).astype(np.float16) # (128, 1)
                print(f"m_block_k: {m_block_k.dtype}")
                # m 对齐到 ln2 整数倍 (ceil), 模拟 NPU MXFP8 精度 (fp16 计算) (参照 mxfp8: m_block_j = ceil(m_block_j / LN2) * LN2)
                m_block_k = (np.ceil(m_block_k * np.float16(1.4423828125)) * np.float16(0.693359375))
                print(f"m_block_k: {m_block_k}")
                # 与前一轮 running max 取 max (fp16) (参照 mxfp8: m_block_j = max(mi, m_block_j))
                m_block_k = np.maximum(mi, m_block_k)
                print("m_block_k fp16:")
                print(f"[{', '.join([str(float(x)) for x in m_block_k.reshape(1,128)[0]])}]")

                # ====== 修改: P = exp(S - m + ln_p_scale), 静态量化 (fp16 计算) (参照 mxfp8: P_ij_raw = exp(S - m + ln_p_scale)) ======
                # 注意: sum 不在此每轮计算, 留到 C2 阶段对大基本块 128*1024 统一求 sum
                # P_ik_raw = np.exp(S_ik_fp16 - m_block_k) # fp16的exp运算，输入相同，输出不同，导致误差。
                dif = S_ik_fp16 - m_block_k
                # P_ik_raw = np.exp(dif)
                P_ik_raw = np.zeros((128, 128), dtype=np.float16)
                for i in range(128):
                    for j in range(128):
                        P_ik_raw[i][j] = valu.expFloat16_polyHp(float(dif[i][j]),op_info,status)
                print(f"P_ik_raw transpose: {P_ik_raw.dtype}")

                P_ik_raw_transpose = P_ik_raw.transpose()
                for i in range(128):
                    print(f"[{', '.join([str(float(f)) for f in P_ik_raw_transpose[i]])}],")


                # ====== 修改: P 量化为 mxfp8 (FP8_E4M3) ======
                # 参照 mxfp8: P_ij_drop = P_ij_raw.to(FP8_DTYPE).to(torch.float32)
                P_ik_drop = torch.from_numpy(
                    np.ascontiguousarray(P_ik_raw)
                ).to(FP8_DTYPE).to(torch.float32).numpy()
                print(f"P_ik_drop: {P_ik_drop}")
                print(f"P_ik_drop transpose: {P_ik_drop.transpose()}")
                print(f"P_ik_drop nz:")
                for j in range(4):
                    for i in range(8):
                        # print(f"[{i}][{j}]:")
                        frac = P_ik_drop.transpose()[i*16:(i+1)*16,j*32:(j+1)*32]
                        # frac_res = []
                        for m in range(16):
                            # frac_slice = []
                            # for n in range(8):
                                # slice = frac[m, 8*n:8*(n+1)]
                                # frac_slice.append("".join([hif4_values_dict.get(s) for s in slice]))
                            # frac_res.append(frac_slice)
                            print(f"[{', '.join([str(f) for f in frac[m]])}],")
                P_drops.append(P_ik_drop)
                print(f"P_drops: {P_drops}")
                # C2V2 需要 fp32 精度, 存储时转回 fp32
                m_blocks.append(m_block_k.astype(np.float32))
                print(f"m_blocks: {m_blocks}")


                mi = m_block_k  # 更新 running max (fp16, 参照 mxfp8: mi 传递给下一轮)

                print(f"++++++++++ end C1V1[{k_idx}] ++++++++++\n")

            # ========== 1 轮 C2V2, 大基本块 S1=128 S2=1024 (参照 mxfp8 C2V2, 从 2 块扩展为 8 块) ==========
            # 8 个 C1V1 结果 (每块 S1=128 S2=128) 合并为大基本块 (S1=128 S2=1024) 计算 PV
            m_final = m_blocks[-1]  # 最终 running max (参照 mxfp8: m_block_j1 为最终 max)

            # ====== 修改: V 分成 8 段 (每段 S2=128), 每段对应一个 K block, PV matmul 合并 ======
            # 参照 mxfp8: P_ij_Vj = matmul(P_ij_drop * exp(m_block_j - m_block_j1), V_part1) + matmul(P_ij1_drop, V_part2)
            # 扩展为 8 块: P_ij_Vj = sum_k matmul(P_k * exp(m_k - m_final), V_part_k)
            # 大基本块 S1=128 S2=1024 = 8 * (S1=128 S2=128) PV matmul 累加
            CANDIDATES = [2.0 ** (-n) for n in range(0, 10)]
            def nearest_pow2_inv(x: float) -> tuple[float, int]:
                """
                在 {2^0, 2^-1 ... 2^-10} 中找最靠近x的数
                return (closest_value, N)
                """
                best = None
                best_n = 0
                min_diff = float("inf")
                for n, val in enumerate(CANDIDATES):
                    diff = abs(x - val)
                    if diff < min_diff:
                        min_diff = diff
                        best = val
                        best_n = n
                return best

            P_ij_Vj = np.zeros((Sq_end - Sq_start, vd), dtype=np.float32)

            for k in range(len(P_drops)):
                v_part_start = k * K_BLOCK_SIZE
                v_part_end = v_part_start + K_BLOCK_SIZE
                V_part = vi_dequant[v_part_start:v_part_end, :]  # (128, vd)

                # P_k rescale 到最终 max (参照 mxfp8: P_ij_drop * exp(m_block_j - m_block_j1))
                rescale_k = np.exp(m_blocks[k] - m_final).astype(np.float32)
                for i in range(128):
                    rescale_k[i][0] = nearest_pow2_inv(rescale_k[i][0])
                print("pscale: ")
                print(f"[{', '.join([str(f) for f in rescale_k.reshape(1, 128)[0]])}]")

                P_ij_Vj += np.matmul(P_drops[k] * rescale_k, V_part)

                if k % 2 == 1:
                    print(f"oi nz:")
                    for j in range(8):
                        for i in range(8):
                            print(f"[{i}][{j}]:")
                            frac = P_ij_Vj[i*16:(i+1)*16,j*16:(j+1)*16]
                            # frac_res = []
                            for m in range(16):
                                print(f"[{', '.join([str(float(f)) for f in frac[m]])}],")


            # ====== 修改: 更新 sum (参照 mxfp8: si_new = exp(mi-m_final)*si + sum_k(s_k * exp(m_k - m_final))) ======
            # 注意: sum 在 C2 阶段对大基本块 128*1024 统一计算
            # 将 8 个 P 块 rescale 到最终 max 后拼接为大基本块 (S1=128, S2=1024), 再整体求 sum
            rescale_old = np.exp(m_i - m_final)  # 参照 mxfp8: update_mul_si = exp(mi - m_block_j1)
            print(f"rescale_old: {rescale_old}")
            sum_i = rescale_old * sum_i
            P_big_list = []
            for k in range(len(P_drops)):
                rescale_k = np.exp(m_blocks[k] - m_final)
                P_big_list.append(P_drops[k] * rescale_k)
            P_big = np.concatenate(P_big_list, axis=-1)  # (S1=128, S2=1024)
            sum_i += np.sum(P_big, axis=-1, keepdims=True)  # 大基本块 128*1024 的 sum
            print(f"sum_i: {sum_i}")

            # ====== 修改: 更新 o (参照 mxfp8: o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj) ======
            o_i = rescale_old * o_i + P_ij_Vj

            m_i = m_final.copy()

        # ========== V2: 每个 Q block 归一化 (参照 mxfp8: out = out / (out_sum + EPSILON)) ==========
        out[Sq_start:Sq_end, :] = o_i / (sum_i + EPSILON)

    out = out.astype(np_bfloat16)

    print("****************************** end _np_pfaattention_act_fp4 ******************************\n")
    return out, None


def broadcast_kv_np(num_heads, num_kv_heads, kv_tensor):
    """GQA 广播: 将 K/V 沿 N 维从 N_kv 广播到 num_heads
    入参 kv_tensor: (B, N_kv, S, D_half) uint8 (packed fp4)
    返回: (B, num_heads, S, D_half)
    """
    factor = num_heads // num_kv_heads
    B, _, S, D_half = kv_tensor.shape
    result = np.zeros((B, num_heads, S, D_half), dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i:i + 1, :, :] = kv_tensor[:, i // factor:i // factor + 1, :, :]
    return result


def broadcast_kv_scale_5d(num_heads, num_kv_heads, scale_5d):
    """GQA 广播 5D scale: (B, 1, N_kv, ...) → (B, 1, num_heads, ...)
    沿 N 维 (axis=2) 广播
    """
    factor = num_heads // num_kv_heads
    _, B, N_kv, *rest = scale_5d.shape
    result = np.zeros((1, B, num_heads, *rest), dtype=scale_5d.dtype)
    for i in range(num_heads):
        result[:, :, i:i + 1, ...] = scale_5d[:, :, i // factor:i // factor + 1, ...]
    return result


def cpu_hifp4_golden(q_fp4, k_fp4, v_fp4,
                     dequant_scale_q, k_antiquant_scale, v_antiquant_scale,
                     actual_seq_q, actual_seq_kv):
    """CPU golden 入口: 适配 6 元组输入
    输入 shape (与 NPU 接口一致, BNSD):
      q_fp4: (B, N_q, Sq, D/2) uint8
      k_fp4: (B, N_kv, Skv, D/2) uint8
      v_fp4: (B, N_kv, Skv, D/2) uint8
      dequant_scale_q: (1, B, N_q, Sq, D/G) uint32 (5D)
      k_antiquant_scale: (1, B, N_kv, Skv, D/G) uint32
      v_antiquant_scale: (1, B, N_kv, D, Skv/G) uint32
    返回: (out bfloat16 tensor, lse=None)

    若 INPUT_LAYOUT == "BSH", 输入 6 元组按 BSH 排布 (Q/K/V 为 (B,S,H/2), scale 5D 保持
    N 维语义); 入口处参考np_bsh_to_bnsd 将 Q/K/V 转回 BNSD 进行计算, 出口处参考 trans_bnsd_to_bsh 将输出
    reshape 回 BSH 排布。CPU golden 内部始终以 BNSD 进行计算。
    """
    pfa_param = {'sparse': SPARSE_MODE, 'lseflag': False}

    # BSH 入口转换
    # 注意: np_bsh_to_bnsd 期望 bsh_shape 为"逻辑 shape"(H 维未 packed, H=N*D),
    # 而 q_fp4 为物理 packed 数据 (last dim = H/2 = N*D/2)
    if INPUT_LAYOUT == "BSH":
        logger.info("[INFO] CPU golden: BSH → BNSD 转换")
        q_shape_bsh = [q_fp4.shape[0], q_fp4.shape[1], N_q * D]    # 逻辑 (B,S,H_q)
        k_shape_bsh = [k_fp4.shape[0], k_fp4.shape[1], N_kv * D]   # 逻辑 (B,S,H_kv)
        v_shape_bsh = [v_fp4.shape[0], v_fp4.shape[1], N_kv * D]
        q_fp4, q_bnsd_shape = np_bsh_to_bnsd(q_fp4, q_shape_bsh, N_q, actual_seq_q, INPUT_LAYOUT)
        k_fp4, k_bnsd_shape = np_bsh_to_bnsd(k_fp4, k_shape_bsh, N_kv, actual_seq_kv, INPUT_LAYOUT)
        v_fp4, v_bnsd_shape = np_bsh_to_bnsd(v_fp4, v_shape_bsh, N_kv, actual_seq_kv, INPUT_LAYOUT)

    B_, N_q_, Sq, _ = q_fp4.shape

    # GQA: 若 N_q != N_kv, 需将 K/V 及其 scale 沿 N 维广播到 N_q
    if N_q_ != N_kv:
        logger.info("[INFO] GQA 广播: N_kv=%d → N_q=%d", N_kv, N_q_)
        k_fp4 = broadcast_kv_np(N_q_, N_kv, k_fp4)
        v_fp4 = broadcast_kv_np(N_q_, N_kv, v_fp4)
        k_antiquant_scale = broadcast_kv_scale_5d(N_q_, N_kv, k_antiquant_scale)
        v_antiquant_scale = broadcast_kv_scale_5d(N_q_, N_kv, v_antiquant_scale)

    out = np.zeros((B_, N_q_, Sq, D), dtype=np_bfloat16)

    # _np_pfaattention_act_fp4_new 原函数假设 B=1, N=1, 直接用 [0,0] 索引
    # 这里逐 (b, n) 调用, 每次切出 (1,1,...) 子张量
    for b in range(B_):
        for n in range(N_q_):
            q_cur = q_fp4[b:b+1, n:n+1, :, :]              # (1,1,Sq,D/2)
            k_cur = k_fp4[b:b+1, n:n+1, :, :]              # (1,1,Skv,D/2)
            v_cur = v_fp4[b:b+1, n:n+1, :, :]              # (1,1,Skv,D/2)
            dq_q = dequant_scale_q[0, b:b+1, n:n+1, :, :]  # (1,1,Sq,D/G)
            dq_k = k_antiquant_scale[0, b:b+1, n:n+1, :, :]
            dq_v = v_antiquant_scale[0, b:b+1, n:n+1, :, :]

            o, _ = _np_pfaattention_act_fp4_new(
                q_cur, k_cur, v_cur,
                None, None, SOFTMAX_SCALE,
                actual_seq_q, actual_seq_kv,
                0, 0, pfa_param,
                dq_q, dq_k, dq_v,
                None, None, None, None, None,
                out_dtype=str(np.int8))
            out[b, n, :, :] = o[:, :]

    # BSH 出口转换
    #   y_all = y_all.transpose(0, 2, 1, 3).reshape(B, S, H))
    if INPUT_LAYOUT == "BSH":
        out = trans_bnsd_to_bsh(out, [B_, N_q_, Sq, D])

    # numpy bfloat16 (ml_dtypes) 不能直接 torch.from_numpy
    # 先 view 成 uint16, 再 view 成 torch.bfloat16 
    out_uint16 = np.ascontiguousarray(out).view(np.uint16)
    return torch.from_numpy(out_uint16).view(torch.bfloat16), None

# ==============================================================================
# NPU 调用 (HiFP4)
# 结构参考 quant_flash_attn_golden_mxfp8.py 的 mxfp8 NPU 链, 语义改为 hifp4:
#   q/k/v: HiFP4 packed uint8 (hifloat4_x2), scale: uint32 (view as fp32)
#   quant_mode=11, 无 p_scale/rope/PA, 直接调 quant_flash_attn_metadata + quant_flash_attn
# ==============================================================================


def _call_npu_fa_op(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    block_size,
    sparse_mode,
    out_dtype,
):
    """调用 NPU 双算子 (QFA: quant_flash_attn_metadata + quant_flash_attn) - HiFP4"""
    if not _HAS_NPU:
        raise ImportError(
            "npu_ops_transformer.ops.quant_flash_attn is not available. "
            "Please check that npu_ops_transformer is installed and all .so are compiled."
        )

    cu_seqlens_q_t = (
        torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
        if cu_seqlens_q is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(cu_seqlens_kv, dtype=torch.int32).npu()
        if cu_seqlens_kv is not None
        else None
    )
    seqused_q_t = (
        torch.tensor(seqused_q, dtype=torch.int32).npu()
        if seqused_q is not None
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=torch.int32).npu()
        if seqused_kv is not None
        else None
    )

    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"

    torch.npu.synchronize()

    # metadata = quant_flash_attn_metadata(
    #     num_heads_q=q_n,
    #     num_heads_kv=kv_n,
    #     head_dim=q.shape[-1] * 2,
    #     quant_mode=11,
    #     cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
    #     cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
    #     seqused_q=seqused_q_t,
    #     seqused_kv=seqused_kv_t,
    #     v_descale=v_descale,
    #     mask_mode=sparse_mode,
    #     layout_q=layout_q,
    #     layout_q_descale=layout_q_descale,
    #     layout_kv=layout_kv,
    #     layout_out=layout_out,
    #     max_seqlen_q=max_seqlen_q,
    #     max_seqlen_kv=max_seqlen_kv,
    # )
    # print(f"metadataliao:{metadata.device}",flush=True)

    metadata = torch.load("/home/l00911739/pytest_121/hif4-1-1-128-2048-metadata_new.pt", map_location="npu:0")

    main_kwargs = dict(
        q=q,
        k=k,
        v=v,
        q_descale=dequant_scale_q,
        k_descale=dequant_scale_k,
        v_descale=v_descale,
        quant_mode=11,
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
        cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        attn_mask=mask,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=sparse_mode,
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )
    main_kwargs.update(_get_npu_fa_kwargs())
    atten_out, lse_out = quant_flash_attn(**main_kwargs)
    torch.npu.synchronize()
    return atten_out, lse_out


class Network(nn.Module):
    """aclgraph 编译目标: forward 只包含两个 torch.library op 调用 (HiFP4)。

    输入已预处理:
      - cu_seqlens_q/kv, seqused_q/kv: 由 hifp4_fa_torch_npu 转好 NPU tensor
      - q/k/v/deq_*: 已转好 layout 的 NPU tensor (HiFP4 packed uint8 / uint32 scale)
      - max_seqlen_q/kv: 已计算好的标量
    """

    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        q_rope,
        k_rope,
        mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        dequant_scale_q,
        dequant_scale_k,
        v_descale,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        block_size,
        sparse_mode,
        out_dtype,
        max_seqlen_q,
        max_seqlen_kv,
    ):
        metadata = quant_flash_attn_metadata(
            num_heads_q=q_n,
            num_heads_kv=kv_n,
            head_dim=q.shape[-1] * 2,
            quant_mode=11,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            v_descale=v_descale,
            mask_mode=sparse_mode,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        print(f"metadataliao1:{metadata.cpu()}")
        main_kwargs = dict(
            q=q,
            k=k,
            v=v,
            q_descale=dequant_scale_q,
            k_descale=dequant_scale_k,
            v_descale=v_descale,
            quant_mode=11,
            block_table=block_table,
            p_scale=p_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            mask_mode=sparse_mode,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        main_kwargs.update(_get_npu_fa_kwargs())
        atten_out, lse_out = quant_flash_attn(**main_kwargs)
        return atten_out, lse_out


def prepare_npu_inputs(
    q_fp4,
    k_fp4,
    v_fp4,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """准备 NPU 侧入参 (HiFP4): q/k/v 为 packed uint8, scale 为 uint32(view as fp32)。
    hif4 仅支持 BNSD layout, 直接透传 (不做校验/格式转换), 无 PA/rope。
    返回字典 key 与 _call_npu_fa_op 形参一一对应。
    """
    torch_npu.npu.set_device(int(DEVICE_ID))

    softmax_scale = (
        SOFTMAX_SCALE
        if SOFTMAX_SCALE is not None
        else 1.0 / math.sqrt(D + (D_rope if ENABLE_ROPE else 0))
    )

    def to_npu_uint8(np_arr):
        return torch.from_numpy(np.ascontiguousarray(np_arr.astype(np.uint8))).npu()

    def to_npu_scale(np_arr):
        return torch.from_numpy(
            np.ascontiguousarray(np_arr.view(np.float32).astype(np.float32))
        ).npu()

    # HiFP4: q/k/v 直接透传 packed uint8 (hifloat4_x2), BNSD layout
    q_npu = to_npu_uint8(q_fp4)
    k_npu = to_npu_uint8(k_fp4)
    v_npu = to_npu_uint8(v_fp4)
    logger.info("[NPU BNSD] q=%s, k=%s, v=%s", q_npu.shape, k_npu.shape, v_npu.shape)

    # HiFP4 scale (含 qscale): uint32, view as fp32 透传 (不做 e8m0 转换)
    deq_q_npu = to_npu_scale(dequant_scale_q)
    deq_k_npu = to_npu_scale(dequant_scale_k)
    deq_v_npu = to_npu_scale(v_descale)
    logger.info(
        "[NPU BNSD] deq_q=%s, deq_k=%s, deq_v=%s",
        deq_q_npu.shape, deq_k_npu.shape, deq_v_npu.shape,
    )

    p_scale_npu = p_scale.npu() if p_scale is not None else None

    out_dtype = torch.bfloat16 if ENABLE_ROPE else torch.float16

    logger.info("[NPU] prepare inputs done.")
    return dict(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        q_rope=None,
        k_rope=None,
        mask=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=deq_k_npu,
        v_descale=deq_v_npu,
        p_scale=p_scale_npu,
        block_table=None,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        layout_q="BNSD",
        layout_q_descale="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
        block_size=0,
        sparse_mode=SPARSE_MODE,
        out_dtype=out_dtype,
    )


def npu_hifp4_fa(
    q_fp4,
    k_fp4,
    v_fp4,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """调用 NPU 算子 (HiFP4, quant_mode=11), 支持 PA / 非 PA 场景。

    正常用例入口：准备入参 → 调用算子 → 输出截断/LSE 处理。
    """
    inputs = prepare_npu_inputs(
        q_fp4,
        k_fp4,
        v_fp4,
        dequant_scale_q,
        dequant_scale_k,
        v_descale,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q,
        max_seqlen_kv,
        block_table_torch=block_table_torch,
        qr_bf16=qr_bf16,
        kr_bf16=kr_bf16,
    )

    logger.info(
        "[NPU] 调用 %s 模式 (GRAPH_PATH=%d)...",
        "PA" if ENABLE_PA else inputs["layout_q"],
        GRAPH_PATH,
    )
    # 通过 hifp4_fa_torch_npu 路由，支持 GRAPH_PATH=7 (aclgraph)
    
    # print(f"seqused_qliao:{seqused_q}",flush=True)
    # print(f"seqused_kvliao:{seqused_kv}",flush=True)
    # print(f"v_descaleliao:{v_descale}",flush=True)
    atten_out, lse_out = hifp4_fa_torch_npu(**inputs)


    act_seqused_q = _get_seqused_q()
    npu_output = atten_out
    T_actual = cu_seqlens_q[-1] if cu_seqlens_q is not None else sum(act_seqused_q)
    if npu_output.shape[0] > T_actual:
        npu_output = npu_output[:T_actual]
    logger.info("[NPU] output=%s", npu_output.shape)
    return npu_output, lse_out


def hifp4_fa_torch_npu(
    q,
    k,
    v,
    q_rope,
    k_rope,
    mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    block_size,
    sparse_mode,
    out_dtype,
):
    """NPU 调用入口 (HiFP4), 支持 GRAPH_PATH=0 (单算子) 和 GRAPH_PATH=7 (aclgraph)"""
    if GRAPH_PATH == 0:
        logger.info("[NPU] 调用 QFA 单算子模式 (HiFP4)...")
        return _call_npu_fa_op(
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout_q,
            layout_q_descale,
            layout_kv,
            layout_out,
            block_size,
            sparse_mode,
            out_dtype,
        )

    # GRAPH_PATH == 7: aclgraph
    cu_seqlens_q_t = (
        torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
        if cu_seqlens_q is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(cu_seqlens_kv, dtype=torch.int32).npu()
        if cu_seqlens_kv is not None
        else None
    )
    seqused_q_t = (
        torch.tensor(seqused_q, dtype=torch.int32).npu()
        if seqused_q is not None
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=torch.int32).npu()
        if seqused_kv is not None
        else None
    )

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    with torch.no_grad():
        torch.npu.synchronize()

        fa_args = (
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout_q,
            layout_q_descale,
            layout_kv,
            layout_out,
            block_size,
            sparse_mode,
            out_dtype,
            max_seqlen_q,
            max_seqlen_kv,
        )

        logger.info("[NPU] 调用 aclgraph (npugraph_ex)...")
        npu_backend = "npugraph_ex"
        npu_mode = torch.compile(
            npu_mode, fullgraph=False, backend=npu_backend, dynamic=False
        )
        for t in (
            q,
            k,
            v,
            q_rope,
            k_rope,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_k,
            v_descale,
            p_scale,
            block_table,
        ):
            if t is not None:
                torch._dynamo.mark_static(t)
        atten_out, lse_out = npu_mode(*fa_args)

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out



# ==============================================================================
# Main / Pipeline
# ==============================================================================

if __name__ == "__main__":
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="HiFP4 Flash Attention Golden")
    parser.add_argument(
        "--mode",
        default="all",
        help="执行模式，支持逗号组合: all/gen/cpu/npu/compare. 例: --mode=npu,compare",
    )
    parser.add_argument(
        "--case-name", default="default", help="case 名称，用于 .pt 文件命名"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="缓存目录路径（默认 golden_cache/）"
    )
    args = parser.parse_args()

    raw_parts = {m.strip() for m in args.mode.split(",") if m.strip()}
    invalid = raw_parts - _VALID_MODES
    if invalid:
        parser.error(f"Invalid mode: {invalid}. Valid: {_VALID_MODES}")
    mode = {"gen", "cpu", "npu", "compare"} if "all" in raw_parts else raw_parts

    case_name = args.case_name
    cdir = args.cache_dir

    logger.info("=" * 60)
    logger.info("HiFP4 Flash Attention Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("=" * 60)
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d, G=%d", B, N_q, N_kv, D, HIFP4_GROUP_SIZE)
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
    logger.info("INPUT_LAYOUT=%s, SPARSE_MODE=%d", INPUT_LAYOUT, SPARSE_MODE)
    logger.info("QUERY_QUANT_MODE=%d, KEY_QUANT_MODE=%d, VALUE_QUANT_MODE=%d",
                QUERY_QUANT_MODE, KEY_QUANT_MODE, VALUE_QUANT_MODE)

    if "gen" in mode:
        logger.info("\n[Step 1] 数据生成")
        data = generate_data()
        golden_cache.save_input(case_name, golden_cache.build_input_dict(*data), cache_dir=cdir)
    else:
        logger.info("\n[Step 1] 加载已保存的输入数据")
        data = golden_cache.load_input(case_name, cache_dir=cdir)
    q_fp4, k_fp4, v_fp4, dequant_scale_q, k_antiquant_scale, v_antiquant_scale = data

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("\n[Done] 数据已保存，退出")
        exit(0)

    if "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        cpu_out, cpu_lse = cpu_hifp4_golden(
            q_fp4, k_fp4, v_fp4,
            dequant_scale_q, k_antiquant_scale, v_antiquant_scale,
            ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("\n[Done] CPU 输出已保存，退出")
        exit(0)

    if "npu" in mode:
        logger.info("\n[Step 3] NPU 调用")
        atten_out, lse_out = npu_hifp4_fa(
            q_fp4, k_fp4, v_fp4,
            dequant_scale_q, k_antiquant_scale, v_antiquant_scale,
            None,
            CU_SEQLENS_Q, CU_SEQLENS_KV,
            SEQUSED_Q, SEQUSED_KV,
            MAX_SEQLEN_Q, MAX_SEQLEN_KV,
            None, None, None)
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        logger.info("\n[Done] NPU 输出已保存，退出")
        exit(0)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    result_compare_method.check_result(cpu_out, atten_out)
