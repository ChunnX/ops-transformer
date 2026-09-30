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
支持：PA / 非PA 场景，GQA
量化：Q/K per-token-group (quant_mode=6), V per-channel-group (quant_mode=8)
输出：逐元素表格 + 统计汇总 (PctRlt 通过率，双千分之五标准)

"""

import logging
import math
from typing import Optional
import torch
import torch_npu
import os as _os

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区
# ==============================================================================
E8M0_MIN_POSITIVE = 2 ** (-127)
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 32

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
    dim4_align = (dim4 + 63) // 64 * 64
    num_groups = math.ceil(dim4_align / group_size)
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


def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    factor = num_heads // num_kv_heads
    B, _, S, D = kv_tensor.shape
    result = torch.zeros([B, num_heads, S, D], dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i : i + 1, :, :] = kv_tensor[:, i // factor : i // factor + 1, :, :]
    return result


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


def e8m0_to_fp32(tensor_e8m0):
    biased_exp = tensor_e8m0.view(torch.uint8).to(torch.float32)
    result = torch.pow(2.0, biased_exp - 127)
    # biased_exp == 0xFF is the NaN sentinel in e8m0fnu — force to zero
    nan_mask = biased_exp == 0xFF
    if nan_mask.any():
        result = result.clone()
        result[nan_mask] = 0.0
    return result


def canonical_q_scale_layout(layout):
    layout = (layout or "TND").upper()
    if layout not in ("TND", "N2TGD"):
        raise ValueError(f"Unsupported Q scale layout: {layout}")
    return layout


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


# ==============================================================================
# TND → BNSD 反向转换 + scale unpack 辅助函数
# ==============================================================================


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
    kv_layout="BnNBsD",
    group_size=32,
    *,
    total_blocks: Optional[int] = None,
):
    """
    MXFP8 PA 预处理: BNSD → PagedAttention KV Cache

    输入: [B, N, S, D]
    输出 (is_scale=False): [BlockNum, N, BlockSize, D] (fp8 K/V)
    输出 (is_scale=True, is_vscale=False): [BlockNum, N, BlockSize, D//64, 2] (K scale)
    输出 (is_scale=True, is_vscale=True): [BlockNum, N, BlockSize//64, D, 2] (V scale)

    kv_layout:
      - BnNBsD: fp8=[Bn,N,Bs,D], Kscale=[Bn,N,Bs,D//64,2], Vscale=[Bn,N,Bs//64,D,2]
      - PA_BNBD: 同 BnNBsD, [Bn,N,Bs,D]
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

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_scale_pack_ratio = group_size * 2
            pack_seq_lens = [
                math.ceil(act_s / v_scale_pack_ratio) for act_s in seq_lens
            ]
            pack_block_size = math.ceil(block_size / v_scale_pack_ratio)
            total_block_num = (
                total_blocks
                if total_blocks is not None
                else sum(math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens)
            )
            out_shape = (total_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K scale D must be even for packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (
                total_blocks
                if total_blocks is not None
                else sum(math.ceil(act_s / block_size) for act_s in seq_lens),
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
            total_blocks
            if total_blocks is not None
            else sum(math.ceil(act_s / block_size) for act_s in seq_lens),
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
    if kv_layout in ("BnNBsD", "PA_BNBD"):
        # PA_BNBD: [Bn, N, Bs, D] — 和 BnNBsD 维度排列一致
        return out_cache
    elif kv_layout in ("BnBsND", "PA_BBND"):
        return out_cache.transpose(1, 2).contiguous()
    elif kv_layout == "PA_NZ":
        Bn, KV_N, Bs, KV_D = out_cache.shape[:4]
        if not is_scale:
            inner = 32
            if KV_D % inner != 0:
                raise ValueError(f"PA_NZ D must be divisible by {inner}, got {KV_D}")
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


# ==============================================================================
# PA 格式逆转换 - PA → BNSD reverse
# ==============================================================================


def pa_reverse_permute_nz(tensor, is_scale, is_vscale):
    tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    if not is_scale:
        # fp8 data: [Bn,N,D//32,Bs,32] → permute(0,1,3,2,4) → [Bn,N,Bs,D//32,32]
        # then reshape → [Bn,N,Bs,D]
        t = tensor.permute(0, 1, 3, 2, 4).contiguous()
        Bn, N, Bs, inner, tail = t.shape
        D = inner * tail
        return t.reshape(Bn, N, Bs, D)
    elif not is_vscale:
        # K scale: [Bn,N,Bs//16,D//2,16,2] → permute(0,1,2,4,3,5) → [Bn,N,Bs//16,16,D//2,2]
        # then reshape → [Bn,N,Bs,D//2,2]
        t = tensor.permute(0, 1, 2, 4, 3, 5).contiguous()
        Bn, N, Bs_div_16, inner, D_half, _pair = t.shape
        Bs = Bs_div_16 * inner
        return t.reshape(Bn, N, Bs, D_half, 2)
    else:
        # V scale: [Bn,N,D//16,Bs,16,2] → permute(0,1,3,2,4,5) → [Bn,N,Bs,D//16,16,2]
        # then reshape → [Bn,N,Bs,D,2]
        t = tensor.permute(0, 1, 3, 2, 4, 5).contiguous()
        Bn, N, Bs, D_div_16, inner, tail2 = t.shape
        D = D_div_16 * inner
        return t.reshape(Bn, N, Bs, D, 2)


def pa_to_bnsd_data(pa_tensor, seq_lens, block_size, block_table, kv_layout="BnNBsD"):
    pa_tensor = (
        pa_tensor if isinstance(pa_tensor, torch.Tensor) else torch.as_tensor(pa_tensor)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    Bn, N, Bs, D = pa_tensor.shape
    B = block_table.shape[0]
    max_skv = max(seq_lens)

    result = torch.zeros(
        B,
        N,
        max_skv,
        D,
        dtype=pa_tensor.dtype,
        device=pa_tensor.device,
    )
    num_blocks = [math.ceil(s / block_size) for s in seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(num_blocks[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            result[b, :, block_offset : block_offset + valid_len] = pa_tensor[
                blockid, :, :valid_len
            ]
    return result[:, :, : max(seq_lens), :].contiguous()


# ==============================================================================
# 数据生成
# ==============================================================================


# ==============================================================================
# CPU Golden
# Flash Attention tiling 策略: C1V1C1V1C2V2 流水
#   Q 按 Q_BLOCK_SIZE 分块, K/V 按 K_BLOCK_SIZE/V_BLOCK_SIZE 分块
#   每次迭代处理 2 个 K block (j, j+1) 和 1 个 V block
#   Online softmax: 维护 running max (m) 和 running sum (s) 实现数值稳定
#   TND layout 下 m 需要对齐到 ln2 的整数倍 (ceil)
# ==============================================================================


def _build_attention_mask(b, Sq, Skv, actual_seq_q, actual_seq_kv, sparse_mode):
    """构建全局 attention mask
    sparse_mode=3: causal + padding mask (左下三角 + 右上 padding)
    其他: 仅 padding mask
    """
    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32)
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32)
    q_lens_acl = q_lens_t.view(b, 1, 1, 1)
    k_lens_acl = k_lens_t.view(b, 1, 1, 1)

    q_range = torch.arange(Sq).view(1, 1, -1, 1)
    k_range = torch.arange(Skv).view(1, 1, 1, -1)
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if sparse_mode == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        return causal_mask | q_padding_mask | k_padding_mask
    else:
        return q_padding_mask | k_padding_mask


def _compute_s_block(Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale):
    # Apply the MX scales after each raw 32-element dot product.
    score = None
    for c0 in range(0, Qi.shape[-1], QUANT_GROUP_SIZE):
        dot = torch.matmul(
            Qi[..., c0 : c0 + QUANT_GROUP_SIZE],
            Kj[..., c0 : c0 + QUANT_GROUP_SIZE].transpose(-1, -2),
        )
        scale = deq_scale_q_i[..., c0, None] * deq_scale_k_j[..., c0].unsqueeze(-2)
        term = dot * scale
        score = term if score is None else score + term
    return score * softmax_scale


def _online_softmax_update(S_ij, mask_j, mi, si, oi, ln_p_scale, tail_mask=None):
    """Keep the rounded, p_scale-adjusted maximum used by the kernel."""
    ln2 = 0.6931471824645996
    inv_ln2 = 1.4426950408889634
    S_ij = S_ij.masked_fill(mask_j, float("-inf"))
    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
    m_block_j = torch.ceil(m_block_j * inv_ln2) * ln2 - ln_p_scale
    m_block_j = torch.max(mi, m_block_j)
    P_ij_raw = torch.exp(S_ij - m_block_j)
    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
    P_ij_drop = P_ij_raw.to(FP8_DTYPE).to(torch.float32)
    if tail_mask is not None:
        P_ij_drop = torch.where(
            tail_mask, P_ij_raw.to(torch.bfloat16).to(torch.float32), P_ij_drop
        )
    return m_block_j, s_block_j, P_ij_drop


def _p_dot_v_scaled(P_eff, V_raw, dv_exp_block):
    """O = P @ V 的 mm2 L0C 语义模拟: 先 fp8 原值 dot, 再按 32 行 group 乘 dv scale。

    - dot 阶段 P/V 均为有限值 (P≤448, V fp8≤448; 尾段 V 为 bf16 注入原值),
      0×有限=0 — 不会产生 0×±inf=NaN 的中毒;
    - scale 阶段: group 内 32 行共享 dv (per-channel), dot×dv 溢出→单符号 ±inf;
      0(dot)×scale(有限)=0 — 与硬件 L0C 反量化在累加之后的行为一致。
    P_eff: (b,n,sq,blk) fp32 (rescale 因子已折叠, ≤p_scale);
    V_raw: (b,n,blk,D) fp32 V 原值 (未反量化);
    dv_exp_block: (b,n,blk,D) 逐行展开的 dv scale (尾段=1)。
    """
    blk = V_raw.shape[2]
    O = None
    for c0 in range(0, blk, 32):
        dot = torch.matmul(P_eff[..., c0 : c0 + 32], V_raw[..., c0 : c0 + 32, :])
        sc = dv_exp_block[:, :, c0, :][:, :, None, :]
        term = dot * sc
        O = term if O is None else O + term
    return O


def cpu_mxfp8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
    v_tail_bnsd=None,
    tail_lens=None,
):
    """CPU Flash Attention golden with MXFP8, C1V1C1V1C2V2 流水

    v_tail_bnsd: (B, N_kv, tail_capacity, D) bf16 尾块原值
    (TTK: 由 golden 插件从 v_tail slot 反布局得到), tail_lens: 每 batch 尾长
    (0<tail<64, kv%64)。注入语义与 pytest 侧一致: 尾段 V 用 bf16 原值替换反量化值
    (尾段 scale 恒 1), 尾列 P 直存 bf16 不走 e4m3 量化往返。
    """
    EPSILON = 1e-20
    Q_BLOCK_SIZE = 128
    if D == 256:
        K_BLOCK_SIZE = 128
        V_BLOCK_SIZE = 256
    else:
        K_BLOCK_SIZE = 256
        V_BLOCK_SIZE = 512

    # FP8 → FP32 反量化
    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    # GQA: 广播 K/V 到与 Q 相同的 head 数
    if N_q != N_kv:
        logger.info("[INFO] GQA 广播")
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        dequant_scale_k = broadcast_kv(N_q, N_kv, dequant_scale_k)
        dequant_scale_v = broadcast_kv(N_q, N_kv, dequant_scale_v)

    b, n, s, d = q_tensor.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    dv = v_tensor.shape[-1]
    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]

    minValue = torch.tensor(-3.402823466e38, dtype=torch.float32)
    out = torch.zeros([b, n, Sq, dv], dtype=torch.float32)
    o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]
    o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[
        ..., None
    ]

    TILES_Q = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    TILES_KV = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    mask_global = _build_attention_mask(
        b, Sq, Skv, actual_seq_q, actual_seq_kv, SPARSE_MODE
    )

    Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
    V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
    o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    # dequant_scale 按 group_size 扩展，用于逐元素反量化
    dequant_scale_q_expanded = dequant_scale_q.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    dequant_scale_k_expanded = dequant_scale_k.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[:, :, :, :D]
    dequant_scale_v_expanded = dequant_scale_v.repeat_interleave(
        QUANT_GROUP_SIZE, dim=2
    )[:, :, :, :D]

    # 尾块高精窗口: CPU参考 = 主体V反量化 + 尾段V的bf16原值(模拟完美尾块cache, 尾段scale恒1)
    # 注: V_BLOCKS 是 v_tensor 的 split 视图(共享存储), in-place 注入对视图可见
    vtail_tail_start = None
    if (
        v_tail_bnsd is not None
        and tail_lens is not None
        and any(t > 0 for t in tail_lens)
    ):
        for b_i in range(v_tensor.shape[0]):
            kv_len = actual_seq_kv[b_i]
            tail_len = tail_lens[b_i]
            main_len = kv_len - tail_len
            if tail_len > 0 and main_len < v_tensor.shape[2]:
                # op的v_tail输入为bf16(接口契约), CPU参考必须注入同精度值:
                # 注入fp32原值会比kernel多出V精度 — 与kernel的bf16直存窗口失配
                tail_seg = (
                    v_tail_bnsd[b_i, :, :tail_len, :]
                    .to(torch.bfloat16)
                    .to(torch.float32)
                )
                # GQA: 尾块原值按KV head组织, 需与已广播的v_tensor头数对齐(同broadcast_kv语义)
                if tail_seg.shape[0] != v_tensor.shape[1]:
                    factor = v_tensor.shape[1] // tail_seg.shape[0]
                    tail_seg = tail_seg.repeat_interleave(factor, dim=0)
                v_tensor[b_i, :, main_len:kv_len, :] = tail_seg
                dequant_scale_v_expanded[b_i, :, main_len:kv_len, :] = 1.0
        # 尾窗口起点(per batch): 尾列P直存bf16的判定边界, 与V注入窗口同语义
        vtail_tail_start = torch.tensor(
            [actual_seq_kv[b_i] - tail_lens[b_i] for b_i in range(v_tensor.shape[0])]
        )
        logger.info("[V_TAIL] CPU参考已注入尾段bf16原值, tail_lens=%s", tail_lens)

    logger.info(
        "[CPU Golden] TILES_Q=%d, TILES_KV=%d, Sq=%d, Skv=%d",
        TILES_Q,
        TILES_KV,
        Sq,
        Skv,
    )

    for i in range(TILES_Q):
        Qi = Q_BLOCKS[i]
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
        deq_scale_q_i = dequant_scale_q_expanded[:, :, Sq_start:Sq_end, :]

        for j in range(0, TILES_KV, 2):
            # C1V1C1V1C2V2: 每次迭代处理 2 个 K block + 1 个 V block
            oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

            Kj = K_BLOCKS[j]
            Sk_start = j * K_BLOCK_SIZE
            Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
            deq_scale_k_j = dequant_scale_k_expanded[:, :, Sk_start:Sk_end, :]

            S_ij = _compute_s_block(Qi, Kj, deq_scale_q_i, deq_scale_k_j, softmax_scale)
            mask_j = mask_global[:, :, Sq_start:Sq_end, Sk_start:Sk_end]
            # 尾列(全局列号>=kv_len-tail_len)的P直存bf16不量化e4m3,
            # 与V注入窗口同语义, 模拟kernel纯尾task的VF bf16直出
            tail_mask_j = None
            if vtail_tail_start is not None:
                col_j = torch.arange(Sk_start, Sk_end)
                tail_mask_j = col_j.view(1, 1, 1, -1) >= vtail_tail_start.view(
                    -1, 1, 1, 1
                )
            m_block_j, s_block_j, P_ij_drop = _online_softmax_update(
                S_ij, mask_j, mi, si, oi, ln_p_scale, tail_mask=tail_mask_j
            )

            if j + 1 < TILES_KV:
                # --- 第二个 K block (j+1) ---
                Kj1 = K_BLOCKS[j + 1]
                Sk1_start = (j + 1) * K_BLOCK_SIZE
                Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                deq_scale_k_j1 = dequant_scale_k_expanded[:, :, Sk1_start:Sk1_end, :]

                S_ij1 = _compute_s_block(
                    Qi, Kj1, deq_scale_q_i, deq_scale_k_j1, softmax_scale
                )
                mask_j1 = mask_global[:, :, Sq_start:Sq_end, Sk1_start:Sk1_end]
                tail_mask_j1 = None
                if vtail_tail_start is not None:
                    col_j1 = torch.arange(Sk1_start, Sk1_end)
                    tail_mask_j1 = col_j1.view(1, 1, 1, -1) >= vtail_tail_start.view(
                        -1, 1, 1, 1
                    )
                m_block_j1, s_block_j1, P_ij1_drop = _online_softmax_update(
                    S_ij1,
                    mask_j1,
                    m_block_j,
                    s_block_j,
                    oi,
                    ln_p_scale,
                    tail_mask=tail_mask_j1,
                )

                # V block: 一个 V_BLOCK_SIZE 对应两个 K_BLOCK_SIZE
                # [mm2语义对齐] 先 fp8 原值 dot 再乘 dv scale (模拟 L0C):
                # 若先反量化 V (V×dv→±inf), 0×±inf=NaN 会中毒整个 O —
                # 而硬件 raw dot 时 0×有限=0, scale 在 dot 之后应用 (0×scale=0),
                # 000108 极端值下 NPU out=-inf 而非 NaN 的关键区别。
                Vj = V_BLOCKS[j // 2]
                Sv_start = (j // 2) * V_BLOCK_SIZE
                Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                deq_scale_v_j = dequant_scale_v_expanded[:, :, Sv_start:Sv_end, :]
                n1 = Kj.shape[2]
                P_eff1 = P_ij_drop * torch.exp(m_block_j - m_block_j1)
                O_part1 = _p_dot_v_scaled(P_eff1, Vj[:, :, :n1, :], deq_scale_v_j)
                O_part2 = _p_dot_v_scaled(
                    P_ij1_drop,
                    Vj[:, :, n1 : n1 + Kj1.shape[2], :],
                    deq_scale_v_j[:, :, n1:, :],
                )
                P_ij_Vj = O_part1 + O_part2

                update_mul_si = torch.exp(mi - m_block_j1)
                si_new = (
                    update_mul_si * si
                    + s_block_j * torch.exp(m_block_j - m_block_j1)
                    + s_block_j1
                )
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j1
            else:
                Vj = V_BLOCKS[j // 2]
                Sv_start = j * K_BLOCK_SIZE
                Sv_end = min(Sv_start + K_BLOCK_SIZE, Skv)
                deq_scale_v_j = dequant_scale_v_expanded[:, :, Sv_start:Sv_end, :]

                P_ij_Vj = _p_dot_v_scaled(
                    P_ij_drop, Vj[:, :, : Kj.shape[2], :], deq_scale_v_j
                )
                update_mul_si = torch.exp(mi - m_block_j)
                si_new = update_mul_si * si + s_block_j
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j

    out = torch.cat(o_BLOCKS, dim=2)
    out_sum = torch.cat(s_BLOCKS, dim=2)
    # [极端值对齐] 无 EPSILON 除法: NPU 的 O/s 为真除法 — s=0 时 0/0=NaN、
    # ±inf/s=±inf 自然传播 (000108: s=0 行 NPU out=NaN)。正常行 s≥p_scale, eps 无影响。
    out = out / out_sum

    o_max = torch.cat(m_BLOCKS, dim=2)
    # [极端值对齐] 全 masked 判定改用 mask 直接判定: m 恒不更新(init=-3.4e38)的行
    # 除"无有效列"外还包括"有效列全为 -inf 溢出"行 (后者 NPU: lse=-inf, out=NaN),
    # 旧的 o_max<=minValue 判定会把二者混为 all_masked(lse=+inf, out=0)。
    row_valid = (~mask_global).any(dim=-1)[..., None]
    lse = torch.where(
        row_valid,
        o_max + torch.log(out_sum),
        torch.full_like(o_max, float("inf")),
    )
    out = torch.where(row_valid, out, torch.zeros_like(out))
    # [极端值对齐] 删除 nan_to_num(NaN→-inf): NPU 极端值行为是自然传播 —
    # m 对齐舍入使 exp 参数>0 → P=+inf → s=+inf → lse=+inf, O/s=NaN;
    # exp 参数<0 → P=0 → s=0 → lse=-inf, 0/0=NaN。CPU 自然传播即与 NPU 一致。
    logger.info("[CPU Golden] output=%s", out.shape)
    return out, lse


# ==============================================================================
# NPU 调用
# GRAPH_PATH: 0=单算子, 3=静态图, 5=动态图, 6=tiling下沉, 7=aclgraph
# PA 模式: Q 用 TND layout, K/V 走 PA 预处理 (block_table + block_size)
# 非 PA 模式: Q/K/V 均按 INPUT_LAYOUT 转换
# ==============================================================================


def _build_causal_mask():
    # sparse_mode=0 不需要 mask，其他模式需要上三角 causal mask
    if SPARSE_MODE == 0:
        return None
    shape = ATTN_MASK_SHAPE
    if not shape:
        shape = (2048, 2048)
    return torch.triu(torch.ones(shape, dtype=torch.int8), diagonal=1).npu()


def prepare_npu_inputs(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    cu_seqlens_q_t=None,
    cu_seqlens_kv_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """准备 NPU 侧入参

    返回字典的 key 与 _call_npu_fa_op 的形参名一一对应:
      q, k, v, mask,
      cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q, max_seqlen_kv,
      dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
      block_table, q_n, kv_n, softmax_scale,
      layout_q, layout_q_descale, layout_kv, layout_out, block_size, sparse_mode, out_dtype
    其中 cu_seqlens_q/kv、seqused_q/kv 为 python list (或 None), 由 _call_npu_fa_op 负责转 NPU tensor;
    其余 tensor 字段均为已就绪的 NPU tensor.
    """
    torch_npu.npu.set_device(int(DEVICE_ID))

    softmax_scale = SOFTMAX_SCALE

    Q_DTYPE = q_fp8.dtype
    DQ_DTYPE = dequant_scale_q.dtype
    P_SCALE_DTYPE = p_scale.dtype
    q_npu = q_fp8.contiguous().view(Q_DTYPE).npu()
    deq_q_npu = dequant_scale_q.view(DQ_DTYPE).npu()
    p_scale_npu = p_scale.view(P_SCALE_DTYPE).npu()

    out_dtype = torch.float16
    mask_arg = _build_causal_mask()

    if ENABLE_PA:
        K_DTYPE = k_fp8.dtype
        V_DTYPE = v_fp8.dtype
        DK_DTYPE = dequant_scale_k.dtype
        DV_DTYPE = dequant_scale_v.dtype
        k_npu = k_fp8.view(K_DTYPE).npu()
        v_npu = v_fp8.view(V_DTYPE).npu()
        deq_k_npu = dequant_scale_k.view(DK_DTYPE).npu()
        deq_v_npu = dequant_scale_v.view(DV_DTYPE).npu()

        logger.info("[NPU PA] kv_layout=%s", KV_CACHE_LAYOUT)
        logger.info("[NPU PA] k=%s, v=%s", k_npu.shape, v_npu.shape)
        logger.info("[NPU PA] k stride=%s, v stride=%s", k_npu.stride(), v_npu.stride())
        logger.info("[NPU PA] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)
        logger.info(
            "[NPU PA] deq_k stride=%s, deq_v stride=%s",
            deq_k_npu.stride(),
            deq_v_npu.stride(),
        )

        block_table_npu = (
            block_table_torch.npu()
            if isinstance(block_table_torch, torch.Tensor)
            else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
        )

        _pa_layout_kv_map = {"BnNBsD": "PA_BNBD", "BnBsND": "PA_BBND", "PA_NZ": "PA_NZ"}
        pa_layout_kv = _pa_layout_kv_map.get(KV_CACHE_LAYOUT, "PA_BNBD")

        # layout_kv 优先用 CSV 透传值 (LAYOUT_KV); 若未注入则回退到 kv_cache_layout 推导
        op_layout_kv = LAYOUT_KV if LAYOUT_KV else pa_layout_kv

        logger.info("[NPU] prepare PA inputs done.")
        return dict(
            q=q_npu,
            k=k_npu,
            v=v_npu,
            mask=mask_arg,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            cu_seqlens_q_t=cu_seqlens_q_t,
            cu_seqlens_kv_t=cu_seqlens_kv_t,
            seqused_q_t=seqused_q_t,
            seqused_kv_t=seqused_kv_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            dequant_scale_q=deq_q_npu,
            dequant_scale_k=deq_k_npu,
            dequant_scale_v=deq_v_npu,
            p_scale=p_scale_npu,
            block_table=block_table_npu,
            q_n=N_q,
            kv_n=N_kv,
            softmax_scale=softmax_scale,
            layout_q=LAYOUT_Q,
            layout_q_descale=LAYOUT_Q_DESCALE,
            layout_kv=op_layout_kv,
            layout_out=LAYOUT_OUT,
            block_size=BLOCK_SIZE,
            sparse_mode=SPARSE_MODE,
            out_dtype=out_dtype,
        )

    # 非 PA 模式
    K_DTYPE = k_fp8.dtype
    V_DTYPE = v_fp8.dtype
    DK_DTYPE = dequant_scale_k.dtype
    DV_DTYPE = dequant_scale_v.dtype
    k_npu = k_fp8.view(K_DTYPE).npu()
    v_npu = v_fp8.view(V_DTYPE).npu()
    deq_k_npu = dequant_scale_k.view(DK_DTYPE).npu()
    deq_v_npu = dequant_scale_v.view(DV_DTYPE).npu()
    logger.info("[NPU TND] k=%s, v=%s", k_npu.shape, v_npu.shape)
    logger.info("[NPU TND] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)
    logger.info(
        "[NPU TND] k is_contiguous=%s, v is_contiguous=%s",
        k_npu.is_contiguous(),
        v_npu.is_contiguous(),
    )
    logger.info(
        "[NPU TND] deq_k is_contiguous=%s, deq_v is_contiguous=%s",
        deq_k_npu.is_contiguous(),
        deq_v_npu.is_contiguous(),
    )

    logger.info("[NPU] prepare non-PA inputs done.")
    return dict(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        mask=mask_arg,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        cu_seqlens_q_t=cu_seqlens_q_t,
        cu_seqlens_kv_t=cu_seqlens_kv_t,
        seqused_q_t=seqused_q_t,
        seqused_kv_t=seqused_kv_t,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=deq_k_npu,
        dequant_scale_v=deq_v_npu,
        p_scale=p_scale_npu,
        block_table=None,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        layout_q=LAYOUT_Q,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=LAYOUT_KV,
        layout_out=LAYOUT_OUT,
        block_size=0,
        sparse_mode=SPARSE_MODE,
        out_dtype=out_dtype,
    )
