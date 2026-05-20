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

import torch
import torch_npu
import math
import result_compare_method

# ==============================================================================
# 配置区
# ==============================================================================
B = 1
N_q = 1           # query heads
N_kv = 1           # kv heads
D = 128

ENABLE_ROPE = False
D_rope = 0

ACTUAL_SEQ_Q = [4]
ACTUAL_SEQ_KV = [512]

ENABLE_PA = True
BLOCK_SIZE = 512

SPARSE_MODE = 3
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 32

# Layout 选择
INPUT_LAYOUT = "TND"  # 支持: BNSD, BSND, TND
Q_SCALE_LAYOUT = "AUTO"  # 支持: AUTO, TND, N2TGD (GQA 专用)
Q_SCALE_AUTO_GS1_THRESHOLD = 80
P_SCALE = 1.0

# PA KV Cache Layout
# BnNBsD: [BlockNum, N, BlockSize, D]
# PA_NZ: fp8=[Bn,N,D//32,Bs,32], Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]
KV_CACHE_LAYOUT = "PA_NZ"

E8M0_MIN_POSITIVE = 2**(-127)

SEED_Q = 54
SEED_K = 3
SEED_V = 4
SEED_QR = 8
SEED_KR = 9

# ==============================================================================
# 量化 scale 计算
# ==============================================================================

def get_mxfp8_per_token_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Q/K per-token-group quant_scale, 输出: (B, N, S, ceil(D/group))"""
    if fp8_dtype == torch.float8_e4m3fn:
        emax_elem = 8
    elif fp8_dtype == torch.float8_e5m2:
        emax_elem = 15
    else:
        raise ValueError(f"{fp8_dtype} not supported")

    dim1, dim2, dim3, dim4 = tensor.shape
    scale = torch.ones([dim1, dim2, dim3, math.ceil(dim4 / group_size)], dtype=torch.float32)

    for b in range(dim1):
        for n in range(dim2):
            for s in range(dim3):
                for d in range(0, dim4, group_size):
                    chunk = tensor[b, n, s, d:min(d + group_size, dim4)]
                    if torch.all(chunk == 0):
                        scale[b, n, s, d // group_size] = 1.0
                        continue
                    max_val = torch.max(torch.abs(chunk)).clamp(min=1e-12)
                    shared_exp = torch.floor(torch.log2(max_val)) - emax_elem
                    scale[b, n, s, d // group_size] = 2 ** shared_exp
    return scale



def get_mxfp8_per_channel_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """V per-channel-group quant_scale, 输出: (B, N, ceil(S/group), D)"""
    if fp8_dtype == torch.float8_e4m3fn:
        emax_elem = 8
    elif fp8_dtype == torch.float8_e5m2:
        emax_elem = 15
    else:
        raise ValueError(f"{fp8_dtype} not supported")

    dim1, dim2, dim3, dim4 = tensor.shape
    scale = torch.ones([dim1, dim2, math.ceil(dim3 / group_size), dim4], dtype=torch.float32)

    for b in range(dim1):
        for n in range(dim2):
            for d in range(dim4):
                for s in range(0, dim3, group_size):
                    chunk = tensor[b, n, s:min(s + group_size, dim3), d]
                    if torch.all(chunk == 0):
                        scale[b, n, s // group_size, d] = 1.0
                        continue
                    max_val = torch.max(torch.abs(chunk)).clamp(min=1e-12)
                    shared_exp = torch.floor(torch.log2(max_val)) - emax_elem
                    scale[b, n, s // group_size, d] = 2 ** shared_exp
    return scale


def mxfp8_per_token_group_quant(tensor, quant_scale, group_size=32):
    dim1, dim2, dim3, dim4 = tensor.shape
    result = torch.zeros_like(tensor, dtype=torch.float32)
    for b in range(dim1):
        for n in range(dim2):
            for s in range(dim3):
                for d in range(0, dim4, group_size):
                    d_end = min(d + group_size, dim4)
                    qs = quant_scale[b, n, s, d // group_size].item()
                    result[b, n, s, d:d_end] = tensor[b, n, s, d:d_end] * qs
    return result


def mxfp8_per_channel_group_quant(tensor, quant_scale, group_size=32):
    dim1, dim2, dim3, dim4 = tensor.shape
    result = torch.zeros_like(tensor, dtype=torch.float32)
    for b in range(dim1):
        for n in range(dim2):
            for d in range(dim4):
                for s in range(0, dim3, group_size):
                    s_end = min(s + group_size, dim3)
                    qs = quant_scale[b, n, s // group_size, d].item()
                    result[b, n, s:s_end, d] = tensor[b, n, s:s_end, d] * qs
    return result


def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    factor = num_heads // num_kv_heads
    B, _, S, D = kv_tensor.shape
    result = torch.zeros([B, num_heads, S, D], dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i:i + 1, :, :] = kv_tensor[:, i // factor:i // factor + 1, :, :]
    return result


# ==============================================================================
# Layout 转换函数 - 数据 (Q/K/V)
# ==============================================================================

def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    """Q/K/V BNSD → 各种 layout，支持 fp8 tensor"""
    tensor = tensor_bnsd if isinstance(tensor_bnsd, torch.Tensor) else torch.as_tensor(tensor_bnsd)
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)
    
    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).reshape(B, max_org_s, N * D).contiguous()
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t:t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_kv_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    """K/V BNSD → 各种 layout"""
    return convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout)


def convert_qk_rope_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    return convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout)


# ==============================================================================
# Layout 转换函数 - Scale (Q/K/V)
# ==============================================================================

def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 → e8m0fnu (uint8)，提取 IEEE 754 biased exponent"""
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0fnu 没有 0 值语义；非有限值进入 0xFF 会在 NPU 侧变 NaN。"""
    if isinstance(scale, torch.Tensor):
        result = scale.to(torch.float32).clone()
        bad_mask = ~torch.isfinite(result)
        zero_mask = result == 0
        bad_count = int(bad_mask.sum().item())
        zero_count = int(zero_mask.sum().item())
        if bad_count:
            print(f"[WARN] {name}: replace {bad_count} non-finite scale values before e8m0 packing")
            result[bad_mask] = E8M0_MIN_POSITIVE
        if zero_count:
            result[zero_mask] = E8M0_MIN_POSITIVE
        return result

    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad_mask = ~torch.isfinite(result)
    zero_mask = result == 0
    bad_count = int(bad_mask.sum().item())
    zero_count = int(zero_mask.sum().item())
    if bad_count:
        print(f"[WARN] {name}: replace {bad_count} non-finite scale values before e8m0 packing")
        result[bad_mask] = E8M0_MIN_POSITIVE
    if zero_count:
        result[zero_mask] = E8M0_MIN_POSITIVE
    return result


def fp32_to_e8m0fnu_safe(scale, name="scale"):
    scale_safe = sanitize_e8m0_scale(scale, name)
    packed = fp32_to_e8m0fnu(scale_safe)
    nan_byte_count = int((packed == 0xFF).sum().item())
    if nan_byte_count:
        raise ValueError(f"{name}: {nan_byte_count} values would become e8m0fnu NaN (0xFF)")
    return packed


def canonical_q_scale_layout(layout):
    layout = (layout or "AUTO").upper()
    if layout not in ("AUTO", "TND", "N2TGD"):
        raise ValueError(f"Unsupported Q scale layout: {layout}")
    return layout


def resolve_q_scale_layout(seq_lens, layout=None):
    """实际算子按 G*S1 自动选择 Q scale layout: >80 用 TND，否则用 N2TGD。"""
    requested = canonical_q_scale_layout(layout or Q_SCALE_LAYOUT)
    if N_kv <= 0 or N_q % N_kv != 0:
        raise ValueError(f"N_q must be divisible by N_kv, got N_q={N_q}, N_kv={N_kv}")
    group = N_q // N_kv
    s1 = max(int(item) for item in seq_lens)
    gs1 = group * s1
    if requested == "AUTO":
        resolved = "TND" if gs1 > Q_SCALE_AUTO_GS1_THRESHOLD else "N2TGD"
    else:
        resolved = requested
    return resolved, group, s1, gs1


def pack_qk_scale_for_npu(scale_flat):
    """Q/K scale packing: (..., D) → (..., D//2, 2)"""
    orig_shape = scale_flat.shape
    last_dim = orig_shape[-1]
    new_shape = orig_shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)


def pack_v_scale_for_npu(scale_flat):
    """
    V scale packing: (..., Sg, D) → (..., Sg//2, D, 2)
    """
    Sg = scale_flat.shape[-2]
    D = scale_flat.shape[-1]
    prefix_shape = scale_flat.shape[:-2]
    
    # 奇数行 pad 用 E8M0_MIN_POSITIVE，避免 0.0 转 e8m0fnu 后变 NaN
    if Sg % 2 != 0:
        pad_shape = prefix_shape + (1, D)
        pad = torch.full(pad_shape, E8M0_MIN_POSITIVE, dtype=scale_flat.dtype, device=scale_flat.device)
        scale_flat = torch.cat([scale_flat, pad], dim=-2)
        Sg += 1
    
    out_Sg = Sg // 2
    out_shape = prefix_shape + (out_Sg, D, 2)

    result = torch.zeros(out_shape, dtype=scale_flat.dtype, device=scale_flat.device)
    result[..., 0] = scale_flat[..., ::2, :]
    result[..., 1] = scale_flat[..., 1::2, :]
    return result


def convert_q_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout):
    """Q scale BNSD → 各种 layout (已 packed: D//2, 2)"""
    layout = canonical_q_scale_layout(layout)
    B, N, _, Dg = scale_bnsd.shape
    max_org_s = max(seq_lens)
    Dg_half = Dg // 2
    
    if layout == "BNSD":
        return scale_bnsd[:, :, :max_org_s, :].reshape(B, N, max_org_s, Dg_half, 2)
    elif layout == "BSND":
        return scale_bnsd[:, :, :max_org_s, :].permute(0, 2, 1, 3).reshape(B, max_org_s, N, Dg_half, 2)
    elif layout == "BSH":
        return scale_bnsd[:, :, :max_org_s, :].permute(0, 2, 1, 3).reshape(B, max_org_s, N * Dg_half, 2)
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, N, Dg_half, 2), dtype=scale_bnsd.dtype, device=scale_bnsd.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t:t + act_s, n, :, :] = scale_bnsd[b, n, :act_s, :].reshape(act_s, Dg_half, 2)
            t += act_s
        return result
    elif layout == "N2TGD":
        # 先转 TND，再做 N2TGD 转换
        tnd_result = convert_q_scale_bnsd_to_layout(scale_bnsd, seq_lens, "TND")
        return convert_q_scale_tnd_to_n2tgd_layout(tnd_result, N_kv)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_k_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout):
    """K scale BNSD → 各种 layout"""
    return convert_q_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout)


def convert_v_scale_bnsd_to_layout(scale_bnsd, seq_lens, layout, group_size=32):
    """
    V scale BNSD → 各种 layout
    """
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)
    
    # 奇数行 pad 用 E8M0_MIN_POSITIVE
    if actual_Sg % 2 != 0:
        actual_Sg_padded = actual_Sg + 1
    else:
        actual_Sg_padded = actual_Sg
    
    S_out = actual_Sg_padded // 2
    
    if layout == "BNSD":
        transposed = scale_bnsd[:, :, :actual_Sg, :]
        if actual_Sg % 2 != 0:
            pad = torch.full((B, N, 1, D), E8M0_MIN_POSITIVE, dtype=transposed.dtype, device=transposed.device)
            transposed = torch.cat([transposed, pad], dim=2)
        result = torch.zeros((B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device)
        result[..., 0] = transposed[..., ::2, :]
        result[..., 1] = transposed[..., 1::2, :]
        return result
    elif layout == "BSND":
        transposed = scale_bnsd[:, :, :actual_Sg, :].permute(0, 2, 1, 3)
        if actual_Sg % 2 != 0:
            pad = torch.full((B, 1, N, D), E8M0_MIN_POSITIVE, dtype=transposed.dtype, device=transposed.device)
            transposed = torch.cat([transposed, pad], dim=1)
        result = torch.zeros((B, S_out, N, D, 2), dtype=torch.float32, device=scale_bnsd.device)
        result[..., 0] = transposed[:, ::2, :, :]
        result[..., 1] = transposed[:, 1::2, :, :]
        return result
    elif layout == "BSH":
        transposed = scale_bnsd[:, :, :actual_Sg, :].permute(0, 2, 1, 3).reshape(B, actual_Sg, N * D)
        if actual_Sg % 2 != 0:
            pad = torch.full((B, 1, N * D), E8M0_MIN_POSITIVE, dtype=transposed.dtype, device=transposed.device)
            transposed = torch.cat([transposed, pad], dim=1)
        result = torch.zeros((B, S_out, N * D, 2), dtype=torch.float32, device=scale_bnsd.device)
        result[..., 0] = transposed[:, ::2, :]
        result[..., 1] = transposed[:, 1::2, :]
        return result
    elif layout == "TND":
        # 计算总 T 维度 (考虑奇数行 pad)
        Tv = 0
        for seq_len in seq_lens:
            sg = math.ceil(seq_len / group_size)
            sg_padded = sg + (sg % 2)
            Tv += sg_padded // 2
        
        result = torch.zeros((Tv, N, D, 2), dtype=torch.float32, device=scale_bnsd.device)
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
                # 奇数行 pad 用 1
                if sg % 2 != 0:
                    pad = torch.full((1, D), E8M0_MIN_POSITIVE, dtype=src.dtype, device=src.device)
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
    """
    T, N, D, _ = tensor_tnd.shape
    G = N // num_kv_heads
    tensor_reshape = tensor_tnd.reshape(T, num_kv_heads, G, D, 2)
    return tensor_reshape.permute(1, 0, 2, 3, 4).contiguous()  # → (N_kv, T, G, D, 2)


def convert_q_scale_tnd_to_n2gtd_layout(tensor_tnd, num_kv_heads):
    return convert_q_scale_tnd_to_n2tgd_layout(tensor_tnd, num_kv_heads)


# ==============================================================================
# PA 格式转换 - mxfp8_pa_preprocessing
# ==============================================================================

def mxfp8_pa_preprocessing(tensor_bnsd, seq_lens, block_size, block_table, 
                           is_vscale=False, is_scale=False, is_rope=False,
                           kv_layout="BnNBsD", group_size=32):
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
    tensor_bnsd = tensor_bnsd if isinstance(tensor_bnsd, torch.Tensor) else torch.as_tensor(tensor_bnsd)
    block_table = block_table if isinstance(block_table, torch.Tensor) else torch.as_tensor(block_table)
    B, N, S, D = tensor_bnsd.shape

    if is_rope and is_scale:
        raise ValueError("rope preprocessing does not support scale packing")

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_scale_pack_ratio = group_size * 2
            pack_seq_lens = [math.ceil(act_s / v_scale_pack_ratio) for act_s in seq_lens]
            pack_block_size = math.ceil(block_size / v_scale_pack_ratio)
            total_block_num = sum(math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens)
            out_shape = (total_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K scale D must be even for packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (sum(math.ceil(act_s / block_size) for act_s in seq_lens), N, block_size, D // 2, 2)
    else:
        tensor_processed = tensor_bnsd
        pack_seq_lens = seq_lens
        pack_block_size = block_size
        out_shape = (sum(math.ceil(act_s / block_size) for act_s in seq_lens), N, block_size, D)

    if is_scale:
        fill_value = E8M0_MIN_POSITIVE
    else:
        fill_value = 0

    out_cache = torch.full(out_shape, fill_value, dtype=tensor_processed.dtype, device=tensor_processed.device)
    block_num = [math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(block_num[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * pack_block_size
            valid_len = min(pack_block_size, pack_seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[blockid, :, :valid_len] = tensor_processed[b, :, block_offset:block_offset + valid_len]

    # 根据 kv_layout 调整输出
    if kv_layout == "BnNBsD":
        return out_cache
    elif kv_layout == "BnBsND":
        return out_cache.transpose(1, 2).contiguous()
    elif kv_layout == "PA_NZ":
        Bn, KV_N, Bs, KV_D = out_cache.shape[:4]
        if not is_scale:
            inner = 16 if is_rope else 32
            if KV_D % inner != 0:
                raise ValueError(f"PA_NZ rope D must be divisible by {inner}, got {KV_D}")
            reshaped = out_cache.reshape(Bn, KV_N, Bs, KV_D // inner, inner)
            return reshaped.permute(0, 1, 3, 2, 4).contiguous()
        elif not is_vscale:
            # K scale: [Bn, KV_N, Bs, KV_D, 2] → [Bn, KV_N, Bs//16, KV_D, 16, 2]
            reshaped = out_cache.reshape(Bn, KV_N, Bs // 16, 16, KV_D, 2)
            return reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
        else:
            # V scale: [Bn, KV_N, Bs, KV_D, 2] → [Bn, KV_N, KV_D//16, Bs, 16, 2]
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
    scale_bnsd = scale_bnsd if isinstance(scale_bnsd, torch.Tensor) else torch.as_tensor(scale_bnsd)
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)
    
    # 取实际数据
    transposed = scale_bnsd[:, :, :actual_Sg, :]
    
    # 奇数行 pad 用 E8M0_MIN_POSITIVE
    if actual_Sg % 2 != 0:
        pad = torch.full((B, N, 1, D), E8M0_MIN_POSITIVE, dtype=transposed.dtype, device=transposed.device)
        transposed = torch.cat([transposed, pad], dim=2)
        actual_Sg += 1
    
    S_out = actual_Sg // 2
    result = torch.zeros((B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device)
    result[..., 0] = transposed[..., ::2, :]
    result[..., 1] = transposed[..., 1::2, :]
    return result


def bnsd_to_pa_kv(tensor_bnsd, seq_lens, block_size, block_table, kv_layout="BnNBsD"):
    """BNSD KV (fp8) → PA, 输出 [BlockNum, N, BlockSize, D]"""
    return mxfp8_pa_preprocessing(
        tensor_bnsd, seq_lens, block_size, block_table,
        is_scale=False, kv_layout=kv_layout
    )


def bnsd_to_pa_kv_scale_token_group(scale_bnsd, seq_lens, block_size, block_table, 
                                      kv_layout="BnNBsD", group_size=32):
    """K scale BNSD → PA, 输出 [BlockNum, N, BlockSize, D//2, 2]"""
    return mxfp8_pa_preprocessing(
        scale_bnsd, seq_lens, block_size, block_table,
        is_scale=True, is_vscale=False, kv_layout=kv_layout, group_size=group_size
    )


def bnsd_to_pa_v_scale_channel_group(scale_bnsd, seq_lens, block_size, block_table,
                                       kv_layout="BnNBsD", group_size=32):
    """V scale BNSD → PA, 输出 [BlockNum, N, BlockSize//64, D, 2]"""
    return mxfp8_pa_preprocessing(
        scale_bnsd, seq_lens, block_size, block_table,
        is_scale=True, is_vscale=True, kv_layout=kv_layout, group_size=group_size
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
    """生成 BNSD FP16 Q/K/V 并做 MXFP8 量化，使用原始序列长度"""
    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV)

    print(f"[INFO] max_sq={max_sq}, max_skv={max_skv}")

    torch.manual_seed(SEED_Q)
    q_fp16 = torch.zeros(B, N_q, max_sq, D, dtype=torch.float16)
    q_fp16[:, :, :, :] = torch.randn(B, N_q, max_sq, D, dtype=torch.float16)

    torch.manual_seed(SEED_K)
    k_fp16 = torch.zeros(B, N_kv, max_skv, D, dtype=torch.float16)
    k_fp16[:, :, :, :] = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1)

    torch.manual_seed(SEED_V)
    v_fp16 = torch.zeros(B, N_kv, max_skv, D, dtype=torch.float16)
    v_fp16[:, :, :, :] = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1)

    qr_bf16 = None
    kr_bf16 = None
    if ENABLE_ROPE and D_rope > 0:
        torch.manual_seed(SEED_QR)
        qr_bf16 = torch.randn(B, N_q, max_sq, D_rope, dtype=torch.bfloat16)
        torch.manual_seed(SEED_KR)
        kr_bf16 = torch.randn(B, N_kv, max_skv, D_rope, dtype=torch.bfloat16)

    print(f"[INFO] q_fp16={q_fp16.shape}, k_fp16={k_fp16.shape}, v_fp16={v_fp16.shape}")

    quant_scale_q = get_mxfp8_per_token_group_quant_scale(q_fp16, FP8_DTYPE, QUANT_GROUP_SIZE)
    quant_scale_k = get_mxfp8_per_token_group_quant_scale(k_fp16, FP8_DTYPE, QUANT_GROUP_SIZE)
    quant_scale_v = get_mxfp8_per_channel_group_quant_scale(v_fp16, FP8_DTYPE, QUANT_GROUP_SIZE)

    dequant_scale_q = 1 / quant_scale_q
    dequant_scale_k = 1 / quant_scale_k
    dequant_scale_v = 1 / quant_scale_v

    v_sg = dequant_scale_v.shape[2]
    print(f"[INFO] V scale Sg={v_sg}, 是否奇数={v_sg % 2 != 0}")

    q_fp8 = mxfp8_per_token_group_quant(q_fp16, quant_scale_q, QUANT_GROUP_SIZE).to(FP8_DTYPE)
    k_fp8 = mxfp8_per_token_group_quant(k_fp16, quant_scale_k, QUANT_GROUP_SIZE).to(FP8_DTYPE)
    v_fp8 = mxfp8_per_channel_group_quant(v_fp16, quant_scale_v, QUANT_GROUP_SIZE).to(FP8_DTYPE)

    block_table_torch = None
    if ENABLE_PA:
        block_num = sum(math.ceil(s / BLOCK_SIZE) for s in ACTUAL_SEQ_KV)
        max_blocks = max(math.ceil(s / BLOCK_SIZE) for s in ACTUAL_SEQ_KV)
        block_idx_list = torch.randperm(block_num, dtype=torch.int32)
        block_table_torch = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(ACTUAL_SEQ_KV[b] / BLOCK_SIZE)
            for j in range(n_blocks):
                block_table_torch[b, j] = block_idx_list[idx]
                idx += 1

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    return (q_fp8, k_fp8, v_fp8,
            dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
            qr_bf16, kr_bf16, block_table_torch)


# ==============================================================================
# CPU Golden
# ==============================================================================

def cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8,
                     dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                     actual_seq_q, actual_seq_kv,
                     qr_bf16=None, kr_bf16=None):
    """CPU Flash Attention golden with MXFP8, C1V1C1V1C2V2 流水"""
    EPSILON = 1e-20
    Q_BLOCK_SIZE = 128
    K_BLOCK_SIZE = 256
    V_BLOCK_SIZE = 512

    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    if N_q != N_kv:
        print("[INFO] GQA 广播")
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        dequant_scale_k = broadcast_kv(N_q, N_kv, dequant_scale_k)
        dequant_scale_v = broadcast_kv(N_q, N_kv, dequant_scale_v)
        if kr_bf16 is not None:
            kr_bf16 = broadcast_kv(N_q, N_kv, kr_bf16)

    b, n, s, d = q_tensor.shape
    d_total = d + (D_rope if ENABLE_ROPE else 0)
    dv = v_tensor.shape[-1]
    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]

    out = torch.zeros([b, n, Sq, dv], dtype=torch.float32)  # numerator buffer; normalize once at the end
    o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]
    o_max = torch.ones(q_tensor.shape[:-1])[..., None] * torch.finfo(torch.float).min

    TILES_Q = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    TILES_KV = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32)
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32)
    q_lens_acl = q_lens_t.view(b, 1, 1, 1)
    k_lens_acl = k_lens_t.view(b, 1, 1, 1)

    q_range = torch.arange(Sq).view(1, 1, -1, 1)
    k_range = torch.arange(Skv).view(1, 1, 1, -1)
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if SPARSE_MODE == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        mask_global = causal_mask | q_padding_mask | k_padding_mask
    else:
        mask_global = q_padding_mask | k_padding_mask

    Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
    V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
    o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))

    if ENABLE_ROPE and qr_bf16 is not None:
        Qr_BLOCKS = list(torch.split(qr_bf16, Q_BLOCK_SIZE, dim=2))
        Kr_BLOCKS = list(torch.split(kr_bf16, K_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    print(f"[CPU Golden] TILES_Q={TILES_Q}, TILES_KV={TILES_KV}, Sq={Sq}, Skv={Skv}")

    for i in range(TILES_Q):
        Qi = Q_BLOCKS[i]
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
        deq_scale_q_i = dequant_scale_q[:, :, Sq_start:Sq_end, :]
        deq_scale_q_i_expanded = deq_scale_q_i.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)

        for j in range(0, TILES_KV, 2):
            oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

            Kj = K_BLOCKS[j]
            Sk_start = j * K_BLOCK_SIZE
            Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
            deq_scale_k_j = dequant_scale_k[:, :, Sk_start:Sk_end, :]
            deq_scale_k_j_expanded = deq_scale_k_j.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)

            S_ij = torch.matmul(Qi * deq_scale_q_i_expanded, (Kj * deq_scale_k_j_expanded).permute(0, 1, 3, 2))
            if ENABLE_ROPE and qr_bf16 is not None:
                Qri, Krj = Qr_BLOCKS[i], Kr_BLOCKS[j]
                Qri = Qri.to(torch.float32)
                Krj = Krj.to(torch.float32)
                S_ij += torch.matmul(Qri, Krj.permute(0, 1, 3, 2))

            S_ij = S_ij / math.sqrt(d_total)
            causal_mask_j = mask_global[:, :, Sq_start:Sq_end, Sk_start:Sk_end]
            S_ij = S_ij.masked_fill(causal_mask_j, float('-inf'))

            m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
            m_block_j = m_block_j - ln_p_scale
            m_block_j = torch.max(mi, m_block_j)
            P_ij_raw = torch.exp(S_ij - m_block_j)
            s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
            P_ij_drop = P_ij_raw.to(FP8_DTYPE).to(torch.float32)

            if j + 1 < TILES_KV:
                Kj1 = K_BLOCKS[j + 1]
                Sk1_start = (j + 1) * K_BLOCK_SIZE
                Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                deq_scale_k_j1 = dequant_scale_k[:, :, Sk1_start:Sk1_end, :]
                deq_scale_k_j1_expanded = deq_scale_k_j1.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)

                S_ij1 = torch.matmul(Qi * deq_scale_q_i_expanded, (Kj1 * deq_scale_k_j1_expanded).permute(0, 1, 3, 2))
                if ENABLE_ROPE and qr_bf16 is not None:
                    Krj1 = Kr_BLOCKS[j + 1]
                    Krj1 = Krj1.to(torch.float32)
                    S_ij1 += torch.matmul(Qri, Krj1.permute(0, 1, 3, 2))

                S_ij1 = S_ij1 / math.sqrt(d_total)
                causal_mask_j1 = mask_global[:, :, Sq_start:Sq_end, Sk1_start:Sk1_end]
                S_ij1 = S_ij1.masked_fill(causal_mask_j1, float('-inf'))

                m_block_j1, _ = torch.max(S_ij1, dim=-1, keepdims=True)
                m_block_j1 = m_block_j1 - ln_p_scale
                m_block_j1 = torch.max(m_block_j, m_block_j1)
                P_ij1_raw = torch.exp(S_ij1 - m_block_j1)
                s_block_j1 = torch.sum(P_ij1_raw, dim=-1, keepdims=True)
                update_mul_j = torch.exp(m_block_j - m_block_j1)
                update_mul_si = torch.exp(mi - m_block_j1)
                P_ij1_drop = P_ij1_raw.to(FP8_DTYPE).to(torch.float32)

                Vj = V_BLOCKS[j // 2]
                Sv_start = (j // 2) * V_BLOCK_SIZE
                Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                v_scale_start = Sv_start // QUANT_GROUP_SIZE
                v_scale_end = (Sv_end - 1) // QUANT_GROUP_SIZE + 1

                deq_scale_v_j = dequant_scale_v[:, :, v_scale_start:v_scale_end, :]
                deq_scale_v_j_tmp = deq_scale_v_j.repeat_interleave(QUANT_GROUP_SIZE, dim=2)
                deq_scale_v_j_expanded = deq_scale_v_j_tmp[:, :, :Vj.shape[2], :]
                Vj_dequant = Vj * deq_scale_v_j_expanded

                V_part1 = Vj_dequant[:, :, :Kj.shape[2], :]
                V_part2 = Vj_dequant[:, :, Kj.shape[2]:Kj.shape[2] + Kj1.shape[2], :]
                P_ij_Vj = torch.matmul(P_ij_drop, V_part1) + torch.matmul(P_ij1_drop, V_part2)

                si_new = update_mul_si * si + s_block_j * update_mul_j + s_block_j1
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j1
            else:
                Vj = V_BLOCKS[j // 2]
                Sv_start = j * K_BLOCK_SIZE
                Sv_end = min(Sv_start + K_BLOCK_SIZE, Skv)
                v_scale_start = Sv_start // QUANT_GROUP_SIZE
                v_scale_end = (Sv_end - 1) // QUANT_GROUP_SIZE + 1

                deq_scale_v_j = dequant_scale_v[:, :, v_scale_start:v_scale_end, :]
                deq_scale_v_j_tmp = deq_scale_v_j.repeat_interleave(QUANT_GROUP_SIZE, dim=2)
                deq_scale_v_j_expanded = deq_scale_v_j_tmp[:, :, :Kj.shape[2], :]
                Vj_expanded = Vj[:, :, :Kj.shape[2], :]
                Vj_dequant = Vj_expanded * deq_scale_v_j_expanded

                P_ij_Vj = torch.matmul(P_ij_drop, Vj_dequant)
                update_mul_si = torch.exp(mi - m_block_j)
                si_new = update_mul_si * si + s_block_j
                o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
                s_BLOCKS[i] = si_new
                m_BLOCKS[i] = m_block_j

    out = torch.cat(o_BLOCKS, dim=2)
    out_sum = torch.cat(s_BLOCKS, dim=2)
    out = out / (out_sum + EPSILON)
    print(f"[CPU Golden] output={out.shape}")
    return out


# ==============================================================================
# NPU 调用
# ==============================================================================

def npu_mxfp8_fa(q_fp8, k_fp8, v_fp8,
                 dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                 actual_seq_q, actual_seq_kv,
                 block_table_torch=None,
                 qr_bf16=None, kr_bf16=None):
    """调用 NPU 算子，支持 N2TGD layout"""
    softmax_scale = 1.0 / math.sqrt(D + (D_rope if ENABLE_ROPE else 0))

    q_runtime_layout, q_group, q_s1, q_gs1 = resolve_q_scale_layout(actual_seq_q)
    print(f"[NPU] Q layout auto: G={q_group}, S1={q_s1}, G*S1={q_gs1}, q_layout={q_runtime_layout}")

    npu_input_layout = "TND" if ENABLE_PA else INPUT_LAYOUT
    q_npu = convert_q_bnsd_to_layout(q_fp8, actual_seq_q, npu_input_layout).contiguous().view(FP8_DTYPE).npu()
    print(f"[NPU {npu_input_layout}] q={q_npu.shape}")

    q_scale_e8m0 = fp32_to_e8m0fnu_safe(convert_q_scale_bnsd_to_layout(dequant_scale_q, actual_seq_q, q_runtime_layout), "Q scale")
    deq_q_npu = q_scale_e8m0.npu()
    print(f"[NPU] Q scale layout={q_runtime_layout}, shape={q_scale_e8m0.shape}")

    p_scale_npu = p_scale.npu()

    if ENABLE_PA:
        k_pa = mxfp8_pa_preprocessing(k_fp8, actual_seq_kv, BLOCK_SIZE, block_table_torch,
                                       is_scale=False, kv_layout=KV_CACHE_LAYOUT)
        v_pa = mxfp8_pa_preprocessing(v_fp8, actual_seq_kv, BLOCK_SIZE, block_table_torch,
                                       is_scale=False, kv_layout=KV_CACHE_LAYOUT)
        k_npu = k_pa.contiguous().view(FP8_DTYPE).npu()
        v_npu = v_pa.contiguous().view(FP8_DTYPE).npu()
        
        k_scale_pa = mxfp8_pa_preprocessing(dequant_scale_k, actual_seq_kv, BLOCK_SIZE, block_table_torch,
                                             is_scale=True, is_vscale=False, kv_layout=KV_CACHE_LAYOUT)
        v_scale_pa = mxfp8_pa_preprocessing(dequant_scale_v, actual_seq_kv, BLOCK_SIZE, block_table_torch,
                                             is_scale=True, is_vscale=True, kv_layout=KV_CACHE_LAYOUT)

        k_scale_e8m0_pa = fp32_to_e8m0fnu_safe(k_scale_pa, "K PA scale")
        v_scale_e8m0_pa = fp32_to_e8m0fnu_safe(v_scale_pa, "V PA scale")
        
        deq_k_npu = k_scale_e8m0_pa.npu()
        deq_v_npu = v_scale_e8m0_pa.npu()
        
        print(f"[NPU PA] kv_layout={KV_CACHE_LAYOUT}")
        print(f"[NPU PA] k={k_npu.shape}, v={v_npu.shape}")
        print(f"[NPU PA] deq_k={deq_k_npu.shape}, deq_v={deq_v_npu.shape}")

        block_table_npu = block_table_torch.npu() if isinstance(block_table_torch, torch.Tensor) else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
        accum_seq_q = make_accum_seq(actual_seq_q)
        
        # SPARSE_MODE=0时不传mask，其他模式传 causal mask
        if SPARSE_MODE == 0:
            mask_arg = None
        else:
            mask_arg = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()

        if ENABLE_ROPE:
            if qr_bf16 is None or kr_bf16 is None:
                raise ValueError("ENABLE_ROPE=True requires qr_bf16 and kr_bf16")
            q_rope_npu = convert_qk_rope_bnsd_to_layout(qr_bf16, actual_seq_q, npu_input_layout).npu()
            k_rope_pa = mxfp8_pa_preprocessing(
                kr_bf16,
                actual_seq_kv,
                BLOCK_SIZE,
                block_table_torch,
                is_rope=True,
                kv_layout=KV_CACHE_LAYOUT,
            )
            k_rope_npu = k_rope_pa.npu()
            print(f"[NPU PA] q_rope={q_rope_npu.shape}, k_rope={k_rope_npu.shape}")
        else:
            q_rope_npu = None
            k_rope_npu = None

        print("[NPU] 调用 PA 模式...")
        npu_out = torch_npu.npu_fused_infer_attention_score_v2(
            q_npu, k_npu, v_npu,
            query_rope=q_rope_npu,
            key_rope=k_rope_npu,
            atten_mask=mask_arg,
            actual_seq_qlen=accum_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            dequant_scale_query=deq_q_npu,
            dequant_scale_key=deq_k_npu,
            dequant_scale_value=deq_v_npu,
            num_query_heads=N_q,
            num_key_value_heads=N_kv,
            block_size=BLOCK_SIZE,
            block_table=block_table_npu,
            softmax_scale=softmax_scale,
            input_layout="TND",
            sparse_mode=SPARSE_MODE,
            query_quant_mode=6,
            key_quant_mode=6,
            value_quant_mode=8,
            query_dtype=FP8_DTYPE,
            key_dtype=FP8_DTYPE,
            value_dtype=FP8_DTYPE,
            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
            quant_scale_p=p_scale_npu,
        )
    else:
        k_npu = convert_kv_bnsd_to_layout(k_fp8, actual_seq_kv, npu_input_layout).contiguous().view(FP8_DTYPE).npu()
        v_npu = convert_kv_bnsd_to_layout(v_fp8, actual_seq_kv, npu_input_layout).contiguous().view(FP8_DTYPE).npu()
        print(f"[NPU {npu_input_layout}] k={k_npu.shape}, v={v_npu.shape}")

        k_scale_e8m0 = fp32_to_e8m0fnu_safe(convert_k_scale_bnsd_to_layout(dequant_scale_k, actual_seq_kv, npu_input_layout), "K scale")
        v_scale_e8m0 = fp32_to_e8m0fnu_safe(convert_v_scale_bnsd_to_layout(dequant_scale_v, actual_seq_kv, npu_input_layout), "V scale")
        deq_k_npu = k_scale_e8m0.npu()
        deq_v_npu = v_scale_e8m0.npu()
        print(f"[NPU {npu_input_layout}] K scale shape={k_scale_e8m0.shape}, V scale shape={v_scale_e8m0.shape}")

        accum_seq_q = make_accum_seq(actual_seq_q)
        accum_seq_kv = make_accum_seq(actual_seq_kv)
        
        if SPARSE_MODE == 0:
            mask_arg = None
        else:
            mask_arg = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()

        if ENABLE_ROPE:
            if qr_bf16 is None or kr_bf16 is None:
                raise ValueError("ENABLE_ROPE=True requires qr_bf16 and kr_bf16")
            q_rope_npu = convert_qk_rope_bnsd_to_layout(qr_bf16, actual_seq_q, npu_input_layout).npu()
            k_rope_npu = convert_qk_rope_bnsd_to_layout(kr_bf16, actual_seq_kv, npu_input_layout).npu()
            print(f"[NPU {npu_input_layout}] q_rope={q_rope_npu.shape}, k_rope={k_rope_npu.shape}")
        else:
            q_rope_npu = None
            k_rope_npu = None

        print(f"[NPU] 调用 {npu_input_layout} 模式...")
        npu_out = torch_npu.npu_fused_infer_attention_score_v2(
            q_npu, k_npu, v_npu,
            query_rope=q_rope_npu,
            key_rope=k_rope_npu,
            atten_mask=mask_arg,
            actual_seq_qlen=accum_seq_q,
            actual_seq_kvlen=accum_seq_kv,
            dequant_scale_query=deq_q_npu,
            dequant_scale_key=deq_k_npu,
            dequant_scale_value=deq_v_npu,
            num_query_heads=N_q,
            num_key_value_heads=N_kv,
            softmax_scale=softmax_scale,
            input_layout=npu_input_layout,
            sparse_mode=SPARSE_MODE,
            query_quant_mode=6,
            key_quant_mode=6,
            value_quant_mode=8,
            query_dtype=FP8_DTYPE,
            key_dtype=FP8_DTYPE,
            value_dtype=FP8_DTYPE,
            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
            quant_scale_p=p_scale_npu,
        )

    npu_output = npu_out[0]
    T_actual = sum(actual_seq_q)
    if npu_output.shape[0] > T_actual:
        npu_output = npu_output[:T_actual]
    print(f"[NPU] output={npu_output.shape}")
    return npu_output

# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MXFP8 Flash Attention Golden")
    print("输出: 逐元素表格 + 统计汇总 (PctRlt 通过率)")
    print("=" * 60)
    print(f"场景: {'PA' if ENABLE_PA else 'TND'}")
    print(f"INPUT_LAYOUT={INPUT_LAYOUT}, Q_SCALE_LAYOUT={Q_SCALE_LAYOUT}")
    print(f"KV_CACHE_LAYOUT={KV_CACHE_LAYOUT}")
    print(f"B={B}, N_q={N_q}, N_kv={N_kv}, D={D}")
    print(f"ACTUAL_SEQ_Q={ACTUAL_SEQ_Q}, ACTUAL_SEQ_KV={ACTUAL_SEQ_KV}")

    print("\n[Step 1] 数据生成")
    (q_fp8, k_fp8, v_fp8,
     dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
     qr_bf16, kr_bf16, block_table_torch) = generate_data()

    print("\n[Step 2] CPU Golden")
    cpu_out = cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8,
                               dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                               ACTUAL_SEQ_Q, ACTUAL_SEQ_KV,
                               qr_bf16, kr_bf16)

    print("\n[Step 3] NPU 调用")
    npu_out = npu_mxfp8_fa(q_fp8, k_fp8, v_fp8,
                           dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                           ACTUAL_SEQ_Q, ACTUAL_SEQ_KV,
                           block_table_torch, qr_bf16, kr_bf16)

    print("\n[Step 4] 精度对比")
    cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out, ACTUAL_SEQ_Q, "TND")
    result_compare_method.check_result(cpu_tnd_torch, npu_out)