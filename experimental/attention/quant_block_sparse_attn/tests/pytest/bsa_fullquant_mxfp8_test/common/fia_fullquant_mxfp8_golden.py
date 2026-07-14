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

功能：参考式生成输入 → CPU golden → NPU 调用 → layout 对齐 → 精度对比
支持：Q/QDescale 公共输入为 TND 或 NTD；K/V 生成后在本文件内部按 BNSD 保存，NPU 侧只支持 PA KV cache
支持：PA 场景、GQA、causal/dense sparse mode、QDescale 固定 TND D-group 打包
数据：随机 FP8 Q/K/V + MXFP8 D-group descale，生成结构对齐 quant_block_sparse_attn_golden.py
输出：逐元素表格 + 统计汇总 (PctRlt 通过率，双千分之五标准)

"""

import argparse
import logging
import math
import os
import random
import sys

import torch
import torch_npu

try:
    from . import result_compare_method
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QBSA_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
TORCH_OPS_EXTENSION_DIR = os.path.join(QBSA_ROOT, "torch_ops_extension")
_CUSTOM_OPS_IMPORTED = False


def _ensure_custom_ops_registered():
    """Import custom_ops before calling torch.ops.custom QBSA APIs."""
    global _CUSTOM_OPS_IMPORTED
    if _CUSTOM_OPS_IMPORTED:
        return

    try:
        import custom_ops  # noqa: F401
    except ImportError as install_import_error:
        sys.modules.pop("custom_ops", None)
        if TORCH_OPS_EXTENSION_DIR not in sys.path:
            sys.path.insert(0, TORCH_OPS_EXTENSION_DIR)
        try:
            import custom_ops  # noqa: F401
        except ImportError as source_import_error:
            raise RuntimeError(
                "import custom_ops failed. Please build/install QBSA PTA extension first, for example: "
                "`cd experimental/attention/quant_block_sparse_attn/torch_ops_extension && bash build_and_install.sh`. "
                f"Installed-package import error: {install_import_error}. "
                f"Source-tree import error: {source_import_error}."
            ) from source_import_error

    if not hasattr(torch.ops, "custom") or not hasattr(torch.ops.custom, "npu_quant_block_sparse_attn"):
        raise RuntimeError("torch.ops.custom.npu_quant_block_sparse_attn is not registered after importing custom_ops")
    if not hasattr(torch.ops.custom, "npu_quant_block_sparse_attn_metadata"):
        raise RuntimeError("torch.ops.custom.npu_quant_block_sparse_attn_metadata is not registered after importing custom_ops")
    _CUSTOM_OPS_IMPORTED = True


def _make_reference_style_mxfp8_case():
    """Single configuration entry for MXFP8 full-quant golden generation and execution."""
    return {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "D": 128,
        "actual_seq_q": [256],
        "actual_seq_kv": [512],
        "S1": 256,
        "S2": 512,
        "block_size": 128,
        "mask_mode": 3,
        "fp8_dtype": torch.float8_e4m3fn,
        "scale_dtype": torch.float8_e8m0fnu,
        "quant_group_size": 32,
        "layout_q": "TND",
        "p_scale_value": 1.0,
        "kv_cache_layout": "BnNBsD",
        "is_contiguous": True,
        "return_softmax_lse": False,
        "seed": 54,
        "data_range_q": 1.0,
        "data_range_k": 1.0,
        "data_range_v": 1.0,
        "device_id": 0,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
        "sparse_count": 4,
        "quant_mode": 2,
        "layout_kv": "PA_BNSD",
        "layout_sparse_indices": "B_N_Qb_Kb",
        "layout_out": "TND",
        "sparse_pattern": "sequential",
        "block_table_pattern": "sequential",
        "softmax_scale": 1.0 / math.sqrt(128.0),
    }

CASE = _make_reference_style_mxfp8_case()

# Backward-compatible aliases derived from CASE. Edit _make_reference_style_mxfp8_case() above for configuration.
B = CASE["B"]
N_q = CASE["N1"]
N_kv = CASE["N2"]
D = CASE["D"]
ACTUAL_SEQ_Q = CASE["actual_seq_q"]
ACTUAL_SEQ_KV = CASE["actual_seq_kv"]
BLOCK_SIZE = CASE["block_size"]
MASK_MODE = CASE["mask_mode"]
FP8_DTYPE = CASE["fp8_dtype"]
SCALE_DTYPE = CASE["scale_dtype"]
QUANT_GROUP_SIZE = CASE["quant_group_size"]
Q_INPUT_LAYOUT = CASE["layout_q"]
KV_CACHE_LAYOUT = CASE["kv_cache_layout"]
IS_CONTIGUOUS = CASE["is_contiguous"]
ENABLE_LSE = CASE["return_softmax_lse"]
DATA_RANGE_Q = CASE["data_range_q"]
DATA_RANGE_K = CASE["data_range_k"]
DATA_RANGE_V = CASE["data_range_v"]
DEVICE_ID = CASE["device_id"]
SPARSE_BLOCK_SIZE = CASE["sparse_q_block_size"]
QUANT_MODE_MXFP8 = CASE["quant_mode"]
KEY_LAYOUT = CASE["layout_kv"]
SPARSE_INDICES_LAYOUT = CASE["layout_sparse_indices"]
OUT_LAYOUT = CASE["layout_out"]

# ==============================================================================
# 固定运行常量
# ==============================================================================
# Q/K/V 使用 fp8_e4m3fn；Q/K/V descale 与 P scale 使用 fp8_e8m0。
# e8m0fnu 最小正数: 2^(-127)，用于替换 descale 中的 0 和非有限值
# e8m0fnu 没有 0 值语义，0 的 biased exponent 会被 NPU 解释为 NaN
E8M0_MIN_POSITIVE = 2**(-127)
_EMAX_MAP = {
    torch.float8_e4m3fn: 8,
}

# ==============================================================================
# 参考式数据生成 helper
# 与 quant_block_sparse_attn_golden.py 保持同一组织方式：
#   case -> rng/generator -> query/key/value/descale/block_table/sparse_indices
# 当前文件的 MXFP8 full-quant 差异：
#   - Q/QDescale 直接生成 TND 或 NTD，不经过 BNSD 中间格式
#   - K/V 先按参考文件的 BSND 语义生成，再适配为本文件 CPU/PA 路径使用的 BNSD
#   - Q/K/V descale 由对应 FP8 数据按 MXFP8 group 规则生成
# ==============================================================================

def _prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _rand_float(shape, generator, data_range=1.0):
    return (torch.rand(shape, dtype=torch.float32, generator=generator) * 2 - 1) * data_range


def _rand_fp8(shape, generator, data_range=1.0):
    return _rand_float(shape, generator, data_range).to(FP8_DTYPE)


def _validate_fp8_dtype(fp8_dtype):
    if fp8_dtype not in _EMAX_MAP:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}; only torch.float8_e4m3fn is supported")


def get_mxfp8_per_token_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Vectorized Q/K per-token-group quant_scale."""
    _validate_fp8_dtype(fp8_dtype)
    emax_elem = _EMAX_MAP[fp8_dtype]
    tensor = tensor.to(torch.float32)

    dim1, dim2, dim3, dim4 = tensor.shape
    num_groups = math.ceil(dim4 / group_size)
    pad_size = num_groups * group_size - dim4
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))

    grouped = tensor.reshape(dim1, dim2, dim3, num_groups, group_size)
    all_zero_mask = torch.all(grouped == 0, dim=-1)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
    return torch.where(all_zero_mask, torch.ones_like(shared_exp), 2 ** shared_exp).to(torch.float32)


def get_mxfp8_per_channel_group_quant_scale(tensor, fp8_dtype, group_size=32):
    """Vectorized V per-channel-group quant_scale for BSND value tensor."""
    _validate_fp8_dtype(fp8_dtype)
    emax_elem = _EMAX_MAP[fp8_dtype]
    tensor = tensor.to(torch.float32)

    batch, seq_len, num_heads, head_dim = tensor.shape
    num_groups = math.ceil(seq_len / group_size)
    pad_size = num_groups * group_size - seq_len
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad_size))

    grouped = tensor.reshape(batch, num_groups, group_size, num_heads, head_dim)
    all_zero_mask = torch.all(grouped == 0, dim=2)
    max_vals = torch.max(torch.abs(grouped), dim=2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
    return torch.where(all_zero_mask, torch.ones_like(shared_exp), 2 ** shared_exp).to(torch.float32)


def _physical_ids(start, count, pattern, rng):
    ids = list(range(start, start + count))
    if pattern == "reverse":
        ids.reverse()
    elif pattern == "random":
        rng.shuffle(ids)
    elif pattern != "sequential":
        raise ValueError(f"Unsupported block table pattern: {pattern}")
    return ids


def _make_reference_block_table(batch, seq_len, block_size, pattern, rng):
    block_num_per_batch = math.ceil(seq_len / block_size)
    block_table = torch.empty((batch, block_num_per_batch), dtype=torch.int32)
    for batch_idx in range(batch):
        physical_ids = _physical_ids(batch_idx * block_num_per_batch, block_num_per_batch, pattern, rng)
        for block_idx, physical_block in enumerate(physical_ids):
            block_table[batch_idx, block_idx] = int(physical_block)
    return block_table


def _allowed_blocks(mask_mode, qb_idx, sparse_q_block_size, sparse_kv_block_size, q_len, kv_len):
    block_num = math.ceil(kv_len / sparse_kv_block_size)
    if block_num <= 0:
        return []
    if mask_mode == 0:
        return list(range(block_num))
    if mask_mode != 3:
        raise ValueError(f"Unsupported sparse mode: {mask_mode}")

    max_token = (qb_idx + 1) * sparse_q_block_size - 1 + kv_len - q_len
    if max_token < 0:
        return []
    max_block = min(block_num - 1, max_token // sparse_kv_block_size)
    return list(range(max_block + 1))


def _select_blocks(blocks, sparse_count, pattern, rng):
    if sparse_count <= 0 or not blocks or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal"):
        return blocks[:min(sparse_count, len(blocks))]
    if pattern == "reverse":
        return list(reversed(blocks[-min(sparse_count, len(blocks)):]))
    if pattern == "tail":
        selected = blocks[:max(0, min(sparse_count, len(blocks)) - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:sparse_count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[:min(sparse_count, len(selected))]
    raise ValueError(f"Unsupported sparse pattern: {pattern}")


def _make_reference_sparse_indices(case, q_lengths, kv_lengths, rng):
    batch = case["B"]
    n1 = case["N1"]
    qb_max = math.ceil(case["S1"] / case["sparse_q_block_size"])
    kv_max = math.ceil(case["S2"] / case["sparse_kv_block_size"])
    sparse_indices = torch.full((batch, n1, qb_max, kv_max), fill_value=-1, dtype=torch.int32)
    sparse_seq_len = torch.zeros((batch, n1, qb_max), dtype=torch.int32)

    for batch_idx in range(batch):
        qb_batch = math.ceil(q_lengths[batch_idx] / case["sparse_q_block_size"])
        for qb_idx in range(qb_max):
            allowed = _allowed_blocks(
                case["mask_mode"], qb_idx, case["sparse_q_block_size"], case["sparse_kv_block_size"],
                q_lengths[batch_idx], kv_lengths[batch_idx])
            for head_idx in range(n1):
                if qb_idx >= qb_batch:
                    selected = []
                else:
                    selected = _select_blocks(allowed, case["sparse_count"], case["sparse_pattern"], rng)
                sparse_seq_len[batch_idx, head_idx, qb_idx] = len(selected)
                if selected:
                    sparse_indices[batch_idx, head_idx, qb_idx, :len(selected)] = torch.tensor(
                        selected, dtype=torch.int32)
    return sparse_indices, sparse_seq_len



def _make_reference_sparse_for_lengths(actual_seq_q, actual_seq_kv):
    case = dict(CASE)
    case["S1"] = max(int(item) for item in actual_seq_q)
    case["S2"] = max(int(item) for item in actual_seq_kv)
    case["sparse_count"] = math.ceil(case["S2"] / case["sparse_kv_block_size"])
    return _make_reference_sparse_indices(
        case, [int(item) for item in actual_seq_q], [int(item) for item in actual_seq_kv],
        random.Random(case["seed"]))



# ==============================================================================
# Layout 转换函数 - 数据
# Q/QDescale 公共输入只接受 TND/NTD；BNSD 仅作为 K/V 和 PA KV cache 的内部布局。
# ==============================================================================

def canonical_q_input_layout(layout=None):
    layout = (layout or Q_INPUT_LAYOUT).upper()
    if layout not in ("TND", "NTD"):
        raise ValueError(f"Unsupported Q input layout: {layout}, expected TND or NTD")
    return layout


def convert_q_tnd_or_ntd_to_tnd(tensor, seq_lens, num_heads, name="Q"):
    """Q/QDescale input must be 3D TND or NTD; normalize to TND."""
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be TND/NTD rank-3 input, got rank {tensor.dim()} and shape {tuple(tensor.shape)}")

    total_s = sum(seq_lens)
    if tensor.shape[0] == total_s and tensor.shape[1] == num_heads:
        return tensor.contiguous()
    if tensor.shape[0] == num_heads and tensor.shape[1] == total_s:
        return tensor.permute(1, 0, 2).contiguous()

    raise ValueError(
        f"Unsupported {name} shape: {tuple(tensor.shape)}, expected TND=({total_s}, {num_heads}, D) "
        f"or NTD=({num_heads}, {total_s}, D)")


def convert_q_tnd_or_ntd_to_layout(tensor, seq_lens, num_heads, layout=None, name="Q"):
    layout = canonical_q_input_layout(layout)
    tensor_tnd = convert_q_tnd_or_ntd_to_tnd(tensor, seq_lens, num_heads, name)
    if layout == "TND":
        return tensor_tnd
    return tensor_tnd.permute(1, 0, 2).contiguous()



# ==============================================================================
# Layout 转换函数 - Descale
# QDescale 公共输入为 TND/NTD；K/V descale 在生成后以内部 BNSD 保存并按 NPU layout 打包。
# ==============================================================================

def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 → e8m0fnu (uint8)，提取 IEEE 754 biased exponent
    e8m0fnu 格式: 只有指数位，没有尾数，表示 2^(e-127)
    biased exponent = 0xFF 时表示 NaN
    """
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0fnu 没有 0 值语义；非有限值进入 0xFF 会在 NPU 侧变 NaN。"""
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad_mask = ~torch.isfinite(result)
    zero_mask = result == 0
    bad_count = int(bad_mask.sum().item())
    zero_count = int(zero_mask.sum().item())
    if bad_count:
        logger.info("[WARN] %s: replace %d non-finite scale values before e8m0 packing", name, bad_count)
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
    return packed.view(SCALE_DTYPE)



def pack_qk_scale_for_npu(scale_flat):
    """Pack Q/K D-group descale: (..., Dg) -> (..., Dg//2, 2).
    NPU 要求 Q/K descale 按 (偶, 奇) 对打包，每两个相邻 scale 值合并为一个 [..., 2]
    """
    orig_shape = scale_flat.shape
    last_dim = orig_shape[-1]
    new_shape = orig_shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)



def convert_q_scale_tnd_or_ntd_to_layout(scale, seq_lens):
    """QDescale public input TND/NTD -> packed NPU TND descale layout."""
    scale_tnd = convert_q_tnd_or_ntd_to_tnd(scale, seq_lens, N_q, "Q descale")
    return pack_qk_scale_for_npu(scale_tnd)


# ==============================================================================
# PA 格式转换 - mxfp8_pa_preprocessing
# 仅处理 K/V 及 K/V descale；Q/QDescale 不走 PA 预处理。
# ==============================================================================

def mxfp8_pa_preprocessing(tensor_bnsd, seq_lens, block_size, block_table,
                           is_vscale=False, is_scale=False,
                           kv_layout="BnNBsD", group_size=32, sparse_indices=None, sparse_seq_len=None):
    """
    MXFP8 PA 预处理: internal K/V BNSD -> PagedAttention KV Cache

    输入: [B, N, S, D]，仅用于 K/V 或 K/V descale
    输出 (is_scale=False): [BlockNum, N, BlockSize, D] (fp8 K/V)
    输出 (is_scale=True, is_vscale=False): [BlockNum, N, BlockSize, D//64, 2] (K descale)
    输出 (is_scale=True, is_vscale=True): [BlockNum, N, BlockSize//64, D, 2] (V descale)

    kv_layout:
      - BnNBsD: fp8=[Bn,N,Bs,D], KDescale=[Bn,N,Bs,D//64,2], VDescale=[Bn,N,Bs//64,D,2]
      - BnBsND: fp8=[Bn,Bs,N,D], KDescale=[Bn,Bs,N,D//64,2], VDescale=[Bn,Bs//64,N,D,2]
    """
    tensor_bnsd = tensor_bnsd if isinstance(tensor_bnsd, torch.Tensor) else torch.as_tensor(tensor_bnsd)
    block_table = block_table if isinstance(block_table, torch.Tensor) else torch.as_tensor(block_table)
    B, N, S, D = tensor_bnsd.shape

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_descale_pack_ratio = group_size * 2
            pack_seq_lens = [math.ceil(act_s / v_descale_pack_ratio) for act_s in seq_lens]
            pack_block_size = math.ceil(block_size / v_descale_pack_ratio)
            total_block_num = sum(math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens)
            out_shape = (total_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K descale Dg must be even for pair packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (sum(math.ceil(act_s / block_size) for act_s in seq_lens), N, block_size, D // 2, 2)
    else:
        tensor_processed = tensor_bnsd
        pack_seq_lens = seq_lens
        pack_block_size = block_size
        out_shape = (sum(math.ceil(act_s / block_size) for act_s in seq_lens), N, block_size, D)

    fill_value = E8M0_MIN_POSITIVE if is_scale else 0

    out_cache = torch.full(out_shape, fill_value, dtype=tensor_processed.dtype, device=tensor_processed.device)
    block_num = [math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        #for n_q_idx in range(N_q):
        for blk_idx in range(block_num[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * pack_block_size
            #block_offset = sparse_indices[b,n_q_idx,blk_idx] * pack_block_size
            valid_len = min(pack_block_size, pack_seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[blockid, :, :valid_len] = tensor_processed[b, :, block_offset:block_offset + valid_len]

    if kv_layout == "BnNBsD":
        return out_cache
    elif kv_layout == "BnBsND":
        return out_cache.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def convert_v_scale_to_pa(scale_bnsd, seq_lens, group_size=32):
    """
    V descale 偶奇行交错 packing，用于 PA 预处理
    输入: [B, N, Sg, D], Sg = ceil(orgS/32)
    输出: [B, N, Sg//2, D, 2]，奇数行 pad 到偶数
    """
    scale_bnsd = scale_bnsd if isinstance(scale_bnsd, torch.Tensor) else torch.as_tensor(scale_bnsd)
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)

    transposed = scale_bnsd[:, :, :actual_Sg, :]

    if actual_Sg % 2 != 0:
        pad = torch.full((B, N, 1, D), E8M0_MIN_POSITIVE, dtype=transposed.dtype, device=transposed.device)
        transposed = torch.cat([transposed, pad], dim=2)
        actual_Sg += 1

    S_out = actual_Sg // 2
    result = torch.zeros((B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device)
    result[..., 0] = transposed[..., ::2, :]
    result[..., 1] = transposed[..., 1::2, :]
    return result



# ==============================================================================
# 数据生成
# ==============================================================================

def _generate_reference_style_mxfp8_inputs():
    """Generate test inputs using the same data-generation structure as quant_block_sparse_attn_golden.py."""
    case = dict(CASE)
    rng = random.Random(case["seed"])
    generator = torch.Generator().manual_seed(case["seed"])

    q_lengths = [int(item) for item in ACTUAL_SEQ_Q]
    kv_lengths = [int(item) for item in ACTUAL_SEQ_KV]
    cu_seqlens_q = _prefix(q_lengths)
    cu_seqlens_kv = _prefix(kv_lengths)
    total_q = int(cu_seqlens_q[-1].item())

    batch = case["B"]
    s2 = case["S2"]
    n1 = case["N1"]
    n2 = case["N2"]
    head_dim = case["D"]
    layout_q = case["layout_q"]

    if layout_q == "NTD":
        query = _rand_fp8((n1, total_q, head_dim), generator, DATA_RANGE_Q)
        q_descale = get_mxfp8_per_token_group_quant_scale(
            query.unsqueeze(0), FP8_DTYPE, QUANT_GROUP_SIZE).squeeze(0)
    else:
        query = _rand_fp8((total_q, n1, head_dim), generator, DATA_RANGE_Q)
        q_descale = get_mxfp8_per_token_group_quant_scale(
            query.unsqueeze(0), FP8_DTYPE, QUANT_GROUP_SIZE).squeeze(0)

    dense_key = _rand_fp8((batch, s2, n2, head_dim), generator, DATA_RANGE_K)
    dense_value = _rand_fp8((batch, s2, n2, head_dim), generator, DATA_RANGE_V)
    dense_k_descale = get_mxfp8_per_token_group_quant_scale(
        dense_key, FP8_DTYPE, QUANT_GROUP_SIZE)
    dense_v_descale = get_mxfp8_per_channel_group_quant_scale(
        dense_value, FP8_DTYPE, QUANT_GROUP_SIZE)

    block_table = _make_reference_block_table(
        batch, s2, BLOCK_SIZE, case["block_table_pattern"], rng)

    sparse_indices, sparse_seq_len = _make_reference_sparse_indices(case, q_lengths, kv_lengths, rng)
    p_scale = torch.tensor([float(case["p_scale_value"])], dtype=torch.float32)

    return {
        "case": case,
        "query": query,
        "key": dense_key.permute(0, 2, 1, 3).contiguous(),
        "value": dense_value.permute(0, 2, 1, 3).contiguous(),
        "q_descale": q_descale,
        "k_descale": dense_k_descale.permute(0, 2, 1, 3).contiguous(),
        "v_descale": dense_v_descale.permute(0, 2, 1, 3).contiguous(),
        "p_scale": p_scale,
        "block_table": block_table,
        "sparse_indices": sparse_indices,
        "sparse_seq_len": sparse_seq_len,
        "q_lengths": q_lengths,
        "kv_lengths": kv_lengths,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
    }


def generate_data():
    """Generate inputs through the reference-style data-generation wrapper."""
    data = _generate_reference_style_mxfp8_inputs()
    logger.info(
        "[INFO] reference-style data: q=%s, k=%s, v=%s, q_descale=%s, k_descale=%s, v_descale=%s",
        data["query"].shape, data["key"].shape, data["value"].shape,
        data["q_descale"].shape, data["k_descale"].shape, data["v_descale"].shape)
    logger.info("[INFO] sparse_indices=%s, sparse_seq_len=%s",
                data["sparse_indices"].shape, data["sparse_seq_len"].shape)

    return (data["query"], data["key"], data["value"],
            data["q_descale"], data["k_descale"], data["v_descale"], data["p_scale"],
            data["block_table"], data["sparse_indices"], data["sparse_seq_len"],
            data["q_lengths"], data["kv_lengths"], data["cu_seqlens_q"], data["cu_seqlens_kv"])


# ==============================================================================
# CPU Golden
# 参考 quant_block_sparse_attn_golden.py 的 sparse block 计算流：
#   - Q/QDescale 公共输入为 TND/NTD；K/V 与 K/V descale 使用内部 BNSD
#   - 按 sparse_indices 中记录的 KV block 顺序收集 positions，并以两个 sparse block 为一组做 online 累加
#   - Q/K 使用 per-token D-group descale；V 使用 per-channel S-group descale
#   - 输出 OUT 固定 TND，LSE 固定 NT1
# ==============================================================================

EMPTY_LSE = -3.4028234663852886e38
MASK_VALUE = -10000.0


def _positions_from_sparse(sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len):
    block_count = int(sparse_seq_len[batch_idx, head_idx, qb_idx].item())
    chunk_positions = []
    cursor = 0
    while cursor < block_count:
        pair_positions = []
        for offset in range(2):
            if cursor + offset >= block_count:
                continue
            block_idx = int(sparse_indices[batch_idx, head_idx, qb_idx, cursor + offset].item())
            if block_idx < 0:
                continue
            start = block_idx * SPARSE_BLOCK_SIZE
            end = min(start + SPARSE_BLOCK_SIZE, kv_len)
            if start < kv_len:
                pair_positions.extend(range(start, end))
        if pair_positions:
            chunk_positions.append(pair_positions)
        cursor += 2
    return chunk_positions


def _expand_d_group_scale(scale, width):
    return scale.to(torch.float32).repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[..., :width]


def _gather_q_block_and_scale(q_tensor, q_scale, layout_q, cu_seqlens_q, batch_idx, q_start, q_end, head_idx):
    if layout_q == "NTD":
        base = int(cu_seqlens_q[batch_idx].item())
        q_block = q_tensor[head_idx, base + q_start:base + q_end]
        q_scale_block = q_scale[head_idx, base + q_start:base + q_end]
    else:
        base = int(cu_seqlens_q[batch_idx].item())
        q_block = q_tensor[base + q_start:base + q_end, head_idx]
        q_scale_block = q_scale[base + q_start:base + q_end, head_idx]
    return q_block.to(torch.float32), q_scale_block.to(torch.float32)


def _valid_mask_for_positions(q_indices, positions, q_len, kv_len):
    nq = len(q_indices)
    npos = len(positions)
    if MASK_MODE == 0:
        return torch.ones((nq, npos), dtype=torch.bool)
    if MASK_MODE != 3:
        raise ValueError(f"Unsupported mask_mode: {MASK_MODE}")
    q_idx_col = torch.as_tensor(q_indices, dtype=torch.long).view(nq, 1)
    pos_tensor = torch.as_tensor(positions, dtype=torch.long).view(1, npos)
    return pos_tensor <= (q_idx_col + kv_len - q_len)


def cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8,
                     dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                     q_lengths, kv_lengths, cu_seqlens_q, sparse_indices, sparse_seq_len):
    """Reference-style CPU golden for MXFP8 block sparse attention."""
    layout_q = canonical_q_input_layout(Q_INPUT_LAYOUT)
    q_tensor = q_fp8
    q_scale = dequant_scale_q
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)
    k_scale = dequant_scale_k
    v_scale = dequant_scale_v

    total_q = int(cu_seqlens_q[-1].item())
    batch = len(q_lengths)
    n1 = N_q
    n2 = N_kv
    group = n1 // n2
    head_dim = D
    softmax_scale = float(CASE["softmax_scale"])
    p_scale_value = float(torch.as_tensor(p_scale).reshape(-1)[0].item())

    attention_out = torch.zeros((total_q, n1, head_dim), dtype=torch.float32)
    softmax_lse = torch.full((n1, total_q, 1), EMPTY_LSE, dtype=torch.float32)

    qb_max = math.ceil(max(q_lengths) / SPARSE_BLOCK_SIZE)
    neg_inf = float("-inf")

    logger.info("[CPU Golden] reference sparse flow: layout_q=%s, OUT=TND, LSE=NT1", layout_q)

    for batch_idx in range(batch):
        q_len = int(q_lengths[batch_idx])
        kv_len = int(kv_lengths[batch_idx])
        q_base = int(cu_seqlens_q[batch_idx].item())
        for head_idx in range(n1):
            n2_idx = head_idx // group
            for qb_idx in range(qb_max):
                q_start = qb_idx * SPARSE_BLOCK_SIZE
                if q_start >= q_len:
                    break
                q_end = min(q_start + SPARSE_BLOCK_SIZE, q_len)
                q_indices = list(range(q_start, q_end))
                nq = q_end - q_start

                # chunk_positions 对齐 kernel 侧一次 C1V1 的 K 读取粒度：
                #   - sparse_indices 记录的是当前 QB 可访问的 KV basic block id。
                #   - _positions_from_sparse 每次取 2 个 KV basic block，并展开成 token 级 KV index。
                #   - 因此 chunk_positions[i] 对应一次 C1V1 中 K 的 token 范围，最多 2 * SPARSE_BLOCK_SIZE。
                #   - 无效 block id(-1) 或超出 kv_len 的 block 会被跳过，所以 chunk 可能不足两个 basic block。
                chunk_positions = _positions_from_sparse(
                    sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len)
                if not chunk_positions:
                    continue

                # positions 是当前 QB 在 sparse 规则下候选的全部 KV token index。
                # 注意这里还没有应用 causal/padding mask；逐 Q token 的实际有效性由 valid_mask 决定。
                positions = [pos for chunk in chunk_positions for pos in chunk]
                pos_tensor = torch.as_tensor(positions, dtype=torch.long)
                npos = int(pos_tensor.numel())

                q_block, q_scale_block = _gather_q_block_and_scale(
                    q_tensor, q_scale, layout_q, cu_seqlens_q, batch_idx, q_start, q_end, head_idx)
                q_dequant = q_block * _expand_d_group_scale(q_scale_block, head_dim)

                k_mat = k_tensor[batch_idx, n2_idx, pos_tensor, :]
                k_scale_mat = k_scale[batch_idx, n2_idx, pos_tensor, :]
                k_dequant = k_mat * _expand_d_group_scale(k_scale_mat, head_dim)

                valid_mask = _valid_mask_for_positions(q_indices, positions, q_len, kv_len)
                scores = torch.matmul(q_dequant, k_dequant.transpose(0, 1)) * softmax_scale
                scores = torch.where(valid_mask, scores, torch.full_like(scores, MASK_VALUE))

                m_run = torch.full((nq,), neg_inf, dtype=torch.float32)
                l_run = torch.zeros((nq,), dtype=torch.float32)
                acc = torch.zeros((nq, head_dim), dtype=torch.float32)
                # 把每个 C1V1 chunk 在展平 positions/scores 中的切片范围记录下来。
                # 后续每轮取连续两个 chunk，模拟 kernel 的 c1v1 + c1v1 -> c2v2 流水。
                chunk_offsets = []
                offset = 0
                for chunk_pos_list in chunk_positions:
                    next_offset = offset + len(chunk_pos_list)
                    chunk_offsets.append((offset, next_offset))
                    offset = next_offset

                for round_idx in range(0, len(chunk_offsets), 2):
                    # 一轮 kernel 流水：
                    #   C1V1-0: Q(最多 128) x K(最多 2 个 128-token basic block)
                    #   C1V1-1: Q(最多 128) x K(最多 2 个 128-token basic block)
                    #   C2V2:   P(最多 128 x 512) x V(最多 512 x D)
                    # 如果最后一轮只剩一个 C1V1 chunk，round_end 只覆盖实际存在的 KV token。
                    round_start = chunk_offsets[round_idx][0]
                    round_end = chunk_offsets[min(round_idx + 1, len(chunk_offsets) - 1)][1]

                    # s_round/vm_round 是本轮最多 4 个 KV basic block 对应的 score 和有效 mask。
                    # round_max 必须基于整轮两个 C1V1 的 score 共同计算，之后 C2V2 使用同一个 m_new。
                    s_round = scores[:, round_start:round_end]
                    vm_round = valid_mask[:, round_start:round_end]
                    masked_scores = torch.where(vm_round, s_round, torch.full_like(s_round, neg_inf))
                    round_max = masked_scores.max(dim=-1).values
                    round_has = vm_round.any(dim=-1)

                    # online softmax 状态：m_run/l_run/acc 分别是历史 KV round 的 max、sum、PV 累加。
                    # 若当前行此前已经有有效 KV，则 m_new=max(m_run, round_max)；否则从本轮 round_max 启动。
                    # 如果本轮对某个 Q 行没有任何有效 KV，保持该行状态不变。
                    run_started = m_run != neg_inf
                    m_new = torch.where(run_started, torch.maximum(m_run, round_max), round_max)
                    m_new = torch.where(round_has, m_new, m_run)

                    # 历史 acc/l_run 需要按 exp(m_old - m_new) 重标定到新的 max 基准。
                    # 对从未启动过的行，rescale=0，避免 -inf 参与 exp 产生 NaN。
                    rescale = torch.where(run_started, torch.exp(m_run - m_new), torch.zeros_like(m_run))
                    rescale = torch.where(torch.isfinite(rescale), rescale, torch.zeros_like(rescale))

                    # 本轮 P 基于同一个 m_new 计算；无效 token 置 0。
                    # p_quant 仅用于 PV，l_run 仍累加未量化的 p_round，和参考 online softmax 数据流一致。
                    p_round = torch.exp(s_round - m_new.view(nq, 1))
                    p_round = torch.where(vm_round, p_round, torch.zeros_like(p_round))
                    p_round = torch.where(round_has.view(nq, 1), p_round, torch.zeros_like(p_round))
                    p_quant = (p_round * p_scale_value).to(FP8_DTYPE).to(torch.float32)

                    # C2V2 阶段读取与本轮 K token 完全对应的 V token。
                    # round_pos_tensor 覆盖最多 4 个 basic block；若 sparse/mask 导致不足 4 块，只处理实际 token。
                    round_pos_tensor = pos_tensor[round_start:round_end]
                    v_mat = v_tensor[batch_idx, n2_idx, round_pos_tensor, :]
                    v_group_idx = torch.div(round_pos_tensor, QUANT_GROUP_SIZE, rounding_mode="floor")
                    v_scale_mat = v_scale[batch_idx, n2_idx, v_group_idx, :].to(torch.float32)
                    v_dequant = v_mat * v_scale_mat

                    # 完成本轮 C2V2，并把历史 online 状态推进到 m_new 基准。
                    pv = torch.matmul(p_quant, v_dequant) / p_scale_value
                    acc = acc * rescale.view(nq, 1) + pv
                    l_run = l_run * rescale + p_round.sum(dim=-1)
                    m_run = torch.where(round_has, m_new, m_run)

                any_valid = valid_mask.any(dim=-1)
                safe_l = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                attn = acc / safe_l.view(nq, 1)
                lse_scores = torch.where(valid_mask, scores, torch.full_like(scores, neg_inf))
                lse = torch.logsumexp(lse_scores, dim=-1)

                for local_idx in range(nq):
                    if not bool(any_valid[local_idx].item()):
                        continue
                    out_idx = q_base + q_start + local_idx
                    attention_out[out_idx, head_idx] = attn[local_idx]
                    softmax_lse[head_idx, out_idx, 0] = lse[local_idx]

    logger.info("[CPU Golden] output(TND)=%s, lse(NT1)=%s", attention_out.shape, softmax_lse.shape)
    return attention_out.contiguous(), softmax_lse.contiguous()


# ==============================================================================
# NPU 调用
# 只支持单算子模式。
# Q: 使用 Q_INPUT_LAYOUT；QDescale: 固定按 TND 打包 D-group pair。
# K/V 与 K/V descale: 固定走 PA 预处理 (block_table + block_size)。
# ==============================================================================

def _to_npu(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.npu() if tensor.device.type == "cpu" else tensor
    return torch.as_tensor(tensor).npu()



def _call_npu_fa_op(q, k, v, mask, q_lengths, kv_lengths, cu_seqlens_q, cu_seqlens_kv,
                    dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                    block_table, q_n, kv_n, softmax_scale, layout_q, layout_kv,
                    sparse_indices, sparse_seq_len):
    """调用 QuantBlockSparseAttn 单算子，入参对齐 custom op schema。"""
    _ensure_custom_ops_registered()
    torch_npu.npu.set_device(int(DEVICE_ID))

    seqused_q = torch.tensor(q_lengths, dtype=torch.int32)
    seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)
    cu_seqlens_q = _to_npu(cu_seqlens_q)
    cu_seqlens_kv = _to_npu(cu_seqlens_kv)
    seqused_q = seqused_q.npu()
    seqused_kv = seqused_kv.npu()
    sparse_indices = _to_npu(sparse_indices)
    sparse_seq_len = _to_npu(sparse_seq_len)
    block_table = _to_npu(block_table)

    max_sq = max(q_lengths)
    max_skv = max(kv_lengths)

    torch_npu.npu.synchronize()
    metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        sparse_seq_len,
        q_n,
        kv_n,
        D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=B,
        sparse_block_size_q=SPARSE_BLOCK_SIZE,
        sparse_block_size_k=SPARSE_BLOCK_SIZE,
        quant_mode=QUANT_MODE_MXFP8,
        mask_mode=MASK_MODE,
        max_seqlen_q=max_sq,
        max_seqlen_kv=max_skv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_indices=SPARSE_INDICES_LAYOUT,
    )
    torch_npu.npu.synchronize()

    output = torch.ops.custom.npu_quant_block_sparse_attn(
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        mask,
        softmax_scale,
        SPARSE_BLOCK_SIZE,
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        metadata=metadata,
        max_seqlen_q=max_sq,
        max_seqlen_kv=max_skv,
        pa_block_stride=0,
        layout_kv=layout_kv,
        layout_q=layout_q,
        layout_sparse_indices=SPARSE_INDICES_LAYOUT,
        layout_out=OUT_LAYOUT,
        quant_mode=QUANT_MODE_MXFP8,
        mask_mode=MASK_MODE,
        return_softmax_lse=ENABLE_LSE,
    )
    torch_npu.npu.synchronize()
    atten_out, lse_out = output
    return atten_out.detach().cpu(), lse_out.detach().cpu()


def _build_causal_mask():
    if MASK_MODE == 0:
        return torch.empty((0,), dtype=torch.bool).npu()
    return torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()


def npu_mxfp8_fa(q_fp8, k_fp8, v_fp8,
                 dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                 q_lengths, kv_lengths, cu_seqlens_q, cu_seqlens_kv,
                 block_table_torch=None,
                 sparse_indices=None, sparse_seq_len=None):
    """调用 QuantBlockSparseAttn NPU 算子。"""
    torch_npu.npu.set_device(int(DEVICE_ID))

    if sparse_indices is None or sparse_seq_len is None:
        sparse_indices, sparse_seq_len = _make_reference_sparse_for_lengths(q_lengths, kv_lengths)

    softmax_scale = 1.0 / math.sqrt(D)

    q_layout = canonical_q_input_layout(Q_INPUT_LAYOUT)
    q_npu = convert_q_tnd_or_ntd_to_layout(q_fp8, q_lengths, N_q, q_layout, "Q").contiguous().view(FP8_DTYPE).npu()
    logger.info("[NPU %s] q=%s", q_layout, q_npu.shape)

    q_scale_e8m0 = fp32_to_e8m0fnu_safe(
        convert_q_scale_tnd_or_ntd_to_layout(dequant_scale_q, q_lengths), "Q descale")
    deq_q_npu = q_scale_e8m0.npu()
    logger.info("[NPU] Q descale layout=TND, shape=%s", q_scale_e8m0.shape)

    p_scale_e8m0 = fp32_to_e8m0fnu_safe(p_scale, "P scale")
    p_scale_npu = p_scale_e8m0.npu()
    logger.info("[NPU] P scale dtype=%s, shape=%s", p_scale_e8m0.dtype, p_scale_e8m0.shape)
    mask_arg = _build_causal_mask()

    if block_table_torch is None:
        raise ValueError("PA KV cache requires block_table_torch")

    k_pa = mxfp8_pa_preprocessing(
        k_fp8, kv_lengths, BLOCK_SIZE, block_table_torch,
        is_vscale=False, is_scale=False,
        kv_layout=KV_CACHE_LAYOUT, group_size=32,
        sparse_indices=sparse_indices, sparse_seq_len=sparse_seq_len)
    v_pa = mxfp8_pa_preprocessing(
        v_fp8, kv_lengths, BLOCK_SIZE, block_table_torch,
        is_vscale=False, is_scale=False,
        kv_layout=KV_CACHE_LAYOUT, group_size=32,
        sparse_indices=sparse_indices, sparse_seq_len=sparse_seq_len)
    k_npu = k_pa.contiguous().view(FP8_DTYPE).npu()
    v_npu = v_pa.contiguous().view(FP8_DTYPE).npu()
    if not IS_CONTIGUOUS:
        kv_cache = torch.stack([k_pa, v_pa], dim=2).npu()
        k_npu = kv_cache[:, :, 0]
        v_npu = kv_cache[:, :, 1]
        logger.info("[NPU] key is_contiguous=%s, value is_contiguous=%s", k_npu.is_contiguous(), v_npu.is_contiguous())

    k_scale_pa = mxfp8_pa_preprocessing(
        dequant_scale_k, kv_lengths, BLOCK_SIZE, block_table_torch,
        is_vscale=False, is_scale=True,
        kv_layout=KV_CACHE_LAYOUT, group_size=32,
        sparse_indices=sparse_indices, sparse_seq_len=sparse_seq_len)
    v_scale_pa = mxfp8_pa_preprocessing(
        dequant_scale_v, kv_lengths, BLOCK_SIZE, block_table_torch,
        is_vscale=True, is_scale=True,
        kv_layout=KV_CACHE_LAYOUT, group_size=32,
        sparse_indices=sparse_indices, sparse_seq_len=sparse_seq_len)

    deq_k_npu = fp32_to_e8m0fnu_safe(k_scale_pa, "K PA descale").npu()
    deq_v_npu = fp32_to_e8m0fnu_safe(v_scale_pa, "V PA descale").npu()
    if not IS_CONTIGUOUS:
        fake_kscale_tensor = torch.ones_like(deq_k_npu)
        fake_vscale_tensor = torch.ones_like(deq_v_npu)
        deq_k_npu = torch.stack([deq_k_npu, fake_kscale_tensor], dim=2)[:, :, 0]
        deq_v_npu = torch.stack([deq_v_npu, fake_vscale_tensor], dim=2)[:, :, 0]
        logger.info("[NPU] deq_k_descale is_contiguous=%s, deq_v_descale is_contiguous=%s",
                    deq_k_npu.is_contiguous(), deq_v_npu.is_contiguous())

    logger.info("[NPU PA] kv_layout=%s", KV_CACHE_LAYOUT)
    logger.info("[NPU PA] k=%s, v=%s", k_npu.shape, v_npu.shape)
    logger.info("[NPU PA] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)

    block_table_npu = block_table_torch.npu() if isinstance(block_table_torch, torch.Tensor) else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
    layout_q = q_layout
    layout_kv = KEY_LAYOUT

    logger.info("[NPU] 调用 QuantBlockSparseAttn 单算子, layout_q=%s, layout_kv=%s", layout_q, layout_kv)
    atten_out, lse_out = _call_npu_fa_op(
        q_npu, k_npu, v_npu, mask_arg, q_lengths, kv_lengths, cu_seqlens_q, cu_seqlens_kv,
        deq_q_npu, deq_k_npu, deq_v_npu, p_scale_npu, block_table_npu,
        N_q, N_kv, softmax_scale, layout_q, layout_kv, sparse_indices, sparse_seq_len)

    npu_output = atten_out
    npu_lse = lse_out
    T_actual = sum(q_lengths)
    if npu_output.shape[0] > T_actual:
        npu_output = npu_output[:T_actual]
    if npu_lse is not None and npu_lse.dim() >= 2 and npu_lse.shape[1] > T_actual:
        npu_lse = npu_lse[:, :T_actual, ...]
    logger.info("[NPU] output=%s, lse=%s", npu_output.shape, None if npu_lse is None else npu_lse.shape)
    return npu_output, npu_lse


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="MXFP8 Flash Attention Golden")
    parser.add_argument("--mode", default="all",
                        help="执行模式，支持逗号组合: all/gen/cpu/npu/compare. 例: --mode=npu,compare")
    parser.add_argument("--case-name", default="default",
                        help="case 名称，用于 .pt 文件命名")
    parser.add_argument("--cache-dir", default=None,
                        help="缓存目录路径（默认 golden_cache/）")
    args = parser.parse_args()

    raw_parts = {m.strip() for m in args.mode.split(",") if m.strip()}
    invalid = raw_parts - _VALID_MODES
    if invalid:
        parser.error(f"Invalid mode: {invalid}. Valid: {_VALID_MODES}")
    mode = {"gen", "cpu", "npu", "compare"} if "all" in raw_parts else raw_parts

    case_name = args.case_name
    cdir = args.cache_dir

    logger.info("=" * 60)
    logger.info("MXFP8 Flash Attention Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("输出: 逐元素表格 + 统计汇总 (PctRlt 通过率)")
    logger.info("=" * 60)
    logger.info("场景: PA")
    logger.info("Q_INPUT_LAYOUT=%s, QDescale layout=TND", Q_INPUT_LAYOUT)
    logger.info("KV_CACHE_LAYOUT=%s", KV_CACHE_LAYOUT)
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d", B, N_q, N_kv, D)
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)

    if "gen" in mode:
        logger.info("\n[Step 1] 数据生成")
        (q_fp8, k_fp8, v_fp8,
         dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
         block_table_torch, sparse_indices, sparse_seq_len,
         q_lengths, kv_lengths, cu_seqlens_q, cu_seqlens_kv) = generate_data()
        golden_cache.save_input(case_name, {
            "q_fp8": q_fp8,
            "k_fp8": k_fp8,
            "v_fp8": v_fp8,
            "dequant_scale_q": dequant_scale_q,
            "dequant_scale_k": dequant_scale_k,
            "dequant_scale_v": dequant_scale_v,
            "p_scale": p_scale,
            "block_table_torch": block_table_torch,
            "sparse_indices": sparse_indices,
            "sparse_seq_len": sparse_seq_len,
            "q_lengths": q_lengths,
            "kv_lengths": kv_lengths,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
        }, cache_dir=cdir)
    else:
        logger.info("\n[Step 1] 加载已保存的输入数据")
        input_path = golden_cache._path(case_name, "input", cdir)
        data = torch.load(input_path, weights_only=False)
        q_fp8 = data["q_fp8"]
        k_fp8 = data["k_fp8"]
        v_fp8 = data["v_fp8"]
        dequant_scale_q = data["dequant_scale_q"]
        dequant_scale_k = data["dequant_scale_k"]
        dequant_scale_v = data["dequant_scale_v"]
        p_scale = data["p_scale"]
        block_table_torch = data.get("block_table_torch")
        sparse_indices = data.get("sparse_indices")
        sparse_seq_len = data.get("sparse_seq_len")
        q_lengths = data.get("q_lengths", [int(item) for item in ACTUAL_SEQ_Q])
        kv_lengths = data.get("kv_lengths", [int(item) for item in ACTUAL_SEQ_KV])
        cu_seqlens_q = data.get("cu_seqlens_q")
        cu_seqlens_kv = data.get("cu_seqlens_kv")
        if cu_seqlens_q is None:
            cu_seqlens_q = _prefix(q_lengths)
        if cu_seqlens_kv is None:
            cu_seqlens_kv = _prefix(kv_lengths)
        if sparse_indices is None or sparse_seq_len is None:
            sparse_indices, sparse_seq_len = _make_reference_sparse_for_lengths(q_lengths, kv_lengths)

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("\n[Done] 数据已保存，退出")
        exit(0)

    if "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        cpu_out, cpu_lse = cpu_mxfp8_golden(q_fp8, k_fp8, v_fp8,
                                   dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                                   q_lengths, kv_lengths, cu_seqlens_q, sparse_indices, sparse_seq_len)
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("\n[Done] CPU 输出已保存，退出")
        exit(0)

    if "npu" in mode:
        logger.info("\n[Step 3] NPU 调用")
        atten_out, lse_out = npu_mxfp8_fa(q_fp8, k_fp8, v_fp8,
                               dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                               q_lengths, kv_lengths, cu_seqlens_q, cu_seqlens_kv,
                               block_table_torch, sparse_indices, sparse_seq_len)
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        logger.info("\n[Done] NPU 输出已保存，退出")
        exit(0)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    result_compare_method.check_result(cpu_out, atten_out)

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        result_compare_method.check_result(cpu_lse, lse_out)
