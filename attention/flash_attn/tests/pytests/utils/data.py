#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
from einops import rearrange


def generate_cpu_mask(b, s1, s2, mask_mode, win_left, win_right, prefix=None, index=None, band_index=None):
    if mask_mode is None or mask_mode == 0:
        return None
    elif mask_mode == 3:
        mask = torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1).bool()
    elif mask_mode == 4:
        # -1 means infinity (no mask on that side)
        effective_win_right = s2 if win_right == -1 else win_right
        effective_win_left = s1 if win_left == -1 else win_left
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=effective_win_right + 1 + s2 - s1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-effective_win_left - 1 + s2 - s1)
        mask = (atten_mask_u + atten_mask_l).bool()
    return mask


def page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max()) + 1
    kv_cache_bnsd_shape = (block_num, bnsd_tensor.shape[1], block_size, bnsd_tensor.shape[3])
    page_cache_tensor = torch.zeros(size=kv_cache_bnsd_shape, dtype=bnsd_tensor.dtype)
    actual_b = block_table.shape[0]
    is_tnd = (b == 1 and actual_b > 1)
    cum_kv_offset = 0
    for i in range(actual_b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > i else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        if is_tnd:
            batch_dim = 0
            seq_offset = cum_kv_offset
        else:
            batch_dim = i
            seq_offset = 0
        for j in range(b_block_num):
            start_idx = seq_offset + j * block_size
            end_idx = seq_offset + min((j + 1) * block_size, b_seq)
            actual_size = end_idx - start_idx
            slice_data = bnsd_tensor[batch_dim, :, start_idx:end_idx, :]
            page_cache_tensor[block_table[i][j], :, :actual_size, :] = slice_data
        cum_kv_offset += b_seq
    return page_cache_tensor


def dtype_sizeof(data_type):
    if data_type == 'fp16' or data_type == 'bf16':
        return 2
    elif data_type == 'int8' or data_type == 'fp8':
        return 1


def rearrange_by_block_table(bnsd_tensor, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype):
    page_cache_tensor = page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size)
    if kv_storage_mode == "PA_BBND":
        return page_cache_tensor.permute(0, 2, 1, 3)
    elif kv_storage_mode == "PA_BNBD":
        return page_cache_tensor
    elif kv_storage_mode == "PA_NZ":  # PA_NZ: shape=(NumBlocks, N_kv, BlockSize/16, D, 16)，分页注意力NZ格式
        blk_elem = 32 // dtype_sizeof(kv_dtype)
        page_cache_tensor = page_cache_tensor.reshape(page_cache_tensor.shape[0],
                                                       page_cache_tensor.shape[1],
                                                       page_cache_tensor.shape[2],
                                                       page_cache_tensor.shape[3] // blk_elem,
                                                       blk_elem).permute(0, 1, 3, 2, 4)
        return page_cache_tensor
    else:
        return None


def trans_bnsd_to_layout(tensor, layout_type, **kwargs):
    b               = kwargs.get("B")
    block_size      = kwargs.get("block_size")
    block_table     = kwargs.get("block_table")
    seqused_kv      = kwargs.get("seqused_kv")
    dtype           = kwargs.get("Dtype", "bf16")

    if tensor is None:
        return None
    else:
        if layout_type == "BSH":
            return rearrange(tensor.clone(), 'b n s d -> b s (n d)')
        elif layout_type == "SBH":
            return rearrange(tensor.clone(), 'b n s d -> s b (n d)')
        elif layout_type == "BSND":
            return rearrange(tensor.clone(), 'b n s d -> b s n d')
        elif layout_type == "TND":
            return rearrange(tensor.clone(), '1 n s d -> s n d')
        elif layout_type == "PA_BBND" or layout_type == "PA_BNBD" or layout_type == "PA_NZ":
            return rearrange_by_block_table(tensor, block_table, block_size, b, seqused_kv, layout_type, dtype)
        return tensor.clone()


def make_noncontiguous(tensor, nc_dims):
    """返回指定维非连续但底层数据相同的 tensor view.

    Args:
        tensor: 连续的 torch.Tensor
        nc_dims: 0     --> 仅 dim0 非连续 (pad dim1 后切片)
                 (0,1) --> dim0+dim1 都非连续 (pad dim2 后切片)

    Note: dim1-only 物理上不可行——dim1 stride 膨胀必然导致 dim0 stride 联动;
          测试用例中所有 dim1 场景已移除。

    Returns:
        非连续 tensor view (shape 不变, .is_contiguous() == False)
    """
    device = tensor.device
    dtype = tensor.dtype
    shape = tensor.shape

    if nc_dims == 0:
        pad_dim = 1
        shp = list(shape)
        shp[pad_dim] += 1
        p = torch.zeros(shp, dtype=dtype, device=device)
        ws = [slice(None)] * len(shape)
        ws[pad_dim] = slice(None, shape[pad_dim])
        p[tuple(ws)] = tensor
        return p[tuple(ws)]
    elif nc_dims == (0, 1):
        pad_dim = 2
        shp = list(shape)
        shp[pad_dim] += 1
        p = torch.zeros(shp, dtype=dtype, device=device)
        ws = [slice(None)] * len(shape)
        ws[pad_dim] = slice(None, shape[pad_dim])
        p[tuple(ws)] = tensor
        return p[tuple(ws)]
    return tensor


import math
import numpy as np

PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHIL0X_W_32 = [0x9E3779B9, 0xBB67AE85]


def _philox4_round(counter, key, philox_m, len_w, mask_w):
    prod = philox_m[0] * counter[0]
    hi_1 = prod >> len_w
    lo_1 = prod & mask_w
    prod = philox_m[1] * counter[2]
    hi_2 = prod >> len_w
    lo_2 = prod & mask_w
    counter[0] = hi_2 ^ counter[1] ^ key[0]
    counter[1] = lo_2
    counter[2] = hi_1 ^ counter[3] ^ key[1]
    counter[3] = lo_1


def _philox4_bumpkey(key, philox_w, mask_w):
    key[0] = (key[0] + philox_w[0]) & mask_w
    key[1] = (key[1] + philox_w[1]) & mask_w


def _philox(counter, key, rounds):
    for i in range(rounds - 1):
        _philox4_round(counter, key, PHILOX_M4_32, 32, 0xffffffff)
        _philox4_bumpkey(key, PHIL0X_W_32, 0xffffffff)
    _philox4_round(counter, key, PHILOX_M4_32, 32, 0xffffffff)
    return counter


def _inc_counter(counter):
    for i in range(4):
        counter[i] = (counter[i] + 1) & 0xffffffff
        if counter[i] != 0:
            return counter
    return counter


def _philox_random(rounds, counter, key, count):
    ret = list()
    for i in range((count + 255) // 256 * 256 // 4):
        ret.extend(_philox(counter[:], key[:], rounds))
        counter = _inc_counter(counter[:])
    return np.array(ret)[:count]


def _uint32_to_uint8_array(data):
    uint8_array = np.zeros((4), dtype=np.uint8)
    uint8_array[0] = data & 0xff
    uint8_array[1] = (data >> 8) & 0xff
    uint8_array[2] = (data >> 16) & 0xff
    uint8_array[3] = (data >> 24) & 0xff
    return uint8_array


def _gen_dropmask_uint8(rounds, counter, key, uint8_count, keep_prob):
    keep_prob_uint8 = keep_prob * 255
    uint32_count = math.ceil(uint8_count / 4)
    philox_uint32 = _philox_random(rounds, counter, key, uint32_count)
    golden_uint8 = []
    for each_uint32 in philox_uint32:
        four_uint8 = _uint32_to_uint8_array(each_uint32)
        for each_philox_uint8 in four_uint8:
            compare_res = 1 if each_philox_uint8 <= keep_prob_uint8 else 0
            golden_uint8.append(compare_res)
    return golden_uint8[:uint8_count]


def gen_dropmask(b, n1, s1, s2, keep_prob=0.9, seed=2, offset=0):
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    s2_aligned = math.ceil(s2 / 16) * 16
    count = b * n1 * s1 * s2_aligned
    golden_uint8 = _gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    if s2_aligned != s2:
        drop_mask_np = np.array(golden_uint8).reshape(b, n1, s1, s2_aligned)[:, :, :, :s2].astype(np.uint8)
    else:
        drop_mask_np = np.array(golden_uint8).reshape(b, n1, s1, s2).astype(np.uint8)
    return torch.from_numpy(drop_mask_np)


def gen_dropmask_gs1(b, n2, g_size, s1, s2, keep_prob=0.9, seed=2, offset=0, layout="BSND"):
    """Generate dropmask with gS1 layout.
    gS1 flattening depends on layout:
      BSND/TND (S1G): gS1 = s1*gSize + g  (s1 outer, g inner)
      BNSD (GS1):     gS1 = g*s1 + s1     (g outer, s1 inner)
    Returns mask in (B, N1, S1, S2) shape where N1 = N2 * G.
    """
    n1 = n2 * g_size
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    s2_aligned = math.ceil(s2 / 16) * 16
    gs1 = s1 * g_size
    count = b * n2 * gs1 * s2_aligned
    golden_uint8 = _gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    if s2_aligned != s2:
        mask_gs1 = np.array(golden_uint8).reshape(b, n2, gs1, s2_aligned)[:, :, :, :s2].astype(np.uint8)
    else:
        mask_gs1 = np.array(golden_uint8).reshape(b, n2, gs1, s2).astype(np.uint8)
    if layout == "BNSD":
        # GS1: gS1 = g*s1 + s1 -> reshape (B, N2, G, S1, S2) -> (B, N1, S1, S2)
        mask = mask_gs1.reshape(b, n2, g_size, s1, s2).reshape(b, n1, s1, s2)
    else:
        # S1G: gS1 = s1*gSize + g -> reshape (B, N2, S1, G, S2) -> transpose -> (B, N2, G, S1, S2) -> (B, N1, S1, S2)
        mask = mask_gs1.reshape(b, n2, s1, g_size, s2).transpose(0, 1, 3, 2, 4).reshape(b, n1, s1, s2)
    return torch.from_numpy(mask)


def gen_dropmask_tnd(length, keep_prob=0.9, seed=2, offset=0):
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    count = math.ceil(length / 16) * 16
    golden_uint8 = _gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    drop_mask_np = np.array(golden_uint8).astype(np.uint8)
    return torch.from_numpy(drop_mask_np)
