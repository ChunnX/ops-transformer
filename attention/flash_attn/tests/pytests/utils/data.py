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


def generate_npu_mask(b, s1, s2, mask_mode, win_left, win_right, prefix=None):
    if mask_mode is None or mask_mode == 0:
        return None
    elif mask_mode in (3, 4):
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool()
    return mask


def generate_cpu_mask(b, s1, s2, mask_mode, win_left, win_right, prefix=None, index=None, band_index=None):
    if mask_mode is None or mask_mode == 0:
        return None
    elif mask_mode == 3:
        mask = torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1).bool()
    elif mask_mode == 4:
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=win_right + 1 + s2 - s1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-win_left - 1 + s2 - s1)
        mask = (atten_mask_u + atten_mask_l).bool()
    return mask


def get_seqlen_list(actual_seqlen):
    seq_list = torch.zeros(len(actual_seqlen), dtype=torch.int64)
    seq_list[0] = actual_seqlen[0]
    for i in range(1, len(actual_seqlen)):
        seq_list[i] = actual_seqlen[i] - actual_seqlen[i - 1]
    return seq_list


def generate_qkv(b, n1, n2, s1, s2, d, d_v, d_rope, input_layout, dtype,
                 q_range=None, k_range=None, v_range=None):
    def _make_tensor(shape, seed, value_range):
        gen = torch.Generator().manual_seed(seed)
        if value_range is not None:
            lo, hi = float(value_range[0]), float(value_range[1])
            return (torch.rand(shape, generator=gen) * (hi - lo) + lo).to(dtype)
        return torch.randint(10, 11, shape, generator=gen, dtype=torch.int).to(dtype)

    q = _make_tensor((b, n1, s1, d),    42, q_range)
    k = _make_tensor((b, n2, s2, d),    43, k_range)
    v = _make_tensor((b, n2, s2, d_v),  44, v_range)

    q_rope = torch.randn(b, n1, s1, d_rope, generator=torch.Generator().manual_seed(45), dtype=dtype)
    k_rope = torch.randn(b, n2, s2, d_rope, generator=torch.Generator().manual_seed(46), dtype=dtype)

    qf = torch.cat((q, q_rope), -1)
    kf = torch.cat((k, k_rope), -1)

    return q, k, v, q_rope, k_rope, qf, kf


def gen_block_table(b, act_seq_lens_kv, block_size, block_table_shape=[]):
    s2_max = max(act_seq_lens_kv)
    max_block_num_per_batch = (s2_max + block_size - 1) // block_size
    if block_table_shape:
        print(f"generating block_table, the block_table_shape is {block_table_shape}")
        b = block_table_shape[0]
        max_block_num_per_batch = block_table_shape[1]
    block_table = torch.full((b, max_block_num_per_batch), -1, dtype=torch.int32)
    block_num = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        block_num += (b_seq + block_size - 1) // block_size
    block_id_array = torch.randperm(block_num, dtype=torch.int32)
    index = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        for j in range(b_block_num):
            block_table[i][j] = block_id_array[index]
            index = index + 1
    return block_table


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
    elif kv_storage_mode == "PA_NZ":
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
