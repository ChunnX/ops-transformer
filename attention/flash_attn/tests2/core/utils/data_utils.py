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
    dtype           = kwargs.get("dtype", "bf16")

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


def _pa_bbnd_to_bnsd(t: torch.Tensor, params: dict) -> torch.Tensor:
    b = params.get("actual_b", params["B"])
    n2 = params.get("N2", params["N1"])
    d = params["D"]
    s2 = params["S2"]
    seq_kv = params["seqused_kv"]
    bs = params.get("block_size", 128)
    bt = params.get("block_table")
    if bt is None:
        return t.clone()
    if isinstance(bt, torch.Tensor):
        bt = bt.cpu().numpy()
    result = torch.zeros(b, n2, s2, d, dtype=t.dtype)
    for bi in range(b):
        slen = seq_kv[bi] if bi < len(seq_kv) else seq_kv[0]
        nblk = (slen + bs - 1) // bs
        for bj in range(nblk):
            blk_id = bt[bi][bj] if bi < len(bt) and bj < len(bt[bi]) else bj
            if blk_id < 0 or blk_id >= t.shape[0]:
                continue
            start = bj * bs
            end = min(start + bs, slen)
            cnt = end - start
            if cnt > 0:
                result[bi, :, start:end, :] = t[blk_id, :cnt, :, :].permute(1, 0, 2)
    return result


def _pa_bnbd_to_bnsd(t: torch.Tensor, params: dict) -> torch.Tensor:
    b = params.get("actual_b", params["B"])
    n2 = params.get("N2", params["N1"])
    d = params["D"]
    s2 = params["S2"]
    seq_kv = params["seqused_kv"]
    bs = params.get("block_size", 128)
    bt = params.get("block_table")
    if bt is None:
        return t.clone()
    if isinstance(bt, torch.Tensor):
        bt = bt.cpu().numpy()
    result = torch.zeros(b, n2, s2, d, dtype=t.dtype)
    for bi in range(b):
        slen = seq_kv[bi] if bi < len(seq_kv) else seq_kv[0]
        nblk = (slen + bs - 1) // bs
        for bj in range(nblk):
            blk_id = bt[bi][bj] if bi < len(bt) and bj < len(bt[bi]) else bj
            if blk_id < 0 or blk_id >= t.shape[0]:
                continue
            start = bj * bs
            end = min(start + bs, slen)
            cnt = end - start
            if cnt > 0:
                result[bi, :, start:end, :] = t[blk_id, :, :cnt, :]
    return result


def _pa_nz_to_bnsd(t: torch.Tensor, params: dict) -> torch.Tensor:
    dtype_str = params.get("dtype", "fp16")
    blk_element = 16 if dtype_str in ("fp16", "bf16") else 32
    b = params.get("actual_b", params["B"])
    n2 = params.get("N2", params["N1"])
    s2 = params["S2"]
    seq_kv = params["seqused_kv"]
    bs = params.get("block_size", 128)
    bt = params.get("block_table")
    if bt is None:
        return t.clone()
    if isinstance(bt, torch.Tensor):
        bt = bt.cpu().numpy()
    d_actual = t.shape[2] * t.shape[4]
    result = torch.zeros(b, n2, s2, d_actual, dtype=t.dtype)
    for bi in range(b):
        slen = seq_kv[bi] if bi < len(seq_kv) else seq_kv[0]
        nblk = (slen + bs - 1) // bs
        for bj in range(nblk):
            blk_id = bt[bi][bj] if bi < len(bt) and bj < len(bt[bi]) else bj
            if blk_id < 0 or blk_id >= t.shape[0]:
                continue
            start = bj * bs
            end = min(start + bs, slen)
            cnt = end - start
            if cnt > 0:
                block_data = t[blk_id, :, :, :cnt, :]
                block_data = block_data.permute(0, 2, 1, 3).reshape(
                    block_data.shape[0], cnt, -1)
                result[bi, :, start:end, :] = block_data
    return result


_PA_TO_BNSD_DISPATCH = {
    "PA_BBND": _pa_bbnd_to_bnsd,
    "PA_BNBD": _pa_bnbd_to_bnsd,
    "PA_NZ": _pa_nz_to_bnsd,
}


def pa_to_bnsd(t: torch.Tensor, layout: str, params: dict) -> torch.Tensor:
    fn = _PA_TO_BNSD_DISPATCH.get(layout)
    if fn is None:
        return t.clone()
    return fn(t, params)
