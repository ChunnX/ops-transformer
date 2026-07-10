#!/usr/bin/python
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
Input plugin for compressor.

The TTK framework generates random tensors matching the API schema.  This function
mutates them in-place so they form a structurally valid test case:
  - fills optional sequence-descriptor tensors (start_pos, cu_seqlens, seqused)
    with valid values derived from tensor shapes
  - builds a valid state_block_table with proper physical block ids
"""

import torch

__input__ = {
    "e2e": {
        "cann_ops_transformer.ops.compressor": "generate_compressor_inputs",
    }
}


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def fill_tensor_from_value(tensor, value):
    if tensor is None or value is None or not torch.is_tensor(tensor):
        return
    data = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(data.reshape(tensor.shape))


def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
    if seqused is not None:
        return seqused[batch_idx]
    if cu_seqlens is not None:
        return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
    return S


def build_block_table(block_table, B, S_max, block_size, cmp_ratio,
                      start_pos, seqused, cu_seqlens, cache_mode, S):
    """Build a valid state_block_table following the same logic as run_compressor_eager.

    For cache_mode == 1 the table has shape (B, max_block_num_per_batch) and maps
    each (batch, block_index) to a physical block id in state_cache.  Block id 0
    means invalid (no data).  We assign sequential ids starting from 1 for blocks
    that need to be read (around start_pos) and blocks that need to be written
    (around start_pos + seq_used).
    """
    if block_table is None or not torch.is_tensor(block_table):
        return

    block_table.zero_()
    max_block_num_per_batch = (S_max + block_size - 1) // block_size
    block_num = B * max_block_num_per_batch
    if block_table.shape[0] < B:
        return


    if cache_mode == 1:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        for i in range(B):
            # 需要读取state的范围
            cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
            cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
            if start_pos[i] % cmp_ratio == 0:
                cur_end = start_pos[i]
            cur_end = min(cur_end, start_pos[i] + S)
            cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
            cur_end_block_id = (cur_end - 1) // block_size
            for j in range(cur_start_block_id, cur_end_block_id + 1):
                if next_block_id < block_num:
                    block_table[i][j] = next_block_id
                    next_block_id = next_block_id + 1
            # 需要写入state的范围
            end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
            next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
            next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
            if (start_pos[i] + end_pos) % cmp_ratio == 0:
                next_end = start_pos[i] + end_pos
            next_end = min(next_end, start_pos[i] + end_pos)
            next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
            next_end_block_id = (next_end - 1) // block_size
            for j in range(next_start_block_id, next_end_block_id + 1):
                if next_block_id < block_num and block_table[i][j] == 0:
                    block_table[i][j] = next_block_id
                    next_block_id = next_block_id + 1
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        


def generate_compressor_inputs(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *,
                                state_block_table=None, cu_seqlens=None, seqused=None,
                                start_pos=None, coff=1, cache_mode=1, **kwargs):
    """Generate valid inputs for the compressor operator.

    All parameters are inferred from tensor shapes and API attributes — no
    test-specific kwargs are expected.  Optional tensors that are None are left
    as None; tensors that exist are filled with valid values in-place.
    """
    if x is None or not torch.is_tensor(x):
        return
    
    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    coff_val = int(coff) if coff is not None else 1
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1

    block_size = 128
    if state_cache is not None and torch.is_tensor(state_cache) and state_cache.dim() >= 2:
        block_size = int(state_cache.shape[1])

    is_th = x.dim() == 2

    cu_seqlens_values = kwargs.get('cu_seqlens_values')
    seqused_values = kwargs.get('seqused_values')
    start_pos_values = kwargs.get('start_pos_values')

    if is_th:
        T = x.shape[0]
        if cu_seqlens_values is not None:
            cu_seqlens_list = [int(v) for v in cu_seqlens_values]
            B = len(cu_seqlens_list) - 1
            S = T // B
    else:
        cu_seqlens_list = None
        B = x.shape[0]
        S = x.shape[1]

    if seqused_values is not None:
        seqused_list = [int(v) for v in seqused_values]
    else:
        seqused_list = None

    if start_pos_values is not None:
        start_pos_list = [max(0, int(v)) for v in start_pos_values]
    else:
        start_pos_list = [0] * B

    if is_th:
        S_max = 0
        for i in range(B):
            if start_pos_list[i] + cu_seqlens_list[i + 1] - cu_seqlens_list[i] > S_max:
                S_max = cu_seqlens_list[i + 1] - cu_seqlens_list[i] + start_pos_list[i]
    else:
        S_max = max(start_pos_list) + S if start_pos_list else S

    if is_th:
        S = max(seqused_list) if seqused_list else 0
        for i in range(B):
            if (cu_seqlens_list[i + 1] - cu_seqlens_list[i]) > S:
                S = cu_seqlens_list[i + 1] - cu_seqlens_list[i]

    if state_block_table is not None and torch.is_tensor(state_block_table):
        build_block_table(state_block_table, B, S_max, block_size,
                          cmp_ratio_val, start_pos_list,
                          seqused_list, cu_seqlens_list, cache_mode_val, S)
