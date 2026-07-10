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

    next_block_id = 1
    for i in range(B):
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        batch_seq_used = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        cur_end = min(cur_end, start_pos[i] + batch_seq_used)
        cur_start_block_id = max(0, cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size if cur_end > 0 else 0
        for j in range(cur_start_block_id, min(cur_end_block_id + 1, block_table.shape[1])):
            if next_block_id < block_num:
                block_table[i][j] = next_block_id
                next_block_id += 1

        end_pos = batch_seq_used
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos
        next_end = min(next_end, start_pos[i] + end_pos)
        next_start = max(0, next_start)
        next_start_block_id = next_start // block_size if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size if next_end > 0 else 0
        for j in range(next_start_block_id, min(next_end_block_id + 1, block_table.shape[1])):
            if next_block_id < block_num and int(block_table[i][j].item()) == 0:
                block_table[i][j] = next_block_id
                next_block_id += 1


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

    cu_seqlens_values = kwargs.get('cu_seqlens_values')
    seqused_values = kwargs.get('seqused_values')
    start_pos_values = kwargs.get('start_pos_values')
    
    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    coff_val = int(coff) if coff is not None else 1
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1

    block_size = 128
    if state_cache is not None and torch.is_tensor(state_cache) and state_cache.dim() >= 2:
        block_size = int(state_cache.shape[1])

    is_th = x.dim() == 2

    if is_th:
        T = x.shape[0]
        if cu_seqlens is not None and torch.is_tensor(cu_seqlens):
            cu_seqlens_list = to_list(cu_seqlens)
            if len(cu_seqlens_list) < 2:
                cu_seqlens_list = [0, T]
            else:
                cu_seqlens_list[-1] = T
                for idx in range(1, len(cu_seqlens_list) - 1):
                    cu_seqlens_list[idx] = min(max(cu_seqlens_list[idx], cu_seqlens_list[idx - 1]), T)
            fill_tensor_from_value(cu_seqlens, cu_seqlens_list)
        else:
            cu_seqlens_list = [0, T]
        B = len(cu_seqlens_list) - 1
    else:
        cu_seqlens_list = None
        B = x.shape[0]

    if seqused is not None and torch.is_tensor(seqused):
        seqused_list = to_list(seqused)
        if not seqused_list:
            if is_th:
                seqused_list = [cu_seqlens_list[i + 1] - cu_seqlens_list[i] for i in range(B)]
            else:
                seqused_list = [x.shape[1]] * B
        else:
            seqused_list = [max(0, min(int(v), x.shape[1] if not is_th else (cu_seqlens_list[i + 1] - cu_seqlens_list[i] if i < B else x.shape[1]))) for i, v in enumerate(seqused_list)]
            if len(seqused_list) < B:
                seqused_list.extend([seqused_list[-1] if seqused_list else 1] * (B - len(seqused_list)))
        fill_tensor_from_value(seqused, seqused_list)
    else:
        if is_th:
            seqused_list = [cu_seqlens_list[i + 1] - cu_seqlens_list[i] for i in range(B)]
        else:
            seqused_list = [x.shape[1]] * B

    if start_pos is not None and torch.is_tensor(start_pos):
        start_pos_list = to_list(start_pos)
        if len(start_pos_list) < B:
            start_pos_list.extend([0] * (B - len(start_pos_list)))
        start_pos_list = [max(0, int(v)) for v in start_pos_list[:B]]
        fill_tensor_from_value(start_pos, start_pos_list)
    else:
        start_pos_list = [0] * B

    if is_th:
        S = 0
        for i in range(B):
            S = max(S, get_seq_used_by_batch(i, S, seqused_list, cu_seqlens_list))
        S_max = 0
        for i in range(B):
            S_max = max(S_max, start_pos_list[i] + get_seq_used_by_batch(i, S, seqused_list, cu_seqlens_list))
        S_max = max(S_max, block_size)
    else:
        S = x.shape[1]
        S_max = max(start_pos_list) + S if start_pos_list else S
        S_max = max(S_max, block_size)

    if state_block_table is not None and torch.is_tensor(state_block_table):
        build_block_table(state_block_table, B, S_max, block_size,
                          cmp_ratio_val, start_pos_list,
                          seqused_list, cu_seqlens_list, cache_mode_val, S)
