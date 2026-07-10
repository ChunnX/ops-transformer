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

import torch


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
    """Build state_block_table consistent with compressor_golden.py run_compressor_eager.

    Key: cur_end for read range uses S (max seq len across batches), NOT per-batch
    batch_seq_used — matching compressor_golden.py:577.
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
        # read range — uses S (max seq len), matching compressor_golden.py:573-577
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        cur_end = min(cur_end, start_pos[i] + S)
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, min(cur_end_block_id + 1, block_table.shape[1])):
            if next_block_id < block_num:
                block_table[i][j] = next_block_id
                next_block_id += 1

        # write range — uses per-batch end_pos, matching compressor_golden.py:585-600
        end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos
        next_end = min(next_end, start_pos[i] + end_pos)
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, min(next_end_block_id + 1, block_table.shape[1])):
            if next_block_id < block_num and int(block_table[i][j].item()) == 0:
                block_table[i][j] = next_block_id
                next_block_id += 1


def _fill_seq_descriptors(x, state_cache, cmp_ratio, coff, cache_mode,
                          cu_seqlens, seqused, start_pos, **kwargs):
    """Shared logic for filling sequence-descriptor tensors and building block_table.

    Parameter assignment logic consistent with compressor_operator_single.py
    output_operator and compressor_golden.py run_compressor_eager.

    Returns (cu_seqlens_list, seqused_list, start_pos_list, B, S, S_max, block_size).
    """
    if x is None or not torch.is_tensor(x):
        return None, None, None, 0, 0, 0, 128

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
            if len(cu_seqlens_list) < 2:
                cu_seqlens_list = [0, T]
            else:
                cu_seqlens_list[-1] = T
                for idx in range(1, len(cu_seqlens_list) - 1):
                    cu_seqlens_list[idx] = min(max(cu_seqlens_list[idx], cu_seqlens_list[idx - 1]), T)
            if cu_seqlens is not None and torch.is_tensor(cu_seqlens):
                fill_tensor_from_value(cu_seqlens, cu_seqlens_list)
        elif cu_seqlens is not None and torch.is_tensor(cu_seqlens):
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

    if seqused_values is not None:
        seqused_list = [int(v) for v in seqused_values]
        if len(seqused_list) < B:
            seqused_list.extend([seqused_list[-1] if seqused_list else 1] * (B - len(seqused_list)))
        if is_th and cu_seqlens_list is not None:
            seqused_list = [max(0, min(v, cu_seqlens_list[i + 1] - cu_seqlens_list[i])) if i < B else v
                            for i, v in enumerate(seqused_list)]
        elif not is_th:
            seqused_list = [max(0, min(v, x.shape[1])) for v in seqused_list]
        if seqused is not None and torch.is_tensor(seqused):
            fill_tensor_from_value(seqused, seqused_list[:B])
    elif seqused is not None and torch.is_tensor(seqused):
        seqused_list = to_list(seqused)
        if not seqused_list:
            if is_th:
                seqused_list = [cu_seqlens_list[i + 1] - cu_seqlens_list[i] for i in range(B)]
            else:
                seqused_list = [x.shape[1]] * B
        else:
            seqused_list = [max(0, min(int(v), x.shape[1] if not is_th else (cu_seqlens_list[i + 1] - cu_seqlens_list[i] if i < B and cu_seqlens_list else x.shape[1]))) for i, v in enumerate(seqused_list)]
            if len(seqused_list) < B:
                seqused_list.extend([seqused_list[-1] if seqused_list else 1] * (B - len(seqused_list)))
        fill_tensor_from_value(seqused, seqused_list)
    else:
        if is_th:
            seqused_list = [cu_seqlens_list[i + 1] - cu_seqlens_list[i] for i in range(B)]
        else:
            seqused_list = [x.shape[1]] * B

    if start_pos_values is not None:
        start_pos_list = [max(0, int(v)) for v in start_pos_values]
        if len(start_pos_list) < B:
            start_pos_list.extend([0] * (B - len(start_pos_list)))
        if start_pos is not None and torch.is_tensor(start_pos):
            fill_tensor_from_value(start_pos, start_pos_list[:B])
    elif start_pos is not None and torch.is_tensor(start_pos):
        start_pos_list = to_list(start_pos)
        if len(start_pos_list) < B:
            start_pos_list.extend([0] * (B - len(start_pos_list)))
        start_pos_list = [max(0, int(v)) for v in start_pos_list[:B]]
        fill_tensor_from_value(start_pos, start_pos_list)
    else:
        start_pos_list = [0] * B

    # Compute S consistent with compressor_golden.py run_compressor_eager:
    #   TH: S = max(seqused) if seqused else max(cu_seqlens diff)
    #   BSH: S = x.shape[1]
    if is_th:
        if seqused is not None or seqused_values is not None:
            S = max(seqused_list) if seqused_list else 0
        else:
            S = 0
            for i in range(B):
                batch_len = cu_seqlens_list[i + 1] - cu_seqlens_list[i] if cu_seqlens_list else 0
                if batch_len > S:
                    S = batch_len
    else:
        S = x.shape[1]

    # Compute S_max consistent with compressor_operator_single.py output_operator:
    #   TH: S_max = max(start_pos[i] + (cu_seqlens[i+1] - cu_seqlens[i]))
    #       or max(cu_seqlens diff) if no start_pos
    #   BSH: S_max = max(start_pos) + S if start_pos else S
    if is_th:
        S_max = 0
        has_start_pos = (start_pos is not None) or (start_pos_values is not None)
        for i in range(B):
            batch_len = cu_seqlens_list[i + 1] - cu_seqlens_list[i] if cu_seqlens_list else 0
            if has_start_pos:
                val = start_pos_list[i] + batch_len
            else:
                val = batch_len
            if val > S_max:
                S_max = val
    else:
        has_start_pos = (start_pos is not None) or (start_pos_values is not None)
        if has_start_pos and start_pos_list:
            S_max = max(start_pos_list) + S
        else:
            S_max = S
    S_max = max(S_max, block_size)

    return cu_seqlens_list, seqused_list, start_pos_list, B, S, S_max, block_size


def generate_compressor_inputs(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *,
                                state_block_table=None, cu_seqlens=None, seqused=None,
                                start_pos=None, coff=1, cache_mode=1, **kwargs):
    """E2E customize_inputs for cann_ops_transformer.ops.compressor."""
    cu_seqlens_list, seqused_list, start_pos_list, B, S, S_max, block_size = \
        _fill_seq_descriptors(x, state_cache, cmp_ratio, coff, cache_mode,
                              cu_seqlens, seqused, start_pos, **kwargs)

    if x is None or not torch.is_tensor(x):
        return

    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1

    if state_block_table is not None and torch.is_tensor(state_block_table):
        build_block_table(state_block_table, B, S_max, block_size,
                          cmp_ratio_val, start_pos_list,
                          seqused_list, cu_seqlens_list, cache_mode_val, S)
    print(B)
    print(S_max)
    print(S)
    print(block_size)
    print(state_block_table)
    print(start_pos_list)
    print(seqused_list)
    print(cu_seqlens_list)
