#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden implementation for aclnnFfnWorkerBatching.

The operator performs token reordering for FFN worker in MoE (Mixture of Experts)
attention/FFN separated deployment:
  1. Sort expert_ids (masked tokens >= EXPERT_MASK_VALUE sorted to end)
  2. Gather token hidden states using sorted indices
  3. Generate group_list (per-expert token counts)
  4. Output session_ids, micro_batch_ids, token_ids, expert_offsets

The golden retrieves cached CPU test data (shared with customize_inputs) via
testcase_name, since the scheduleContext tensor contains device pointers that are
meaningless on CPU.
"""

import numpy as np
import torch

try:
    from .inputs import _DATA_CACHE
except ImportError:
    import importlib.util
    from pathlib import Path
    _inputs_path = Path(__file__).with_name("inputs.py")
    _spec_mod = importlib.util.spec_from_file_location(
        "ffn_wb_inputs", _inputs_path)
    _inputs_module = importlib.util.module_from_spec(_spec_mod)
    _spec_mod.loader.exec_module(_inputs_module)
    _DATA_CACHE = _inputs_module._DATA_CACHE


EXPERT_MASK_VALUE = 1000000


def _get_cached_data(testcase_name):
    if testcase_name not in _DATA_CACHE:
        raise KeyError(
            f"No cached data for testcase '{testcase_name}'. "
            f"Available: {list(_DATA_CACHE.keys())}")
    return _DATA_CACHE[testcase_name]


def _sort_expert_ids(expert_ids_flat):
    """Sort expert IDs ascending; masked entries (>= EXPERT_MASK_VALUE) go last.

    Returns (sorted_order, valid_count, sorted_valid_expert_ids).
    """
    sort_key = np.where(
        expert_ids_flat < EXPERT_MASK_VALUE,
        expert_ids_flat,
        np.iinfo(np.int32).max).astype(np.int64)
    sorted_order = np.argsort(sort_key, kind='stable')
    valid_count = int(np.sum(expert_ids_flat < EXPERT_MASK_VALUE))
    sorted_valid_ids = expert_ids_flat[sorted_order[:valid_count]]
    return sorted_order, valid_count, sorted_valid_ids


def _generate_group_list(sorted_expert_ids, valid_count, expert_num):
    """Build group_list of shape (expert_num, 2): [expert_id, token_count]."""
    group_list = np.zeros((expert_num, 2), dtype=np.int64)
    if valid_count == 0:
        return group_list

    offset = 0
    cur_expert = int(sorted_expert_ids[0])
    token_count = 0

    for i in range(valid_count):
        eid = int(sorted_expert_ids[i])
        if eid != cur_expert:
            if offset < expert_num:
                group_list[offset, 0] = cur_expert
                group_list[offset, 1] = token_count
                offset += 1
            cur_expert = eid
            token_count = 1
        else:
            token_count += 1

    if offset < expert_num:
        group_list[offset, 0] = cur_expert
        group_list[offset, 1] = token_count
        offset += 1

    if offset < expert_num:
        group_list[offset, 0] = 0
        group_list[offset, 1] = 0

    return group_list


def _to_torch(arr, dtype):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    return t


def ffn_worker_batching_golden(scheduleContext, expertNum, maxOutShape,
                               tokenDtype, needSchedule, layerNum,
                               y, groupList, sessionIds, microBatchIds,
                               tokenIds, expertOffsets, dynamicScale,
                               actualTokenNum, **kwargs):
    """CPU golden reference for aclnnFfnWorkerBatching.

    All 14 positional args match aclnnFfnWorkerBatchingGetWorkspaceSize
    (without workspaceSize & executor).  Output tensors are pre-allocated
    placeholders and ignored — the function returns fresh golden tensors.
    """
    testcase_name = kwargs.get('testcase_name', 'default')
    data = _get_cached_data(testcase_name)

    A = data['A']
    M = data['M']
    BS = data['BS']
    K = data['K']
    H = data['H']
    token_data = data['token_data']
    token_scales = data.get('token_scales')
    session_ids_buf = data['session_ids_buf']
    micro_batch_ids_buf = data['micro_batch_ids_buf']
    expert_ids = data['expert_ids']

    Y = A * BS * K

    if tokenDtype == 0:
        out_torch_dtype = torch.float16
        out_np_dtype = np.float16
    elif tokenDtype == 1:
        out_torch_dtype = torch.bfloat16
        out_np_dtype = np.float32
    else:
        out_torch_dtype = torch.int8
        out_np_dtype = np.int8

    expert_ids_flat = expert_ids.reshape(-1).astype(np.int32)

    sorted_order, valid_count, sorted_valid_ids = _sort_expert_ids(
        expert_ids_flat)

    y_out = np.zeros((Y, H), dtype=out_np_dtype)
    session_ids_out = np.zeros(Y, dtype=np.int32)
    micro_batch_ids_out = np.zeros(Y, dtype=np.int32)
    token_ids_out = np.zeros(Y, dtype=np.int32)
    expert_offsets_out = np.zeros(Y, dtype=np.int32)
    dynamic_scale_out = np.ones(Y, dtype=np.float32)

    bsk = BS * K
    cur_mb = data.get('cur_micro_batch_id', 0)

    for i in range(valid_count):
        gidx = int(sorted_order[i])
        a_idx = gidx // bsk
        rem = gidx % bsk
        bs_idx = rem // K
        k_idx = rem % K

        if needSchedule == 0:
            s_idx = int(session_ids_buf[a_idx])
            mb_idx = int(micro_batch_ids_buf[a_idx])
        else:
            s_idx = a_idx
            mb_idx = cur_mb

        session_ids_out[i] = s_idx
        micro_batch_ids_out[i] = mb_idx
        token_ids_out[i] = bs_idx
        expert_offsets_out[i] = k_idx

        y_out[i] = token_data[s_idx, mb_idx, bs_idx, k_idx, :]

        if tokenDtype == 2 and token_scales is not None:
            dynamic_scale_out[i] = token_scales[s_idx, mb_idx, bs_idx, k_idx]

    group_list_out = _generate_group_list(
        sorted_valid_ids, valid_count, expertNum)
    actual_token_num_out = np.array([valid_count], dtype=np.int64)

    return [
        _to_torch(y_out, out_torch_dtype),
        _to_torch(group_list_out, None),
        _to_torch(session_ids_out, None),
        _to_torch(micro_batch_ids_out, None),
        _to_torch(token_ids_out, None),
        _to_torch(expert_offsets_out, None),
        _to_torch(dynamic_scale_out, None),
        _to_torch(actual_token_num_out, None),
    ]
