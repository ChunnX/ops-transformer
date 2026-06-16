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


def normalize_npu_result(npu_result):
    if isinstance(npu_result, (tuple, list)) and len(npu_result) == 2:
        return npu_result[0].detach().cpu().to(torch.int32), npu_result[1].detach().cpu().to(torch.int32)
    raise TypeError("StemIndexer should return (sparse_indices, sparse_seq_len).")


def assert_stem_indexer_result(expected_indices, expected_seq_len, npu_result, case):
    actual_indices, actual_seq_len = normalize_npu_result(npu_result)
    expected_indices = expected_indices.to(torch.int32)
    expected_seq_len = expected_seq_len.to(torch.int32)

    assert tuple(actual_seq_len.shape) == tuple(expected_seq_len.shape), (
        f"{case['case_id']} sparse_seq_len shape mismatch: "
        f"actual={tuple(actual_seq_len.shape)}, expected={tuple(expected_seq_len.shape)}"
    )
    assert tuple(actual_indices.shape) == tuple(expected_indices.shape), (
        f"{case['case_id']} sparse_indices shape mismatch: "
        f"actual={tuple(actual_indices.shape)}, expected={tuple(expected_indices.shape)}"
    )
    assert torch.equal(actual_seq_len, expected_seq_len), f"{case['case_id']} sparse_seq_len mismatch"

    failures = []
    for index in torch.nonzero(expected_seq_len >= 0, as_tuple=False):
        b_idx, q_head_idx, q_block_idx = [int(item) for item in index]
        valid_len = int(expected_seq_len[b_idx, q_head_idx, q_block_idx])
        if valid_len == 0:
            continue
        actual_prefix = actual_indices[b_idx, q_head_idx, q_block_idx, :valid_len].tolist()
        expected_prefix = expected_indices[b_idx, q_head_idx, q_block_idx, :valid_len].tolist()
        if sorted(actual_prefix) != sorted(expected_prefix):
            failures.append({
                "b": b_idx,
                "q_head": q_head_idx,
                "q_block": q_block_idx,
                "actual": actual_prefix,
                "expected": expected_prefix,
            })
        if len(failures) >= 8:
            break

    assert not failures, f"{case['case_id']} sparse_indices valid prefix mismatch: {failures}"
