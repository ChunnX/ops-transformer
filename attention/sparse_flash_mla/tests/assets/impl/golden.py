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

"""CPU golden adapter for SparseFlashMla TTK cases."""


class CaseDataStore:
    """Share pytest-generated case data between input and golden callbacks."""

    def __init__(self):
        self.case_data = {}

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.case_data[str(testcase_name)] = data

    def get(self, testcase_name):
        if testcase_name is None:
            return None
        return self.case_data.get(str(testcase_name))


CASE_DATA = CaseDataStore()


def cpu_sparse_flash_mla(q, *, return_softmax_lse=False, testcase_name=None,
                         layout_q="BSND", **kwargs):
    del q, layout_q, kwargs
    data = CASE_DATA.get(testcase_name)
    if data is None:
        raise RuntimeError(
            "SparseFlashMla TTK golden requires customize_inputs to generate pytest data first"
        )
    lse = data.get("softmax_lse")
    if lse is None or not bool(return_softmax_lse):
        lse = None
    return data["cpu_output"], lse


def cpu_aclnn_sparse_flash_mla(
        q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q,
        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
        ori_topk_length, cmp_topk_length, sinks, metadata, softmax_scale,
        cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
        ori_win_right, layout_q, layout_kv, topk_value_mode,
        return_softmax_lse, attn_out, softmax_lse_out,
        testcase_name=None, **kwargs):
    """Return the same pytest Golden for the ACLNN C API parameter order."""
    del (
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_kv,
        topk_value_mode,
        attn_out,
        softmax_lse_out,
    )
    return cpu_sparse_flash_mla(
        q,
        return_softmax_lse=return_softmax_lse,
        testcase_name=testcase_name,
        layout_q=layout_q,
        **kwargs,
    )
