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

"""Raw Kernel CPU golden adapter for QuantLightningIndexer V2."""

import torch


class CaseDataStore:
    """Share one pytest-generated case between input, golden, and compare callbacks."""

    def __init__(self):
        self.case_data = {}
        self.topk_scores = None
        self.index_offsets = None
        self.score_layout = None
        self.cu_seqlens_q = None

    def clear(self):
        self.case_data.clear()
        self.topk_scores = None
        self.index_offsets = None
        self.score_layout = None
        self.cu_seqlens_q = None

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.clear()
            self.case_data = {str(testcase_name): data}

    def activate(self, testcase_name):
        data = self.case_data.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "QuantLightningIndexer V2 TTK golden requires customize_inputs "
                "to generate pytest data first"
            )
        self.topk_scores = data["topk_value"]
        self.index_offsets = data.get("output_idx_offset")
        self.score_layout = data.get("score_layout")
        self.cu_seqlens_q = data.get("cu_seqlens_q")
        return data


CASE_DATA = CaseDataStore()


def get_topk_scores():
    return CASE_DATA.topk_scores


def get_index_offsets():
    return CASE_DATA.index_offsets


def get_score_context():
    return CASE_DATA.score_layout, CASE_DATA.cu_seqlens_q


def cpu_quant_lightning_indexer_v2(query, key, weights, query_dequant_scale,
                                   key_dequant_scale, *, return_value=0,
                                   testcase_name=None, **kwargs):
    """Return the CPU outputs produced while generating this exact pytest case."""
    del query, key, weights, query_dequant_scale, key_dequant_scale, kwargs
    data = CASE_DATA.activate(testcase_name)
    if int(return_value):
        sparse_value = data["cpu_topk_value"]
    else:
        sparse_value = torch.zeros(0, dtype=data["topk_value"].dtype)
    return data["cpu_result"], sparse_value


def cpu_quant_lightning_indexer_v2_kernel(
        query, key, weights, query_dequant_scale, key_dequant_scale,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        block_table, output_idx_offset, metadata, *, return_value=0,
        testcase_name=None, **kwargs):
    """Return NumPy goldens for the raw Kernel operator signature."""
    from ttk.utilities import torch_to_numpy_tensor

    del cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k
    del cmp_residual_k, block_table, output_idx_offset, metadata
    outputs = cpu_quant_lightning_indexer_v2(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        return_value=return_value,
        testcase_name=testcase_name,
        **kwargs,
    )
    return tuple(
        torch_to_numpy_tensor(output.detach().cpu()) if torch.is_tensor(output) else output
        for output in outputs
    )
