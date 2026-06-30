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

import test
import torch
import torch_npu
import pytest
import torchair
import torch.nn as nn
from torchair.configs.compiler_config import CompilerConfig
import cann_ops_transformer
import quant_lightning_indexer_v2_golden


class QLIV2Network(nn.Module):
    def __init__(self):
        super(QLIV2Network, self).__init__()

    def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale, cu_seqlens_q, cu_seqlens_k,
                seqused_q, seqused_k, cmp_residual_k, output_idx_offset, max_seqlen_q, block_table, metadata,
                quant_mode, layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio, return_value):
        return torch.ops.cann_ops_transformer.quant_lightning_indexer(query, key, weights,
                                                    query_dequant_scale,
                                                    key_dequant_scale,
                                                    cu_seqlens_q=cu_seqlens_q,
                                                    cu_seqlens_k=cu_seqlens_k,
                                                    seqused_q=seqused_q,
                                                    seqused_k=seqused_k,
                                                    cmp_residual_k=cmp_residual_k,
                                                    output_idx_offset=output_idx_offset,
                                                    max_seqlen_q=max_seqlen_q,
                                                    block_table=block_table,
                                                    metadata=metadata,
                                                    quant_mode=quant_mode,
                                                    layout_q=layout_query,
                                                    layout_k=layout_key,
                                                    topk=sparse_count,
                                                    mask_mode=sparse_mode,
                                                    cmp_ratio=cmp_ratio,
                                                    return_value=return_value)


def qliv2_output_acl_graph(params):
    qk_dtype = params[10]
    max_seqlen_q = params[18]
    return_value = params[30]

    test_data = quant_lightning_indexer_v2_golden.qliv2_output_single(params, is_batch=True)
    cpu_result = test_data["cpu_result"]
    topk_value = test_data["topk_value"]
    cpu_topk_value = test_data["cpu_topk_value"]

    query = test_data["query"]
    key = test_data["key"]
    if qk_dtype == torch.float8_e4m3fn:
        query = query.to(dtype=torch.float8_e4m3fn)
        key = key.to(dtype=torch.float8_e4m3fn)

    weights = test_data["weights"]
    query_dequant_scale = test_data["query_dequant_scale"]
    key_dequant_scale = test_data["key_dequant_scale"]
    cu_seqlens_query = test_data["cu_seqlens_query"]
    cu_seqlens_key = test_data["cu_seqlens_key"]
    seqused_q = test_data["seqused_q"]
    seqused_k = test_data["seqused_k"]
    cmp_residual_k = test_data["cmp_residual_k_for_npu"]
    output_idx_offset = test_data["output_idx_offset"]
    block_table = test_data["block_table"]
    metadata = test_data["metadata"]
    quant_mode = test_data["quant_mode"]
    layout_query = test_data["layout_query"]
    layout_key = test_data["layout_key"]
    sparse_count = test_data["sparse_count"]
    sparse_mode = test_data["sparse_mode"]
    cmp_ratio = test_data["cmp_ratio"]

    print("acl_graph")

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    config.mode = "reduce-overhead"
    npu_mode = torch.compile(QLIV2Network().npu(), fullgraph=True, backend=npu_backend, dynamic=False)
    npu_result, npu_topk_value = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale,
                        cu_seqlens_q=cu_seqlens_query,
                        cu_seqlens_k=cu_seqlens_key,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        cmp_residual_k=cmp_residual_k,
                        output_idx_offset=output_idx_offset,
                        max_seqlen_q=max_seqlen_q,
                        block_table=block_table,
                        metadata=metadata,
                        quant_mode=quant_mode,
                        layout_query=layout_query,
                        layout_key=layout_key,
                        sparse_count=sparse_count,
                        sparse_mode=sparse_mode,
                        cmp_ratio=cmp_ratio,
                        return_value=return_value)
    torch.npu.synchronize()
    npu_topk_value, _ = npu_topk_value.sort(dim=-1, descending=True)
    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value
