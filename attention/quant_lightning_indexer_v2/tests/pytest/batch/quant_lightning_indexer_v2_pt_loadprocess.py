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

import os
import pandas as pd 
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
import cann_ops_transformer


def test_qliv2_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu")

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    topk_value = test_data['topk_value']
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    if params[10] == 'FLOAT8_E4M3FN':
        query = test_data['query'].to(dtype=torch.float8_e4m3fn).npu()
        key = test_data['key'].to(dtype=torch.float8_e4m3fn).npu()
    else:
        query = test_data['query'].npu()
        key = test_data['key'].npu()

    max_seqlen_q = params[18]
    return_value = params[30]
    weights =test_data['weights'].npu()
    query_dequant_scale = test_data['query_dequant_scale'].npu()
    key_dequant_scale = test_data['key_dequant_scale'].npu()
    actual_seq_lengths_query = test_data['actual_seq_lengths_query'].npu()
    actual_seq_lengths_key = test_data['actual_seq_lengths_key'].npu()
    if test_data['cu_seqlens_query'] is not None:
        cu_seqlens_query = test_data['cu_seqlens_query'].npu()
    else:
        cu_seqlens_query = None
    if test_data['cu_seqlens_key'] is not None:
        cu_seqlens_key = test_data['cu_seqlens_key'].npu()
    else:
        cu_seqlens_key = None
    block_table = test_data['block_table'].npu()
    metadata = test_data['metadata'].npu()
    quant_mode = test_data['quant_mode']
    layout_query = test_data['layout_query']
    layout_key = test_data['layout_key']
    sparse_count = test_data['sparse_count']
    sparse_mode = test_data['sparse_mode']
    cmp_ratio = test_data['cmp_ratio']
    if test_data['cmp_residual_k_for_npu'] is not None:
        cmp_residual_k_for_npu = test_data['cmp_residual_k_for_npu'].npu()
    else:
        cmp_residual_k_for_npu = None

    #调用SFA算子
    npu_result,_ = torch.ops.cann_ops_transformer.quant_lightning_indexer(query, key, weights, 
                                                    query_dequant_scale,
                                                    key_dequant_scale,
                                                    cu_seqlens_q = cu_seqlens_query,
                                                    cu_seqlens_k = cu_seqlens_key,
                                                    seqused_q = actual_seq_lengths_query,
                                                    seqused_k = actual_seq_lengths_key,
                                                    cmp_residual_k = cmp_residual_k_for_npu,
                                                    output_idx_offset = None,
                                                    max_seqlen_q = max_seqlen_q,
                                                    block_table=block_table,
                                                    metadata = metadata,
                                                    quant_mode = quant_mode,
                                                    layout_q = layout_query,
                                                    layout_k = layout_key, 
                                                    topk = sparse_count,
                                                    mask_mode = sparse_mode,
                                                    cmp_ratio = cmp_ratio,
                                                    return_value = return_value)
    
    torch.npu.synchronize()

    return cpu_result, npu_result, topk_value, params

