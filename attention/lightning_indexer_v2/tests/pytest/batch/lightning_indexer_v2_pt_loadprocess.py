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
from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata


def test_liv2_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu")

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    topk_value = test_data['topk_value']
    output_idx_offset = test_data['output_idx_offset'].npu()
    max_seqlen_q = test_data['max_seqlen_q']
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    query = test_data['query'].npu()
    key = test_data['key'].npu()

    topk = params[19]
    mask_mode = params[20]
    return_value = params[25]
    weights =test_data['weights'].npu()
    seqused_q = test_data['seqused_q'].npu()
    seqused_k = test_data['seqused_k'].npu()
    if test_data['cu_seqlens_q'] is not None:
        cu_seqlens_q = test_data['cu_seqlens_q'].npu()
    else:
        cu_seqlens_q = None
    if test_data['cu_seqlens_k'] is not None:
        cu_seqlens_k = test_data['cu_seqlens_k'].npu()
    else:
        cu_seqlens_k = None
    block_table = test_data['block_table'].npu()
    if test_data['metadata'] is not None:
        metadata = test_data['metadata'].npu()
    else:
        metadata = None
    layout_query = test_data['layout_query']
    layout_key = test_data['layout_key']
    cmp_ratio = test_data['cmp_ratio']
    if test_data['cmp_residual_k'] is not None:
        cmp_residual_k = test_data['cmp_residual_k'].npu()
    else:
        cmp_residual_k = None

    #调用SFA算子
    npu_result, npu_topk_value = lightning_indexer(query, key, weights,
                                             cu_seqlens_q = cu_seqlens_q,
                                             cu_seqlens_k = cu_seqlens_k,
                                             seqused_q = seqused_q,
                                             seqused_k = seqused_k,
                                             cmp_residual_k = cmp_residual_k,
                                             block_table = block_table,
                                             output_idx_offset = output_idx_offset,
                                             metadata = metadata,
                                             topk = topk,
                                             max_seqlen_q = max_seqlen_q,
                                             layout_q = layout_query,
                                             layout_k = layout_key,
                                             mask_mode = mask_mode,
                                             cmp_ratio = cmp_ratio,
                                             return_value = return_value)
    torch.npu.synchronize()

    return cpu_result, npu_result, topk_value, output_idx_offset, params

