#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import os
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math
import custom_ops as ops
import torchair
from torchair.configs.compiler_config import CompilerConfig

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, q, ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table, 
        cmp_block_table, cu_seqlens_q, seqused_kv, sinks, metadata, kv_quant_mode, tile_size, rope_head_dim, 
        softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv):
        npu_result, _ = torch_npu.npu_kv_quant_sparse_attn_sharedkv(
                                            q=q,
                                            ori_kv=ori_kv,
                                            cmp_kv=cmp_kv,
                                            cmp_sparse_indices=cmp_sparse_indices,
                                            ori_block_table=ori_block_table,
                                            cmp_block_table=cmp_block_table,
                                            cu_seqlens_q=cu_seqlens_q,
                                            seqused_kv=seqused_kv,
                                            sinks=sinks,
                                            metadata=metadata,
                                            kv_quant_mode=kv_quant_mode,
                                            tile_size=tile_size,
                                            rope_head_dim=rope_head_dim,
                                            softmax_scale=softmax_scale,
                                            cmp_ratio=cmp_ratio,
                                            ori_mask_mode=ori_mask_mode,
                                            cmp_mask_mode=cmp_mask_mode,
                                            ori_win_left=ori_win_left,
                                            ori_win_right=ori_win_right,
                                            layout_q=layout_q,
                                            layout_kv=layout_kv)
        return npu_result


def test_sas_quant_process(test_data, device_id=0):
    params = test_data['params']
    metadata_input = test_data['metadata_input']
    input = test_data['input']
    cpu_output = test_data['cpu_output']

    npu_mode = Network().npu()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    config.mode = "reduce-overhead"
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)

    # print("执行用例：", test_data['Testcase_Name'])
    print("test_data:", params)

    torch_npu.npu.set_device(device_id)

    print("npu_kv_quant_sparse_attn_sharedkv_metadata...")
    metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
                                                        q=input['q'].npu() if input['q'] is not None else None,
                                                        num_heads_q = metadata_input['num_heads_q'],
                                                        num_heads_kv = metadata_input['num_heads_kv'],
                                                        head_dim = metadata_input['head_dim'],
                                                        kv_quant_mode = 1,
                                                        cu_seqlens_q=metadata_input['cu_seqlens_q'].npu() if input['cu_seqlens_q'] is not None else None,
                                                        cu_seqlens_ori_kv = torch.tensor([]).npu(),
                                                        cu_seqlens_cmp_kv = torch.tensor([]).npu(),
                                                        seqused_q = torch.tensor([]).npu(),
                                                        seqused_kv = metadata_input['seqused_kv'].npu() if input['seqused_kv'] is not None else None,
                                                        batch_size = metadata_input['batch_size'],
                                                        max_seqlen_q = metadata_input['max_seqlen_q'],
                                                        max_seqlen_kv = metadata_input['max_seqlen_kv'],
                                                        cmp_topk = metadata_input['topk'],
                                                        cmp_ratio = metadata_input['cmp_ratio'],
                                                        ori_mask_mode = metadata_input['ori_mask_mode'],
                                                        cmp_mask_mode = metadata_input['cmp_mask_mode'],
                                                        ori_win_left = metadata_input['ori_win_left'],
                                                        ori_win_right = metadata_input['ori_win_right'],
                                                        layout_q = metadata_input['layout_q'],
                                                        layout_kv = metadata_input['layout_kv'],
                                                        has_ori_kv = metadata_input['has_ori_kv'],
                                                        has_cmp_kv = metadata_input['has_cmp_kv'])

    torch.npu.synchronize()
    metadata.npu()
    print("npu_kv_quant_sparse_attn_sharedkv...")

    npu_result = npu_mode(
                            q=input['q'].npu() if input['q'] is not None else None,
                            ori_kv=input['ori_kv'].npu() if input['ori_kv'] is not None else None,
                            cmp_kv=input['cmp_kv'].npu() if input['cmp_kv'] is not None else None,
                            cmp_sparse_indices=input['cmp_sparse_indices'].npu() if input['cmp_sparse_indices'] is not None else None,
                            ori_block_table=input['ori_block_table'].npu() if input['ori_block_table'] is not None else None,
                            cmp_block_table=input['cmp_block_table'].npu() if input['cmp_block_table'] is not None else None,
                            cu_seqlens_q=input['cu_seqlens_q'].npu() if input['cu_seqlens_q'] is not None else None,
                            seqused_kv=input['seqused_kv'].npu() if input['seqused_kv'] is not None else None,
                            sinks=input['sinks'].npu() if input['sinks'] is not None else None,
                            metadata=metadata,
                            kv_quant_mode=input['kv_quant_mode'],
                            tile_size=input['tile_size'],
                            rope_head_dim=input['rope_head_dim'],
                            softmax_scale=input['softmax_scale'],
                            cmp_ratio=input['cmp_ratio'],
                            ori_mask_mode=input['ori_mask_mode'],
                            cmp_mask_mode=input['cmp_mask_mode'],
                            ori_win_left=input['ori_win_left'],
                            ori_win_right=input['ori_win_right'],
                            layout_q=input['layout_q'],
                            layout_kv=input['layout_kv'])

    torch.npu.synchronize()

    return npu_result, cpu_output