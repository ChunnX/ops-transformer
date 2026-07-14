#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
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
from quant_lightning_indexer_v2_golden import _load_metadata_from_cpu

# ======================== 场景开关 ========================
# False: 场景一 — CPU 上跑仿真，用 cpu_test 二进制生成 metadata，再将所有数据搬到 NPU 调 QLI 算子
# True : 场景二 — 先将所有数据搬到 NPU，再调用真正的 NPU metadata 算子，最后调 QLI 算子
IS_NPU_NOW = False


def test_qliv2_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu")

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    topk_value = test_data['topk_value']
    cpu_topk_value = test_data.get('cpu_topk_value')
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    if params[10] == 'FLOAT8_E4M3FN':
        query = test_data['query'].to(dtype=torch.float8_e4m3fn).cpu()
        key = test_data['key'].to(dtype=torch.float8_e4m3fn).cpu()
    else:
        query = test_data['query'].cpu()
        key = test_data['key'].cpu()

    # params 含 weight_dtype 在 [11]，索引比旧版整体后移 1 位
    batch_size = int(params[0])
    q_head_num = int(params[5])
    k_head_num = int(params[6])
    head_dim = int(params[7])
    max_seqlen_q = int(params[19])
    return_value = int(params[31])

    weights = test_data['weights'].cpu()
    query_dequant_scale = test_data['query_dequant_scale'].cpu()
    key_dequant_scale = test_data['key_dequant_scale'].cpu()
    actual_seq_lengths_query = test_data['actual_seq_lengths_query'].cpu()
    actual_seq_lengths_key = test_data['actual_seq_lengths_key'].cpu()
    if test_data.get('output_idx_offset') is not None:
        output_idx_offset = test_data['output_idx_offset'].cpu()
    else:
        output_idx_offset = None
    if test_data['cu_seqlens_query'] is not None:
        cu_seqlens_query = test_data['cu_seqlens_query'].cpu()
    else:
        cu_seqlens_query = None
    if test_data['cu_seqlens_key'] is not None:
        cu_seqlens_key = test_data['cu_seqlens_key'].cpu()
    else:
        cu_seqlens_key = None
    block_table = test_data['block_table'].cpu()
    quant_mode = test_data['quant_mode']
    layout_query = test_data['layout_query']
    layout_key = test_data['layout_key']
    sparse_count = test_data['sparse_count']
    sparse_mode = test_data['sparse_mode']
    cmp_ratio = test_data['cmp_ratio']
    if test_data['cmp_residual_k_for_npu'] is not None:
        cmp_residual_k_for_npu = test_data['cmp_residual_k_for_npu'].cpu()
    else:
        cmp_residual_k_for_npu = None

    # max_seqlen 从个体长度中取（两种场景都需要）
    max_seqlen_q_meta = actual_seq_lengths_query.max().item()
    max_seqlen_k_meta = actual_seq_lengths_key.max().item()

    if not IS_NPU_NOW:
        # ======================== 场景一：先在 CPU 上生成 metadata ========================
        # 从 .pt 中存的 tensor 还原 _load_metadata_from_cpu 所需的 list 参数

        # cu_seqlens 还原为 list（TND: [B+1] 前缀和, 非TND: None）
        cu_seqlens_q_list = cu_seqlens_query.tolist() if cu_seqlens_query is not None else None
        cu_seqlens_k_list = cu_seqlens_key.tolist() if cu_seqlens_key is not None else None

        # seqused (个体长度) 还原为 list
        #   TND: 从 cu_seqlens 差分推导
        #   非TND: actual_seq_lengths 就是个体长度
        if cu_seqlens_q_list is not None:
            lengths_q_list = [cu_seqlens_q_list[i+1] - cu_seqlens_q_list[i] for i in range(len(cu_seqlens_q_list) - 1)]
        else:
            lengths_q_list = actual_seq_lengths_query.tolist()

        if cu_seqlens_k_list is not None:
            lengths_k_list = [cu_seqlens_k_list[i+1] - cu_seqlens_k_list[i] for i in range(len(cu_seqlens_k_list) - 1)]
        else:
            lengths_k_list = actual_seq_lengths_key.tolist()

        # cmp_residual_k_for_cpu 还原为 list
        if cmp_residual_k_for_npu is not None:
            cmp_residual_k_list = cmp_residual_k_for_npu.tolist()
        else:
            cmp_residual_k_list = [0] * batch_size

        metadata = _load_metadata_from_cpu(
            batch_size=batch_size,
            cu_seqlens_q=cu_seqlens_q_list,
            cu_seqlens_k=cu_seqlens_k_list,
            seqused_q_list=lengths_q_list,
            seqused_k_list=lengths_k_list,
            cmp_residual_k_list=cmp_residual_k_list,
            max_seqlen_q=max_seqlen_q_meta,
            max_seqlen_k=max_seqlen_k_meta,
            q_head_num=q_head_num,
            k_head_num=k_head_num,
            head_dim=head_dim,
            topk=sparse_count,
            quant_mode=quant_mode,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )

    # ======================== tensor 搬运到 NPU（两种场景公共） ========================
    def log(tensor, name):
        if tensor is not None:
            print(name, tensor.shape, tensor.dtype)
            return tensor.npu()
        return None

    query = log(query, "query")
    key = log(key, "key")
    weights = log(weights, "weights")
    query_dequant_scale = log(query_dequant_scale, "query_dequant_scale")
    key_dequant_scale = log(key_dequant_scale, "key_dequant_scale")
    cu_seqlens_query = log(cu_seqlens_query, "cu_seqlens_query")
    cu_seqlens_key = log(cu_seqlens_key, "cu_seqlens_key")
    # seqused_q/k: TND 布局下为 None（用 cu_seqlens），非 TND 用 actual_seq_lengths
    if layout_query == "TND":
        seqused_q_tensor = None
    else:
        seqused_q_tensor = log(actual_seq_lengths_query, "seqused_q")
    if layout_key == "TND":
        seqused_k_tensor = None
    else:
        seqused_k_tensor = log(actual_seq_lengths_key, "seqused_k")
    cmp_residual_k_for_npu = log(cmp_residual_k_for_npu, "cmp_residual_k_for_npu")
    output_idx_offset = log(output_idx_offset, "output_idx_offset")
    block_table = log(block_table, "block_table")

    if IS_NPU_NOW:
        # ======================== 场景二：tensor 已在 NPU 上，调用真正的 NPU metadata 算子 ========================
        metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
                                        cu_seqlens_q = cu_seqlens_query,
                                        cu_seqlens_k = cu_seqlens_key,
                                        seqused_q = seqused_q_tensor,
                                        seqused_k = seqused_k_tensor,
                                        cmp_residual_k = cmp_residual_k_for_npu,
                                        batch_size = batch_size,
                                        max_seqlen_q = max_seqlen_q_meta,
                                        max_seqlen_k = max_seqlen_k_meta,
                                        num_heads_q = q_head_num,
                                        num_heads_k = k_head_num,
                                        head_dim = head_dim,
                                        topk = sparse_count,
                                        quant_mode = quant_mode,
                                        mask_mode = sparse_mode,
                                        layout_q = layout_query,
                                        layout_k = layout_key,
                                        cmp_ratio = cmp_ratio)
        metadata = metadata.npu()
    else:
        # 场景一：metadata 在 CPU 上生成，搬到 NPU
        metadata = log(metadata, "metadata")

    print("max_seqlen_q", max_seqlen_q)
    print("quant_mode", quant_mode)
    print("layout_query", layout_query)
    print("layout_key", layout_key)
    print("sparse_count", sparse_count)
    print("sparse_mode", sparse_mode)
    print("cmp_ratio", cmp_ratio)
    print("return_value", return_value)

    #调用QLI算子
    npu_result, npu_topk_value = torch.ops.cann_ops_transformer.quant_lightning_indexer(query, key, weights,
                                                    query_dequant_scale,
                                                    key_dequant_scale,
                                                    cu_seqlens_q = cu_seqlens_query,
                                                    cu_seqlens_k = cu_seqlens_key,
                                                    seqused_q = seqused_q_tensor,
                                                    seqused_k = seqused_k_tensor,
                                                    cmp_residual_k = cmp_residual_k_for_npu,
                                                    output_idx_offset = output_idx_offset,
                                                    max_seqlen_q = max_seqlen_q,
                                                    block_table = block_table,
                                                    metadata = metadata,
                                                    quant_mode = quant_mode,
                                                    layout_q = layout_query,
                                                    layout_k = layout_key,
                                                    topk = sparse_count,
                                                    mask_mode = sparse_mode,
                                                    cmp_ratio = cmp_ratio,
                                                    return_value = return_value)

    npu_result = npu_result.cpu()
    npu_topk_value = npu_topk_value.cpu()
    if output_idx_offset is not None:
        output_idx_offset = output_idx_offset.cpu()
    npu_result, _ = torch.sort(npu_result)
    cpu_result, _ = torch.sort(cpu_result)
    npu_topk_value, _ = npu_topk_value.sort(dim=-1, descending=True)

    torch.npu.synchronize()

    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value, params, output_idx_offset
