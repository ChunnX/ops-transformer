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
import random
import numpy as np
import math
import ctypes
import copy
import torchair
import torch.nn as nn
from torchair.configs.compiler_config import CompilerConfig
from lightning_indexer_v2_golden import GeneralizedLIV2

class LIV2Network(nn.Module):
    def __init__(self):
        super(LIV2Network, self).__init__()

    def forward(self, query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table,
                output_idx_offset, metadata, topk, max_seqlen_q, layout_q, layout_k, mask_mode, cmp_ratio, return_value):
        return torch.ops.cann_ops_transformer.lightning_indexer(query, key, weights,
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
                                                                layout_q = layout_q,
                                                                layout_k = layout_k,
                                                                mask_mode = mask_mode,
                                                                cmp_ratio = cmp_ratio,
                                                                return_value = return_value)

def liv2_output_acl_graph(params):
    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, output_idx_offset, \
    layout_query, layout_key, topk, mask_mode, query_datarange, key_datarange, weights_datarange, \
    cmp_ratio, return_value, max_seqlen_q = params

    test_liv2 = GeneralizedLIV2(batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num,
                                head_dim, block_size, block_num, qk_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
                                cmp_residual_k, layout_query, layout_key, topk, max_seqlen_q, mask_mode, cmp_ratio, return_value)

    if cu_seqlens_q is not None:
        cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32).npu()
    if cu_seqlens_k is not None:
        cu_seqlens_k = torch.tensor(cu_seqlens_k).to(torch.int32).npu()
    if seqused_q is not None:
        seqused_q = torch.tensor(seqused_q).to(torch.int32).npu()
    if seqused_k is not None:
        seqused_k = torch.tensor(seqused_k).to(torch.int32).npu()
    if cmp_residual_k is not None:
        cmp_residual_k = torch.tensor(cmp_residual_k).to(torch.int32).npu()

    if layout_query == "BSND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (batch_size, q_seq, q_head_num))).to(torch.float32).npu()
        if output_idx_offset is not None:
            output_idx_offset = torch.tensor(output_idx_offset).reshape(batch_size, q_seq, 1).to(torch.int32).npu()
    elif layout_query == "TND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_t_size, q_head_num, head_dim))).to(qk_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num))).to(torch.float32).npu()
        if output_idx_offset is not None:
            output_idx_offset = torch.tensor(output_idx_offset).reshape(q_t_size, 1).to(torch.int32).npu()

    if layout_key == "BSND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (batch_size, k_seq, k_head_num, head_dim))).to(qk_dtype).npu()
        block_table = None
        cpu_result, topk_value, cpu_topk_value = test_liv2.forward(query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset)

    elif layout_key == "TND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (k_t_size, k_head_num, head_dim))).to(qk_dtype).npu()
        block_table = None
        cpu_result, topk_value, cpu_topk_value = test_liv2.forward(query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset)

    elif layout_key == "PA_BBND":
        # 以不同batch中最大seq为标准初始化key(bnsd)
        k_max_s2 = math.floor(max(seqused_k))
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num

        key_bnsd = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1],(batch_size, k_head_num, k_max_s2, head_dim))).to(qk_dtype)
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in seqused_k:
            cur_cmp_act_k = math.floor(cur_act_k)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError(f"key actual block num < needed block num")
        # 构建block table
        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table = np.full((batch_size, k_max_block_num_per_batch), fill_value = -1, dtype=np.int32)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1
        # 构建PA场景的key
        # [batch_size, s2, k_head_num, head_dim] expand to [batch_size, k_max_block_num_per_batch * block_size, k_head_num, head_dim]
        key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype = qk_dtype)
        key_expand[:,:,:k_max_s2,:] = key_bnsd
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype = qk_dtype)
        for i_batch in range(batch_size):
            for  i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size,:]
        # kv_cache 0轴非连续：将key和key_dequant_scale融合到blockFusion (ref v1 commit keyStride0)
        properties = torch.npu.get_device_properties()
        if "Ascend950" in properties.name:
            key_stride = 10  # 0轴非连续增加stride
            bytes_per_token = head_dim + key_stride # 整个非连续的长度
            blockFusion = torch.zeros((block_num, block_size * k_head_num * bytes_per_token), dtype=qk_dtype)
            key_flat = key.view(block_num, block_size * k_head_num * head_dim)
            blockFusion[:, :block_size * k_head_num * head_dim] = key_flat
            blockFusion = blockFusion.npu()
            key = blockFusion[:, :block_size * k_head_num * head_dim].view(block_num, block_size, k_head_num, head_dim)

        key = key.npu()
        cpu_result, topk_value, cpu_topk_value = test_liv2.forward(query, key_bnsd, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset)
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()
    
    if layout_query == "TND":
        if seqused_q is not None:
            max_seqlen_q = max(seqused_q).item()
        else:
            seqlen = []
            for b_idx in range(batch_size):
                seqlen.append(cu_seqlens_q[b_idx + 1] - cu_seqlens_q[b_idx])
            max_seqlen_q = max(seqlen).item()
    else:
        if seqused_q is not None:
            max_seqlen_q = max(seqused_q).item()
        else:
            max_seqlen_q = q_seq
    
    if layout_key == "TND":
        if seqused_k is not None:
            max_seqlen_k = max(seqused_k).item()
        else:
            seqlen = []
            for b_idx in range(batch_size):
                seqlen.append(cu_seqlens_q[b_idx + 1] - cu_seqlens_k[b_idx])
            max_seqlen_k = max(seqlen).item()
    elif layout_key == "PA_BBND":
        max_seqlen_k = max(seqused_k).item()
    else:
        if seqused_k is not None:
            max_seqlen_k = max(seqused_k).item()
        else:
            max_seqlen_k = k_seq
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
                                num_heads_q = q_head_num,
                                num_heads_k = k_head_num,
                                head_dim = head_dim,
                                topk = topk,
                                cu_seqlens_q = cu_seqlens_q,
                                cu_seqlens_k = cu_seqlens_k,
                                seqused_q = seqused_q,
                                seqused_k = seqused_k,
                                cmp_residual_k = cmp_residual_k,
                                batch_size = batch_size,
                                max_seqlen_q = max_seqlen_q,
                                max_seqlen_k = max_seqlen_k,
                                layout_q = layout_query,
                                layout_k = layout_key,
                                mask_mode = mask_mode,
                                cmp_ratio = cmp_ratio)

    metadata = metadata.npu()

    print("run acl_graph:")
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    config.mode = "reduce-overhead"
    npu_mode = torch.compile(LIV2Network().npu(), fullgraph=True, backend=npu_backend, dynamic=False)

    npu_result, npu_topk_value = npu_mode(query, key, weights,
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

    npu_topk_value, _ = npu_topk_value.sort(dim=-1, descending=True)
    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value
