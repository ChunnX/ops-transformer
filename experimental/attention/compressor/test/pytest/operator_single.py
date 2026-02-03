# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import test
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math
import math, copy
import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn

from cpu_compressor import cpu_compressor, get_seq_used_by_batch

class Generalized_operator():
    def forward(self,
                x,
                wkv,
                wgate,
                kv_state,
                score_state,
                ape,
                norm_weight, 
                rope_sin,
                rope_cos,
                block_table,
                cu_seqlens,
                seqused,
                start_pos,
                rope_head_dim,
                cmp_ratio,
                coff,
                norm_eps,
                rotary_mode):
        return cpu_compressor(
            x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos,
            block_table=block_table, cu_seqlens=cu_seqlens, seqused=seqused, start_pos=start_pos,
            rope_head_dim=rope_head_dim, cmp_ratio=cmp_ratio, coff=coff, norm_eps=norm_eps, rotary_mode=rotary_mode)

def output_operator(params):
    #构造输入
    batch_size, hidden_size, Seq_len, head_dim, block_size, rope_head_dim, cmp_ratio, coff, norm_eps, \
    start_p, rotary_mode, layout_x, data_type, cu_seqlens, seqused, start_pos = params

    if layout_x == "TH":
        if cu_seqlens is not None:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        else:
            T = batch_size * Seq_len
            cu_seqlens = torch.arange(0, T + 1, Seq_len, dtype=torch.int32)
        
        if seqused is not None:
            seqused = torch.tensor(seqused).to(torch.int32)
            S_max = max(seqused)
        else:
            S_max = 0
            for i in range(1, batch_size + 1):
                if S_max < cu_seqlens[i] - cu_seqlens[i - 1]:
                    S_max = cu_seqlens[i] - cu_seqlens[i - 1] 

        if start_pos is not None:
            start_pos = torch.tensor(start_pos).to(torch.int32)
        else:
            start_pos = [0] * batch_size
    else:
        cu_seqlens = None
        if start_pos == None:
            start_pos = [0] * batch_size
        else:
            start_pos = torch.tensor(start_pos).to(torch.int32)
        S_max = max(start_pos) + Seq_len

        if seqused is not None:
            seqused = torch.tensor(seqused).to(torch.int32)

    ### ======================== check input params start ========================
    print(f"params = {params}")

    actseqs = []
    if seqused is not None:
        actseqs = seqused
    else:
        if cu_seqlens is not None:
            for i in range(len(cu_seqlens) - 1):
                diff = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
                actseqs.append(diff)
        else:
            actseqs = [Seq_len] * batch_size

    max_block_num_per_batch = (S_max + cmp_ratio + block_size - 1) // block_size
    block_num = 0
    for i in range(batch_size):
        block_num += math.ceil((int(actseqs[i]) + int(start_pos[i])) / block_size)
    block_num = block_num - 1
    shuffled_indices = torch.randperm(block_num)
    index = torch.arange(1, block_num + 1, 1, dtype=torch.int32)
    index = index[shuffled_indices]
    index_id = 0
    block_table = torch.zeros(size=(batch_size, max_block_num_per_batch), dtype=torch.int32)
    for i in range(batch_size):
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0: #如果start_pos[i]正好被整除，这个batch的本次需要处理的第一个r的数据都通过x传入了，不需要从state中取，不需要预留空间
            cur_end = start_pos[i]
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            if index_id < block_num:
                block_table[i][j] = index[index_id]
                index_id += 1
        end_pos = get_seq_used_by_batch(i, Seq_len, seqused, cu_seqlens)
        next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio 
        if (start_pos[i] + end_pos) % cmp_ratio == 0:
            next_end = start_pos[i] + end_pos       #如果start_pos[i] + end_pos正好被整除，下一个r的数据一个都没有给，所以不需要给下一个r预留空间
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            if block_table[i][j] == 0 and index_id < block_num:
                block_table[i][j] = index[index_id]
                index_id += 1
    kv_state = torch.tensor(np.random.uniform(-10, 10, (block_num + 1, block_size, coff * head_dim))).to(torch.float32)
    score_state = torch.tensor(np.random.uniform(-10, 10, (block_num + 1, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if layout_x == "TH":
        x_shape = (cu_seqlens[-1], hidden_size)
        rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + batch_size), rope_head_dim)
        rope_cos_shape = rope_sin_shape
    else:
        x_shape = (batch_size, Seq_len, hidden_size)
        rope_sin_shape = (batch_size, (Seq_len + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
        rope_cos_shape = rope_sin_shape

    x = torch.tensor(np.random.uniform(-10, 10, x_shape)).to(data_type)
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type)
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type)
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32)
    norm_weight = torch.tensor(np.random.uniform(-10, 10, (head_dim))).to(data_type)
    rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type)
    rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type)
    ### ======================== gen input data finish =============================
    ### ======================== execute cpu start =================================
    cpu_kv_state = kv_state.clone()
    cpu_score_state = score_state.clone()

    test_operator = Generalized_operator()
    cpu_result, kv_mask_result = test_operator.forward( x,
                                        wkv,
                                        wgate,
                                        cpu_kv_state,
                                        cpu_score_state,
                                        ape,
                                        norm_weight, 
                                        rope_sin,
                                        rope_cos,
                                        block_table = block_table,
                                        cu_seqlens = cu_seqlens,
                                        seqused = seqused,
                                        start_pos = start_pos,
                                        rope_head_dim = rope_head_dim,
                                        cmp_ratio = cmp_ratio,
                                        coff = coff,
                                        norm_eps = norm_eps,
                                        rotary_mode = rotary_mode)
    update_kv = cpu_kv_state != kv_state
    update_score = cpu_score_state != score_state
    
    npu_kv_state = kv_state.npu()
    npu_score_state = score_state.npu()
    if cu_seqlens is not None: 
        cu_seqlens = cu_seqlens.npu()
    if seqused is not None: 
        seqused = seqused.npu()
    if start_pos is not None: 
        start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if block_table is not None: 
        kv_block_table = block_table.npu()
        score_block_table = block_table.npu()

    npu_result, _, _, _, _ = torch.ops.custom.compressor(
                x.npu(),
                wkv.npu(),
                wgate.npu(),
                npu_kv_state,
                npu_score_state,
                ape.npu(),
                norm_weight.npu(), 
                rope_sin.npu(),
                rope_cos.npu(),
                kv_block_table = kv_block_table,
                score_block_table = score_block_table,
                cu_seqlens = cu_seqlens,
                seqused = seqused,
                start_pos = start_pos,
                rope_head_dim = rope_head_dim,
                cmp_ratio = cmp_ratio,
                coff = coff,
                norm_eps = norm_eps,
                rotary_mode = rotary_mode
            )
    return cpu_result, kv_mask_result, npu_result ,cpu_kv_state, npu_kv_state, update_kv, cpu_score_state, npu_score_state, update_score