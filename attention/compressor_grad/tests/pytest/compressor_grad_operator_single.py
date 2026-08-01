# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import pytest
import random
import numpy as np
import math
import math, copy
import torch
import torch_npu
import torchair
import numpy as np
import torch.nn as nn

from compressor_grad_golden import run_compressor_grad_eager

class Generalized_operator():
    def forward(self,
                x,
                wkv,
                wgate,
                d_cmp_kv,
                softmax_score,
                kv,
                cmp_ratio,
                layout_x,
                layout_d_cmp_kv,
                layout_softmax_score,
                layout_kv,
                layout_d_x,
                cu_seqlens,
                seqused,
                start_pos,
                coff):

        return run_compressor_grad_eager(
            x, wkv, wgate, d_cmp_kv, softmax_score, kv, cmp_ratio,
            layout_x, layout_d_cmp_kv, layout_softmax_score, layout_kv, layout_d_x,
            cu_seqlens=cu_seqlens, seqused=seqused, start_pos=start_pos, coff=coff)

def output_operator(params):
    #构造输入
    batch_size, hidden_size, Seq_len, head_dim, block_size, cmp_ratio, coff, \
    layout_x, layout_d_cmp_kv, layout_softmax_score, layout_kv, layout_d_x, data_type, cu_seqlens, seqused, start_pos, \
    x_datarange, wkv_datarange,  wgate_datarange, d_cmp_kv_datarange, softmax_score_datarange, kv_datarange = params

    bs_combine_flag = False
    if layout_x == "TH":
        bs_combine_flag = True
        if cu_seqlens is None:
            T = batch_size * Seq_len
            if T !=0:
                cu_seqlens = torch.arange(0, T + 1, Seq_len, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((batch_size+1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(batch_size):
            if start_pos is not None:
                if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                    S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] 
            else:
                if cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                    S_max = cu_seqlens[i + 1] - cu_seqlens[i] 

    else:
        cu_seqlens = None
        if start_pos is not None:
            S_max = max(start_pos) + Seq_len
        else:
            S_max = Seq_len
    print(f"params = {params}")

    run_compressor_grad_eager(batch_size, S_max, head_dim, coff, cmp_ratio, bs_combine_flag, layout_x, layout_d_cmp_kv, layout_softmax_score, layout_kv, layout_d_x, \
    S = Seq_len, start_pos = start_pos, \
    seqused = seqused, cu_seqlens = cu_seqlens, block_size = block_size, data_type = data_type, \
    hidden_size = hidden_size, \
    x_datarange = x_datarange, wkv_datarange = wkv_datarange, wgate_datarange = wgate_datarange, d_cmp_kv_datarange = d_cmp_kv_datarange, \
    softmax_score_datarange = softmax_score_datarange, kv_datarange = kv_datarange)