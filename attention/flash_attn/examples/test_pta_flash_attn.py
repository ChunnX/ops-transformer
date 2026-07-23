# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import math
import numpy as np
import cann_ops_transformer
from cann_ops_transformer.ops import npu_flash_attn
torch.manual_seed(42)

# B = 1
B_list = [1]
numHeads = 1
numKeyValueHeads = 1
# Sq = 1
Sq_list = [128]
Skv = 128
D = 64
# T = 4096
type=torch.float16

for B in B_list:
    for Sq in Sq_list:
        query = torch.randn(B, numHeads, Sq, D, dtype=type).npu()
        key = torch.randn(B, numKeyValueHeads, Skv, D, dtype=type).npu()
        value = torch.randn(B, numKeyValueHeads, Skv, D, dtype=type).npu()
        scale_value = 1/math.sqrt(float(D))

        actual_seq_lengths_kv = [Skv]*B
        attention_mask = torch.tril(torch.ones(2048,2048)).to(torch.bool).npu()

        for _ in range(1):
            out, _ = npu_flash_attn(
                    query, key, value,
                    block_table=None,
                    cu_seqlens_q=None,
                    cu_seqlens_kv=None,
                    seqused_q=None,
                    seqused_kv=None,
                    sinks=None,
                    metadata=None,
                    softmax_scale=scale_value,
                    mask_mode=0,
                    win_left=0,
                    win_right=0,
                    max_seqlen_q=0,
                    max_seqlen_kv=0,
                    layout_q = "BNSD",
                    layout_kv = "BNSD",
                    layout_out = "BNSD",
                    return_softmax_lse=0)
        out = out.cpu()

        print("************end*************", out, out.shape)