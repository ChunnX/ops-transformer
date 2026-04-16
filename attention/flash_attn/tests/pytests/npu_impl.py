#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import torch_npu
import math
import numpy as np
import random
from einops import rearrange
import npu_ops_transformer
from npu_ops_transformer.ops import npu_flash_attn
from test_utils import trans_bnsd_to_layout

device_id = 0
device = torch.device(f"npu:{device_id}")

def flash_attn_npu(q, k, v, q_rope, k_rope, atten_mask, pse, **kwargs):
    input_layout = kwargs.get("input_layout", "BNSD")
    n1 = kwargs.get("N1")
    d = kwargs.get("D")
    d_rope = kwargs.get("DRope", 0)
    sparse_mode = kwargs.get("sparse_mode", 0)
    prefix = kwargs.get("prefix", None)
    pre_tokens = kwargs.get("pre_tokens", 2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    actual_seq_qlen = kwargs.get("actual_seq_qlen", None)
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen", actual_seq_qlen)
    scale = kwargs.get("scale", 1 / (d ** 0.5))
    pse_type = kwargs.get("pse_type", 1)
    keep_prob = kwargs.get("keep_prob", 1)
    q_start_idx = kwargs.get("q_start_idx", None)
    kv_start_idx = kwargs.get("kv_start_idx", None)
    seed = kwargs.get("seed", 0)

    q1 = trans_bnsd_to_layout(q, input_layout).contiguous().to(device)
    k1 = trans_bnsd_to_layout(k, input_layout).contiguous().to(device)
    v1 = trans_bnsd_to_layout(v, input_layout).contiguous().to(device)

    if d_rope != 0:
        query_rope = trans_bnsd_to_layout(q_rope, input_layout).contiguous().to(device)
        key_rope = trans_bnsd_to_layout(k_rope, input_layout).contiguous().to(device)
    else:
        query_rope = None
        key_rope = None
    if pse is not None:
        pse1 = pse.to(device)
    else:
        pse1 = None

    if atten_mask is not None:
        atten_mask1 = atten_mask.to(device)
    else:
        atten_mask1 = None

    torch.npu.manual_seed(seed)

    out, _ = npu_flash_attn(
        q1, k1, v1,
        cu_seqlens_q = sum(actual_seq_qlen) if actual_seq_qlen is not None else None,
        cu_seqlens_kv = sum(actual_seq_kvlen) if actual_seq_kvlen is not None else None,
        seqused_q = actual_seq_qlen,
        seqused_kv = actual_seq_kvlen,
        softmax_scale = scale,
        mask_mode = sparse_mode,
        layout_q = input_layout,
        layout_kv = input_layout,
        layout_out = input_layout
        )
    torch.npu.synchronize()

    npu_out = out.cpu()
    return npu_out