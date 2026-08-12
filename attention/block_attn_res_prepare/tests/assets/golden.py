#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch


__golden__ = {
    "kernel": {"block_attn_res_prepare": "block_attn_res_prepare_golden"},
}


def block_attn_res_prepare_golden(
    block_residual, effective_query, valid_blocks, epsilon=1.0e-6, **kwargs
):
    """FP32 Phase 1 reference for BlockAttnResPrepare."""
    inputs_are_numpy = not hasattr(block_residual, "detach")
    residual = torch.as_tensor(block_residual, dtype=torch.float32)
    query = torch.as_tensor(effective_query, dtype=torch.float32)
    valid = int(torch.as_tensor(valid_blocks, dtype=torch.int32).reshape(-1)[0].item())
    valid = max(1, min(valid, residual.shape[1]))
    history = residual[:, :valid, :]
    inv_rms = torch.rsqrt(torch.mean(history * history, dim=-1) + float(epsilon))
    logits = torch.einsum("sd,tnd->stn", query, history) * inv_rms.unsqueeze(0)
    softmax_max = torch.max(logits, dim=-1).values
    weights = torch.exp(logits - softmax_max.unsqueeze(-1))
    softmax_sum = torch.sum(weights, dim=-1)
    weighted_output = torch.einsum("stn,tnd->std", weights, history)
    outputs = (
        softmax_max.to(torch.float32),
        weighted_output.to(torch.float32),
        softmax_sum.to(torch.float32),
    )
    if inputs_are_numpy:
        return tuple(output.cpu().numpy().astype(np.float32) for output in outputs)
    return outputs
