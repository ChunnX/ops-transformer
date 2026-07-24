#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {"kernel": {"matmul_reduce_scatter_v2": "matmul_reduce_scatter_v2_golden"}}

import numpy as np
import torch


def matmul_reduce_scatter_v2_golden(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    quant_scale=None,
    group="",
    reduce_op="sum",
    is_trans_a=False,
    is_trans_b=False,
    comm_turn=0,
    rank_size=0,
    block_size=0,
    group_size=0,
    is_amax_out=False,
    y_dtype=0,
    comm_mode="aiv",
    **kwargs,
):
    x1_dtype = kwargs.get("x1_dtype", "bf16")
    x2_dtype = kwargs.get("x2_dtype", "bf16")
    is_quant = x1_dtype not in ["fp16", "bf16"]

    x1 = torch.from_numpy(x1.astype(np.float32))
    x2 = torch.from_numpy(x2.astype(np.float32))
    if bias is not None:
        bias = torch.from_numpy(bias.astype(np.float32))

    if is_trans_b:
        x2 = x2.transpose()

    if not is_quant:
        output = torch.matmul(x1, x2)
        if bias is not None:
            output += bias
    else:
        if x1_scale is not None:
            x1_scale = torch.from_numpy(x1_scale.astype(np.float32))
        if x2_scale is not None:
            x2_scale = torch.from_numpy(x2_scale.astype(np.float32))

        output = torch.matmul(x1, x2)
        if bias is not None:
            output += bias
        if x1_scale is not None and x2_scale is not None:
            x1scale_np = x1_scale.numpy()
            x2scale_np = x2_scale.numpy()
            double_scale = scale_generate(x1scale_np * x2scale_np)
            double_scale_tensor = torch.unsqueeze(
                torch.from_numpy(double_scale), dim=1
            ).to(torch.float32)
            output *= double_scale_tensor

    output = reduce_scatter_compute(output, rank_size)
    return output.numpy()


def reduce_scatter_compute(output, rank_size):
    output_all = output.repeat(rank_size, 1)
    scatter_shape_m = output.shape[0] // rank_size
    scatter_output = output_all.narrow(0, 0, scatter_shape_m)
    return scatter_output


def scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32_deq_scale &= 0xFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
    return fp32_deq_scale
