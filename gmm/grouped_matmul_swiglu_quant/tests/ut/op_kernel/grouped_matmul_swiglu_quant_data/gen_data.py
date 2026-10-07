#!/usr/bin/python3
# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import os
import numpy as np
import random
np.random.seed(0)

def gen_input_data(group_num, e, m, k, n):
    e = int(e)
    m = int(m)
    k = int(k)
    n = int(n)
    group_num = int(group_num)
    x = np.random.uniform(-5, 5, [m, k]).astype(np.int8)
    weight = np.random.uniform(-5, 5, [e, k, n]).astype(np.int8)
    xScale = np.random.uniform(-5, 5, [m,]).astype(np.float32)
    weightScale = np.random.uniform(-5, 5, [e, n]).astype(np.float32)
    groupedList = np.linspace(1, m, group_num, dtype=np.int64)
    return x, weight, xScale, weightScale, groupedList

def swiglu_quant_per_group(x, weight, weightScale, xScale):
    # 反量化 -> SwiGLU -> 重量化, 与 ST 标杆
    # tests/st/aclnnGroupedMatmulSwigluQuant/executor_aclnnGroupedMatmulSwigluQuant.py
    # 里 GMM_Swiglu_quant 的计算顺序保持一致
    res = np.matmul(x.astype(np.int32), weight.astype(np.int32)).astype(np.float32)
    res = res * weightScale
    res = res * xScale.reshape(-1, 1)
    half = res.shape[-1] // 2
    activated, gate = res[:, :half], res[:, half:]
    with np.errstate(over="ignore"):
        sigmoid = 1.0 / (1.0 + np.exp(-activated))
    merged = activated * sigmoid * gate
    amax = np.max(np.abs(merged), axis=-1)
    quantScale = 127.0 / amax
    y = np.round(merged * quantScale.reshape(-1, 1)).astype(np.int8)
    yScale = (1.0 / quantScale).astype(np.float32)
    return y, yScale

def gen_golden_data(group_num, e, m, k, n):
    e = int(e)
    m = int(m)
    n = int(n)
    x, weight, xScale, weightScale, groupedList = gen_input_data(group_num, e, m, k, n)
    # y 是 (m, n / 2) 的 int8, y_scale 是长度为 m 的 float, 与 def.cpp 的输出声明一致
    y = np.zeros([m, n // 2], dtype=np.int8)
    yScale = np.zeros([m], dtype=np.float32)
    start = 0
    preV = 0
    for group, currV in enumerate(groupedList.tolist()):
        num = currV - preV
        preV = currV
        if num > 0:
            y[start:start + num], yScale[start:start + num] = swiglu_quant_per_group(
                x[start:start + num], weight[group], weightScale[group], xScale[start:start + num])
        start += num
    x.tofile(f"x.bin")
    weight.tofile(f"weight.bin")
    weightScale.tofile(f"weightScale.bin")
    xScale.tofile(f"xScale.bin")
    groupedList.tofile(f"groupedList.bin")
    y.tofile(f"y.bin")
    yScale.tofile(f"y_scale.bin")

if __name__ == "__main__": 
    os.system("rm -rf *.bin")
    gen_golden_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
