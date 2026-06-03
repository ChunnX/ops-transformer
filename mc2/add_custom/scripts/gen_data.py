#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import os
import numpy as np


def gen_golden_data_simple():
    shape = (8, 2048)
    input_x = np.random.uniform(1, 10, shape).astype(np.float16)
    input_y = np.random.uniform(1, 10, shape).astype(np.float16)
    golden = (input_x.astype(np.float32) + input_y.astype(np.float32)).astype(np.float16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

    print(f"[INFO] Generated input_x, input_y, golden (shape={shape}, dtype=float16)")
    print(f"[INFO] input_x[:5]={input_x.ravel()[:5]}")
    print(f"[INFO] input_y[:5]={input_y.ravel()[:5]}")
    print(f"[INFO] golden[:5] ={golden.ravel()[:5]}")


if __name__ == "__main__":
    gen_golden_data_simple()
