#!/usr/bin/python3
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


def gen_data_and_golden(b, s, d, dtype_str):
    """
    生成 rotary_position_embedding3d 的随机测试数据。

    参数:
        b: batch 大小
        s: sequence 长度
        d: head dimension（必须为偶数）
    """
    shape = (int(b), int(s), int(d))
    half_d = int(d) // 2

    # x1: 输入数据，范围 [-2, 2]
    data_x = np.random.uniform(-2, 2, shape).astype(dtype_str)

    # x2: embeddings（左半 cos, 右半 sin），范围 [-1, 1]
    data_cos_sin = np.random.uniform(-1, 1, shape).astype(dtype_str)

    data_x.tofile("./x.bin")
    data_cos_sin.tofile("./cos_sin.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    exit(0)
