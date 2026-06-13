#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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


def gen_data_and_golden(b, s, n, d, cos_b, cos_s, cos_n, cos_d, dtype_dy, dtype_cos):
    """Generate random input data.

    Args:
        b, s, n, d: dy tensor shape
        cos_b, cos_s, cos_n, cos_d: cos/sin tensor shape (may be broadcast)
        dtype_dy: dy data type string ("float16", "float32", "bfloat16")
        dtype_cos: cos/sin data type string
    """
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float16,  # approximate bf16 with fp16 for test data generation
    }
    dtype_q = dtype_map[dtype_dy]
    dtype_c = dtype_map[dtype_cos]

    data_dy = np.random.uniform(-2, 2, [int(b), int(s), int(n), int(d)]).astype(dtype_q)
    data_cos = np.random.uniform(-3, 4, [int(cos_b), int(cos_s), int(cos_n), int(cos_d)]).astype(dtype_c)
    data_sin = np.random.uniform(-4, 5, [int(cos_b), int(cos_s), int(cos_n), int(cos_d)]).astype(dtype_c)

    data_dy.tofile("./dy.bin")
    data_cos.tofile("./cos.bin")
    data_sin.tofile("./sin.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    # Args: b s n d cos_b cos_s cos_n cos_d dtype_dy dtype_cos
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                        sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8],
                        sys.argv[9], sys.argv[10])
    exit(0)
