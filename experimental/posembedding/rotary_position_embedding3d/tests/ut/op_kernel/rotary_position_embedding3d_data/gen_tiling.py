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
"""
生成 rotary_position_embedding3d 的 tiling 二进制数据。

Tiling 数据结构（每个字段均为 int64，共 24 字节）：
    totalLength: 总数据量 = B * S * D
    lastDim:     最后一维大小 = D
    tileNum:     tile 数量 = 8（常量）

用例说明：
    case_2_4_8:     B=2, S=4, D=8   → totalLength=64
    case_4_16_64:   B=4, S=16, D=64 → totalLength=4096
    case_1_32_128:  B=1, S=32, D=128 → totalLength=4096
"""

import numpy as np
import sys

TILE_NUM = 8  # 来自 rotary_position_embedding3d_tiling.cpp

# 各 case 对应的 (B, S, D) 参数
case_configs = {
    "case_2_4_8":    (2, 4, 8),
    "case_4_16_64":  (4, 16, 64),
    "case_1_32_128": (1, 32, 128),
}


def gen_tiling(b, s, d):
    total_length = b * s * d
    last_dim = d
    params = np.array([total_length, last_dim, TILE_NUM], dtype=np.int64)
    with open("tiling.bin", "wb") as f:
        params.tofile(f)


def main():
    case_name = sys.argv[1]  # e.g. "case_2_4_8"
    if case_name not in case_configs:
        print(f"Unknown case: {case_name}, available: {list(case_configs.keys())}")
        sys.exit(1)
    b, s, d = case_configs[case_name]
    gen_tiling(b, s, d)


if __name__ == "__main__":
    main()
