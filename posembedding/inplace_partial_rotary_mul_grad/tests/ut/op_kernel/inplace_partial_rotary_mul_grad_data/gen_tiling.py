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
import numpy as np
import sys

# Tiling data format: b, s, n, d, partialStart, sliceLen, rotaryMode, dSplitCoef, dyDtypeCode, cosDtypeCode,
#                      cosStrideB, cosStrideS, cosStrideN, usedCoreNum, blockFactorM, ubFactorM, isFullBroadcast
# Test case: B=1, S=1, N=1, D=128, fp32, mode=1 (interleave), partialSlice=[0,128]
case0_params = [1, 1, 1, 128,    # b, s, n, d
                0, 128,           # partialStart, sliceLen
                1,                # rotaryMode=interleave
                1,                # dSplitCoef=1
                0, 0,             # dyDtypeCode=fp32, cosDtypeCode=fp32
                0, 0, 0,          # cosStrideB,S,N (full broadcast: 111)
                1,                # usedCoreNum
                1,                # blockFactorM
                1,                # ubFactorM
                1]                # isFullBroadcast

# Test case: B=2, S=4, N=8, D=64, fp16, mode=0 (half), partialSlice=[0,64], cos=(1,4,1,64)
case1_params = [2, 4, 8, 64,     # b, s, n, d
                0, 64,            # partialStart, sliceLen
                0,                # rotaryMode=half
                2,                # dSplitCoef=2
                1, 1,             # dyDtypeCode=fp16, cosDtypeCode=fp16
                0, 1, 0,          # cosStrideB=0 (cosB=1), S=1 (cosS=4→N=1→stride=1), N=0 (cosN=1)
                32,               # usedCoreNum
                2,                # blockFactorM
                2,                # ubFactorM
                0]                # isFullBroadcast

params_info = {
    "case0": case0_params,
    "case1": case1_params,
}


def main():
    params_list = params_info[sys.argv[1]]
    base_params = np.array(params_list, dtype=np.int64)
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
