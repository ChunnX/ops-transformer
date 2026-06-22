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

# Binary layout matches InplacePartialRotaryMulGradTilingData (296 bytes):
#   [0..7]     sliceLength              (int64_t)
#   [8..159]   params (19 int64_t)       InplacePartialRotaryMulGradRegbaseParams
#   [160..295] abParams (17 int64_t)     InplacePartialRotaryMulGradRegbaseABParams
#
# params layout (offset 8):
#   b, s, d, n, blockNumB, blockFactorB, blockNumS, blockFactorS,
#   ubFactorS, ubFactorB, ubLoopNumN, ubFactorN, ubTailFactorN,
#   usedCoreNum, rotaryMode, sliceStart, sliceEnd, sliceLength, dSplitCoef
#
# abParams layout (offset 160):
#   b, s, d, n, dAlign, dSplitCoef, blockNumBS, blockFactorBS, blockTailBS,
#   blockNumN, blockFactorN, blockTailN, ubFactorBS, ubFactorN,
#   usedCoreNum, rotaryMode, sliceStart, sliceEnd, sliceLength

def make_tiling_data(slice_length, params_tuple, ab_params_tuple):
    """Build the full 296-byte tiling data array."""
    data = np.zeros(37, dtype=np.int64)  # 1 + 19 + 17 = 37 int64_t = 296 bytes
    data[0] = slice_length

    # params at offset 8 (index 1..19)
    for i, val in enumerate(params_tuple):
        data[1 + i] = val

    # abParams at offset 160 (index 20..36)
    for i, val in enumerate(ab_params_tuple):
        data[20 + i] = val

    return data


# ===========================================================================
# case0: B=1, S=1, N=1, D=128, fp32, mode=1 (interleave), partialSlice=[0,128]
# Layout: NO_BROADCAST → AAndB → TILING_KEY_A (205)
# This template reads from params.
# ===========================================================================
# Tiny problem (1×1×1×128), 48 cores, most factors = 1
case0_slice_len = 128
case0_params = (
    1,    # b
    1,    # s
    128,  # d
    1,    # n
    1,    # blockNumB
    1,    # blockFactorB
    1,    # blockNumS
    1,    # blockFactorS
    1,    # ubFactorS
    1,    # ubFactorB
    1,    # ubLoopNumN
    1,    # ubFactorN
    1,    # ubTailFactorN
    1,    # usedCoreNum
    1,    # rotaryMode (interleave)
    0,    # sliceStart
    128,  # sliceEnd
    128,  # sliceLength
    1,    # dSplitCoef (interleave uses 1, halves handled in VF)
)
case0_ab_params = (0,) * 17  # unused for this template; zero-fill

case0_data = make_tiling_data(case0_slice_len, case0_params, case0_ab_params)

# ===========================================================================
# case1: B=2, S=4, N=8, D=64, fp16, mode=0 (half), partialSlice=[0,64]
# cos=(1,4,1,64) → SBND layout → AB template → TILING_KEY_AB (204)
# This template reads from abParams.
# ===========================================================================
case1_slice_len = 64
case1_params = (0,) * 19  # unused for AB template; zero-fill
case1_ab_params = (
    2,    # b
    4,    # s
    64,   # d
    8,    # n
    64,   # dAlign = Ceil(64/2, 32/2) * 2 = 64 (maxTypeSize=2, BLOCK_TYPE_SIZE=32)
    2,    # dSplitCoef (half mode)
    4,    # blockNumBS
    2,    # blockFactorBS
    2,    # blockTailBS (= 8 - 3*2)
    6,    # blockNumN
    2,    # blockFactorN
    2,    # blockTailN (= 8 - 3*2)
    2,    # ubFactorBS
    2,    # ubFactorN
    24,   # usedCoreNum (= 4 * 6)
    0,    # rotaryMode (half)
    0,    # sliceStart
    64,   # sliceEnd
    64,   # sliceLength
)
case1_data = make_tiling_data(case1_slice_len, case1_params, case1_ab_params)

params_info = {
    "case0": case0_data,
    "case1": case1_data,
}


def main():
    params_list = params_info[sys.argv[1]]
    tiling_file = open("tiling.bin", "wb")
    params_list.tofile(tiling_file)


if __name__ == '__main__':
    main()
