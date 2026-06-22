/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_
#define TEST_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

// Mirrors InplacePartialRotaryMulGradTilingData layout from
// op_kernel/arch35/inplace_partial_rotary_mul_grad_tiling_data.h.
//
// TEST STRUCT LAYOUT (must match byte-for-byte with the real struct):
//   [0..7]    sliceLength              (int64_t, offset 0)
//   [8..159]  params (19 int64_t)       (offset 8)
//   [160..295] abParams (17 int64_t)    (offset 160)
//   Total: 296 bytes

struct InplacePartialRotaryMulGradRegbaseParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t blockNumB;
    int64_t blockFactorB;
    int64_t blockNumS;
    int64_t blockFactorS;
    int64_t ubFactorS;
    int64_t ubFactorB;
    int64_t ubLoopNumN;
    int64_t ubFactorN;
    int64_t ubTailFactorN;
    int64_t usedCoreNum;
    int64_t rotaryMode;
    int64_t sliceStart;
    int64_t sliceEnd;
    int64_t sliceLength;
    int64_t dSplitCoef;
};

struct InplacePartialRotaryMulGradRegbaseABParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t dAlign;
    int64_t dSplitCoef;
    int64_t blockNumBS;
    int64_t blockFactorBS;
    int64_t blockTailBS;
    int64_t blockNumN;
    int64_t blockFactorN;
    int64_t blockTailN;
    int64_t ubFactorBS;
    int64_t ubFactorN;
    int64_t usedCoreNum;
    int64_t rotaryMode;
    int64_t sliceStart;
    int64_t sliceEnd;
    int64_t sliceLength;
};

struct InplacePartialRotaryMulGradUTTilingData {
    int64_t sliceLength;                                         // offset 0
    InplacePartialRotaryMulGradRegbaseParams params;             // offset 8
    InplacePartialRotaryMulGradRegbaseABParams abParams;         // offset 160
};

inline void InitInplacePartialRotaryMulGradTilingData(
    uint8_t* tiling, InplacePartialRotaryMulGradUTTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(InplacePartialRotaryMulGradUTTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingDataType, tilingData, tilingPointer) \
    tilingDataType tilingData; \
    InitInplacePartialRotaryMulGradTilingData(tilingPointer, &tilingData)

#endif // TEST_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_
