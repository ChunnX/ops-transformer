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

struct InplacePartialRotaryMulGradUTTilingData {
    int64_t b;
    int64_t s;
    int64_t n;
    int64_t d;
    int64_t partialStart;
    int64_t sliceLen;
    int64_t rotaryMode;
    int64_t dSplitCoef;
    int64_t dyDtypeCode;
    int64_t cosDtypeCode;
    int64_t cosStrideB;
    int64_t cosStrideS;
    int64_t cosStrideN;
    int64_t usedCoreNum;
    int64_t blockFactorM;
    int64_t ubFactorM;
    int64_t isFullBroadcast;
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
