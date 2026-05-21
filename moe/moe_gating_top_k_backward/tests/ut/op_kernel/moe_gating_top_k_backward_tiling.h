/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_backward_tiling.h
 * \brief
 */

#ifndef __MOE_GATING_TOP_K_BACKWARD_TILING_H__
#define __MOE_GATING_TOP_K_BACKWARD_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MoeGatingTopKBackwardTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreRows = 0;
    int64_t lastCoreRows = 0;
    int64_t baseRows = 0;
    int64_t perLoopTimes = 0;
    int64_t perTailRows = 0;
    int64_t lastLoopTimes = 0;
    int64_t lastTailRows = 0;
    int64_t tokenCount = 0;
    int64_t expertCount = 0;
    int64_t k = 0;
    int64_t gradYDtypeSize = 0;
    int64_t renorm = 1;
    int64_t normType = 1;
    float routedScalingFactor = 1.0f;
    float eps = 1e-20f;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeGatingTopKBackwardTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(MoeGatingTopKBackwardTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeGatingTopKBackwardTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeGatingTopKBackwardTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeGatingTopKBackwardTilingData tiling_data;         \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_GRAD_Y float

#endif