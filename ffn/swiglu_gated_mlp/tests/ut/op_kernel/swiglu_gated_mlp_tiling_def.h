/**
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SWIGLU_GATED_MLP_TILING_DEF_H
#define TEST_SWIGLU_GATED_MLP_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "tiling/platform/platform_ascendc.h"
#include "glu_tiling.hpp"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

inline void InitSwigluGatedMlpTilingData(uint8_t *tiling, SwigluGatedMlpTilingData *tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(SwigluGatedMlpTilingData));
}

#ifdef GET_TILING_DATA
#undef GET_TILING_DATA
#endif

#define GET_TILING_DATA(tilingData, tilingArg)                                      \
    SwigluGatedMlpTilingData tilingData;                                            \
    InitSwigluGatedMlpTilingData(reinterpret_cast<uint8_t *>(tilingArg), &tilingData)

#ifdef GET_TILING_DATA_WITH_STRUCT
#undef GET_TILING_DATA_WITH_STRUCT
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg)            \
    tilingStruct tilingData;                                                        \
    InitSwigluGatedMlpTilingData(reinterpret_cast<uint8_t *>(tilingArg), &tilingData)

#endif  // TEST_SWIGLU_GATED_MLP_TILING_DEF_H
