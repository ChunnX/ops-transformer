/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _UND_GEN_QKV_RMS_NORM_ROPE_CACHE_TILING_H_
#define _UND_GEN_QKV_RMS_NORM_ROPE_CACHE_TILING_H_

#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

#ifndef DTYPE_UND_QKV
#define DTYPE_UND_QKV bfloat16_t
#endif
#ifndef DTYPE_K_CACHE
#define DTYPE_K_CACHE bfloat16_t
#endif

// 字段顺序必须与 op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h 中的 TilingData 保持一致
struct UndGenQkvRmsNormRopeCacheTilingData {
    int64_t totalTokens;
    int64_t undLen;
    int64_t genLen;
    int64_t numHead;
    int64_t numHeadQ;
    int64_t numHeadK;
    int64_t numHeadV;
    int64_t headDim;
    int64_t maxPos;
    int64_t blockNum;
    int64_t blockSize;
    int64_t hasGen;
    int64_t hasCatIndices;
    int64_t mropeSectionT;
    int64_t mropeSectionH;
    int64_t mropeSectionW;
    float epsilon;
    float reciprocal;
    int64_t usedCoreNum;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t ubFactor;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, UndGenQkvRmsNormRopeCacheTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0U; i < sizeof(UndGenQkvRmsNormRopeCacheTilingData) / sizeof(uint32_t); ++i) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, UndGenQkvRmsNormRopeCacheTilingData* const_data)
{
    std::memcpy(const_data, tiling, sizeof(UndGenQkvRmsNormRopeCacheTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif
