/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_dispatch_ffn_combine_tiling.h
 * \brief
 */

#ifndef ASCENDC_MOE_DISPATCH_FFN_COMBINE_TILING
#define ASCENDC_MOE_DISPATCH_FFN_COMBINE_TILING
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_arch35_tiling_def.h"

using namespace AscendC;

struct MoeDispatchFFNCombineTilingData {
    MoeInitRoutingV3Arch35TilingData moeInitRoutingTilingData;
    uint32_t expertPerRank;
    uint8_t groupListType;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t m0; // 目前在gmm1 tileShape中使用
    uint32_t k0;
    uint32_t n0;
    uint32_t EP;
    uint32_t blockNumPerEP; // 和dispatchRows相关，如果每个EP搬运的row小于dispatchRows，则dispatchRows失效
    uint32_t maxOutputSize;
    uint32_t rank;
    uint32_t topK;
    int32_t dispatchRows;
    bool transX;
    bool transW;
    bool transW2;
    uint64_t tilingKey;
};

struct MoeDispatchFFNCombineAuxTilingContext {
    // 平台信息
    int64_t aivCoreNum;
    int64_t totalUbSize;
    int64_t availUbSize;

    // 重要变量
    int64_t sortLoopMaxElement;
    int64_t totalLength;
    int64_t n;
    int64_t k;
    int64_t cols;
    int64_t inputXDtypeSize;
    int64_t isInputScale;
    int64_t isInputOffset;
    int64_t sortMode;

    // 算子属性
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensNumType;
    bool expertTokensNumFlag;
    int64_t quantMode;
    int64_t expertStart;
    int64_t expertEnd;
    int64_t rowIdxType;
};

#endif