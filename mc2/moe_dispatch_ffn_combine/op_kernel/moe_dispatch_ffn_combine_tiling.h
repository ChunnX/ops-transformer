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

#ifndef MOE_DISPATCH_FFN_COMBINE_TILING_H
#define MOE_DISPATCH_FFN_COMBINE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeDispatchFFNCombineInfo {
    // 原有字段
    uint32_t epWorldSize;
    uint32_t moeExpertNum;
    uint32_t sharedExpertNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t h;
    uint32_t n;
    uint32_t k;
    uint32_t maxRecvTokenNum;
    uint32_t dispatchQuantMode;
    uint32_t combineQuantMode;
    uint64_t cclBufferSize;
    
    // 从MegaMoeTilingData补充的字段
    uint32_t expertPerRank;        // 每个rank的expert数量
    uint8_t groupListType;         // grouplist类型
    uint32_t m;                    // 矩阵m维度
    uint32_t m0;                   // GMM1 tileShape M
    uint32_t k0;                   // GMM1 tileShape K
    uint32_t n0;                   // GMM1 tileShape N
    uint32_t EP;                   // EP大小（与epWorldSize相同）
    uint32_t blockNumPerEP;        // 每个EP的block数量
    uint32_t maxOutputSize;        // 最大输出大小
    uint32_t rank;                 // 当前rank
    uint32_t topK;                 // topK值
    int32_t dispatchRows;          // dispatch行数
    bool transX;                   // X转置标志
    bool transW;                   // W转置标志
    bool transW2;                  // W2转置标志
};

struct MoeDispatchFFNCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDispatchFFNCombineInfo megaMoeInfo;
};

#endif
