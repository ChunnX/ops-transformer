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
 * \file dispatch_v2_tiling_ctx.h
 * \brief MOE Distribute Dispatch V2 Tiling 中间值传递结构体
 */

#ifndef DISPATCH_V2_TILING_CTX_H
#define DISPATCH_V2_TILING_CTX_H

#include <cstdint>

namespace optiling {

// dispatch_v2 主流程中间值传递结构体
// 用于解决参数膨胀问题，将阶段间传递的中间值统一收束
struct DispatchV2TilingCtx {
    // ========== 属性值（阶段 B）==========
    uint32_t epWorldSize{0};
    uint32_t tpWorldSize{0};
    uint32_t epRankId{0};
    uint32_t tpRankId{0};
    uint32_t moeExpertNum{0};
    uint32_t sharedExpertNum{0};
    uint32_t sharedExpertRankNum{0};
    uint32_t quantMode{0};
    uint32_t expertShardType{0};
    uint32_t globalBs{0};
    uint32_t expertTokenNumsType{0};
    int32_t  zeroComputeExpertNum{0};

    // ========== 标志位（阶段 B/C）==========
    bool isLayered{false};
    bool isFullMeshV2{false};
    bool hasElastic{false};
    bool hasActiveMask{false};
    bool isTokenMask{false};      // 1D activeMask
    bool isExpertMask{false};     // 2D activeMask
    bool hasPerformance{false};
    bool hasScales{false};
    bool isMc2Context{false};
    bool isSharedRank{false};

    // ========== 中间计算值（阶段 D）==========
    uint32_t localExpertNum{0};
    uint32_t A{0};                // expandX dim0 计算基数

    // ========== tensor 维度（阶段 E）==========
    uint32_t bs{0};
    uint32_t h{0};
    uint32_t k{0};

    // ========== 量化计算值（阶段 F）==========
    uint64_t scalesRow{0};
    uint64_t scalesCol{0};
    uint32_t scalesTypeSize{0};

    // ========== 平台信息（阶段 I）==========
    uint32_t aivNum{0};
    uint64_t totalUbSize{0};
    uint64_t totalWinSizeEp{0};
};

} // namespace optiling

#endif // DISPATCH_V2_TILING_CTX_H
