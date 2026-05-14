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
 * \file dispatch_v2_layout.h
 * \brief MOE Distribute Dispatch V2 Layout + TilingCtx 定义
 */

#ifndef DISPATCH_V2_LAYOUT_H
#define DISPATCH_V2_LAYOUT_H

#include <cstdint>

namespace optiling {

// ========== 可选索引缺省值 ==========
constexpr int32_t LAYOUT_INDEX_ABSENT = -1;

// ========== MoeDispatchTilingLayout -- 算子原型索引映射 ==========
// 各算子定义自己的 Layout，存储 tensor 和 attr 在 context 中的索引位置
struct MoeDispatchTilingLayout {
    // ========== 输入 tensor 索引 ==========
    // 必选索引用 uint32_t；可选索引用 int32_t（LAYOUT_INDEX_ABSENT 表示不存在）
    int32_t  contextIndex{LAYOUT_INDEX_ABSENT};   // 仅 v3 使用，v2 为 ABSENT
    uint32_t xIndex{0U};
    uint32_t expertIdsIndex{1U};
    int32_t  scalesIndex{2U};                     // 可选
    int32_t  activeMaskIndex{3U};                 // 可选
    int32_t  expertScalesIndex{4U};               // 可选
    int32_t  elasticInfoIndex{5U};                // 可选
    int32_t  performanceInfoIndex{6U};            // 可选

    // ========== 输出 tensor 索引 ==========
    uint32_t outputStartIndex{0U};
    uint32_t outputCount{6U};

    // ========== 属性索引 ==========
    uint32_t attrGroupEpIndex{0U};
    uint32_t attrEpWorldSizeIndex{1U};
    uint32_t attrEpRankIdIndex{2U};
    uint32_t attrMoeExpertNumIndex{3U};
    int32_t  attrCclBufferSizeIndex{LAYOUT_INDEX_ABSENT};  // v3 专用，v2 为 ABSENT
    uint32_t attrGroupTpIndex{4U};
    uint32_t attrTpWorldSizeIndex{5U};
    uint32_t attrTpRankIdIndex{6U};
    uint32_t attrExpertSharedTypeIndex{7U};
    uint32_t attrSharedExpertNumIndex{8U};
    uint32_t attrSharedExpertRankNumIndex{9U};
    uint32_t attrQuantModeIndex{10U};
    uint32_t attrGlobalBsIndex{11U};
    uint32_t attrExpertTokenNumsTypeIndex{12U};
    int32_t  attrCommAlgIndex{13U};               // 可选（setup/teardown 无此属性）
    int32_t  attrZeroExpertNumIndex{14U};         // 可选
    int32_t  attrCopyExpertNumIndex{15U};         // 可选
    int32_t  attrConstExpertNumIndex{16U};        // 可选

    // ========== 标志位 ==========
    bool     isMc2Context{false};                 // v3 设为 true，v2 为 false
};

// ========== DispatchV2TilingCtx -- 阶段间中间值载体 ==========
// 承载阶段间传递的全部中间值，避免参数膨胀
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

#endif // DISPATCH_V2_LAYOUT_H