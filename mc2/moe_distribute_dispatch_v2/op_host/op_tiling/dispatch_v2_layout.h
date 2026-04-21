/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - v2 Layout 定义与 TilingCtx
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §4.1 §4.5
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §2.1
 *
 * MoeDispatchTilingLayout：算子原型索引映射，使公共层校验与具体算子原型解耦。
 * DispatchV2TilingCtx：阶段间中间值载体，避免参数膨胀。
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_OP_HOST_OP_TILING_DISPATCH_V2_LAYOUT_H
#define MOE_DISTRIBUTE_DISPATCH_V2_OP_HOST_OP_TILING_DISPATCH_V2_LAYOUT_H

#include <cstdint>
#include "../../../../common/op_host/op_tiling/moe_dispatch/defs.h"

using namespace moe_dispatch_defs;

// ========================================
// MoeDispatchTilingLayout -- 算子原型索引映射
// 见 architecture_v0.md §4.1
// ========================================

struct MoeDispatchTilingLayout {
    // ========== 输入 tensor 索引 ==========
    // 必选用 uint32_t；可选用 int32_t（LAYOUT_INDEX_ABSENT 表示不存在）
    int32_t  contextIndex;              // 仅 v3 使用（isMc2Context=true），v2 为 ABSENT
    uint32_t xIndex;
    uint32_t expertIdsIndex;
    int32_t  scalesIndex;               // 可选
    int32_t  activeMaskIndex;           // 可选
    int32_t  expertScalesIndex;         // 可选
    int32_t  elasticInfoIndex;          // 可选
    int32_t  performanceInfoIndex;      // 可选

    // ========== 输出 tensor 索引 ==========
    uint32_t outputStartIndex;          // 第一个输出 tensor 的索引
    uint32_t outputCount;               // 输出 tensor 个数

    // ========== 属性索引（共 17 个，对应当前 DispatchV2Config）==========
    uint32_t attrGroupEpIndex;
    uint32_t attrEpWorldSizeIndex;
    uint32_t attrEpRankIdIndex;
    uint32_t attrMoeExpertNumIndex;
    int32_t  attrCclBufferSizeIndex;    // v3 专用，v2 为 ABSENT
    uint32_t attrGroupTpIndex;
    uint32_t attrTpWorldSizeIndex;
    uint32_t attrTpRankIdIndex;
    uint32_t attrExpertSharedTypeIndex;
    uint32_t attrSharedExpertNumIndex;
    uint32_t attrSharedExpertRankNumIndex;
    uint32_t attrQuantModeIndex;
    uint32_t attrGlobalBsIndex;
    uint32_t attrExpertTokenNumsTypeIndex;
    int32_t  attrCommAlgIndex;          // 可选（setup/teardown 无此属性）
    int32_t  attrZeroExpertNumIndex;    // 可选
    int32_t  attrCopyExpertNumIndex;    // 可选
    int32_t  attrConstExpertNumIndex;   // 可选

    // ========== 标志位 ==========
    bool isMc2Context;                  // v3 设为 true，v2 为 false
};

// ========================================
// v2 Layout 实例（编译期常量）
// 见 architecture_v0.md §4.1，pseudocode_v0.md §2.1
// ========================================

// v2 A3 Layout（索引值参照 moe_distribute_dispatch_v2_def.cpp 中的算子定义）
constexpr MoeDispatchTilingLayout V2_A3_LAYOUT = {
    .contextIndex              = LAYOUT_INDEX_ABSENT,  // v2 无 context 输入
    .xIndex                    = 0,
    .expertIdsIndex            = 1,
    .scalesIndex               = 2,
    .activeMaskIndex           = 3,
    .expertScalesIndex         = 4,
    .elasticInfoIndex          = 5,
    .performanceInfoIndex      = 6,
    .outputStartIndex          = 0,
    .outputCount               = 6,  // expandX + expandExpertIds + expertTokenNums + expandXDynamicScales + epRecvCounters + assistInfo
    .attrGroupEpIndex          = 0,
    .attrEpWorldSizeIndex      = 1,
    .attrEpRankIdIndex         = 2,
    .attrMoeExpertNumIndex     = 3,
    .attrCclBufferSizeIndex    = LAYOUT_INDEX_ABSENT,
    .attrGroupTpIndex          = 4,
    .attrTpWorldSizeIndex      = 5,
    .attrTpRankIdIndex         = 6,
    .attrExpertSharedTypeIndex = 7,
    .attrSharedExpertNumIndex  = 8,
    .attrSharedExpertRankNumIndex = 9,
    .attrQuantModeIndex        = 10,
    .attrGlobalBsIndex         = 11,
    .attrExpertTokenNumsTypeIndex = 12,
    .attrCommAlgIndex          = 13,
    .attrZeroExpertNumIndex    = 14,
    .attrCopyExpertNumIndex    = 15,
    .attrConstExpertNumIndex   = 16,
    .isMc2Context              = false,
};

// v2 A5 Layout：索引与 A3 完全相同，仅 arch 类型不同（通过 TilingCtx 区分）
constexpr MoeDispatchTilingLayout V2_A5_LAYOUT = V2_A3_LAYOUT;

// ========================================
// DispatchV2TilingCtx -- 阶段间中间值载体
// 见 architecture_v0.md §4.5
// ========================================

enum class ArchType { ARCH_A3, ARCH_A5 };

struct DispatchV2TilingCtx {
    // ----- 阶段 B 产出：属性值 -----
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t tpWorldSize;
    uint32_t tpRankId;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t globalBs;
    uint32_t commAlg;
    uint32_t zeroExpertNum;
    bool     isLayered;
    bool     isFullMeshV2;
    ArchType archType;

    // ----- 阶段 C 产出：可选特性标志 -----
    bool hasElastic;
    bool hasActiveMask;
    bool hasPerformance;

    // ----- 阶段 D 产出：中间计算值 -----
    uint32_t localExpertNum;
    bool     isSharedRank;
    uint32_t moeExpertPerRankNum;
    uint32_t A;

    // ----- 阶段 F 产出：量化计算值 -----
    bool     hasScales;
    uint32_t tokenMsgSize;
    uint32_t dynamicScalesDim1;
};

#endif  // MOE_DISTRIBUTE_DISPATCH_V2_OP_HOST_OP_TILING_DISPATCH_V2_LAYOUT_H
