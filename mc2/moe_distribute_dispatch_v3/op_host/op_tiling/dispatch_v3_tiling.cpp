/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - dispatch_v3 主流程（v2 薄包装）
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §七 7.2
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §3
 *
 * v3 仍为 v2 薄包装：
 *   - 构造 V3_LAYOUT（输入索引全部偏移 +1，isMc2Context=true）
 *   - 直接调用 v2 公共入口 DispatchV2TilingFunc
 *   - 属性中 cclBufferSizeIndex 有值（v2 为 ABSENT）
 */

#include "../../moe_distribute_dispatch_v2/op_host/op_tiling/dispatch_v2_layout.h"
// TODO: include framework headers

// v3 Layout：输入索引全部偏移 +1（context 排在第 0 位）
// 见 architecture_v0.md §七 7.2，pseudocode_v0.md §3
constexpr MoeDispatchTilingLayout V3_LAYOUT = {
    .contextIndex              = 0,                    // v3 独有：mc2Context
    .xIndex                    = 1,                    // v2 是 0
    .expertIdsIndex            = 2,                    // v2 是 1
    .scalesIndex               = 3,                    // v2 是 2
    .activeMaskIndex           = 4,                    // v2 是 3
    .expertScalesIndex         = 5,                    // v2 是 4
    .elasticInfoIndex          = 6,                    // v2 是 5
    .performanceInfoIndex      = 7,                    // v2 是 6
    .outputStartIndex          = 0,
    .outputCount               = 7,                    // v3 多一个 moeRecvCount 输出
    .attrGroupEpIndex          = 0,
    .attrEpWorldSizeIndex      = 1,
    .attrEpRankIdIndex         = 2,
    .attrMoeExpertNumIndex     = 3,
    .attrCclBufferSizeIndex    = 3,                    // v3 有此属性（v2 为 ABSENT）
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
    .isMc2Context              = true,                 // v3 核心差异
};

// v3 直接复用 v2 主流程，仅 Layout 不同
// 见 pseudocode_v0.md §3
graphStatus DispatchV3Entry(TilingContext* ctx) {
    // TODO: 调用 DispatchV2TilingFunc(ctx, V3_LAYOUT, ArchType::ARCH_A5)
    // 需要将 DispatchV2TilingFunc 暴露为可链接符号（或通过 #include dispatch_v2_tiling.cpp 的内部接口）
    return ge::GRAPH_SUCCESS;
}

// TODO: REGISTER_TILING_FUNC 宏注册 v3 入口
