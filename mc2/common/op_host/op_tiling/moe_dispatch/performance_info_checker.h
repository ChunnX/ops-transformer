/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - PerformanceInfo 可选特性校验
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.4
 *
 * 支持情况（见 architecture_v0.md §六 可选特性矩阵）：
 *   v2-A3/A5 MTE: 支持
 *   arch35 CCU / setup / teardown: 不支持
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H

#include <cstdint>
// TODO: include framework headers

namespace performance_info_check {

/**
 * 互斥规则校验：performanceInfo 与 layered 互斥。
 * 阶段 C 调用。
 */
inline graphStatus CheckCompatibility(const char* nodeName, bool isLayered) {
    if (isLayered) {
        // TODO: OPS_LOG_E(nodeName, "performanceInfo is not supported with layered mode")
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * performanceInfo shape 精确值校验。
 * 阶段 G 调用。
 * @param inputIndex  performanceInfo 在 context 中的输入索引
 * @param epWorldSize 从属性读取的 epWorldSize 值
 */
inline graphStatus CheckShape(TilingContext* ctx, uint32_t inputIndex, uint32_t epWorldSize) {
    // TODO: 获取 performanceInfo tensor desc
    // TODO: 校验 dim0 == epWorldSize
    return ge::GRAPH_SUCCESS;
}

}  // namespace performance_info_check

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H
