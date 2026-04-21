/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - ActiveMask 可选特性校验
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.4
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.6
 *
 * 支持情况（见 architecture_v0.md §六 可选特性矩阵）：
 *   v2-A3/A5 MTE : 支持（1D + 2D）
 *   arch35 CCU   : 支持（仅 1D）
 *   setup        : 支持（仅 1D）
 *   teardown     : 不支持
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_ACTIVE_MASK_CHECKER_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_ACTIVE_MASK_CHECKER_H

#include <cstdint>
// TODO: include framework headers

namespace active_mask_check {

/**
 * 互斥/约束规则校验。
 * 阶段 C 调用。
 * @param isLayered    是否为 layered 模式
 * @param isExpertMask 是否为 expert mask 模式（影响 2D 支持）
 * @param globalBs     globalBs 属性值（约束条件之一）
 */
inline graphStatus CheckCompatibility(const char* nodeName,
                                      bool isLayered,
                                      bool isExpertMask,
                                      uint32_t globalBs) {
    // TODO: 校验互斥规则（参照原 v2 tiling CheckActiveMaskCompatibility 逻辑）
    // - isLayered 时某些 mask 类型不支持
    // - globalBs 约束
    return ge::GRAPH_SUCCESS;
}

/**
 * xActiveMask shape 精确值校验。
 * 阶段 G 调用。
 * @param inputIndex  activeMask 在 context 中的输入索引
 * @param bs          batch size 值
 * @param k           k 值（topK）
 */
inline graphStatus CheckShape(TilingContext* ctx, uint32_t inputIndex,
                               uint32_t bs, uint32_t k) {
    // TODO: 获取 activeMask tensor desc
    // TODO: 1D 时: dim0 == bs
    // TODO: 2D 时: dim0 == bs, dim1 == k
    return ge::GRAPH_SUCCESS;
}

}  // namespace active_mask_check

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_ACTIVE_MASK_CHECKER_H
