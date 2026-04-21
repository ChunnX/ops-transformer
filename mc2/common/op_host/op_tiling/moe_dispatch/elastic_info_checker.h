/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - ElasticInfo 可选特性校验
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.4
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.6
 *
 * 支持情况（见 architecture_v0.md §六 可选特性矩阵）：
 *   v2-A3/A5 MTE: 支持
 *   arch35 CCU / setup / teardown: 不支持
 *
 * 设计原则：
 *   - 仅做互斥规则校验 + 产出标志/ElasticResult
 *   - elasticInfo tensor 的基本校验（dim+dtype+format）在阶段 E 通过 TensorCheckRule 处理
 *   - shape 精确值校验在阶段 G 中执行
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_ELASTIC_INFO_CHECKER_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_ELASTIC_INFO_CHECKER_H

#include <cstdint>
#include "utils.h"
// TODO: include framework headers

namespace elastic_info_check {

using moe_dispatch_utils::ElasticResult;

/**
 * 互斥规则校验：elastic 与 layered/tp 互斥。
 * 阶段 C 调用（可选特性互斥校验阶段）。
 */
inline graphStatus CheckCompatibility(const char* nodeName, bool isLayered, bool hasTp) {
    if (isLayered) {
        // TODO: OPS_LOG_E(nodeName, "elasticInfo is not supported with layered mode")
        return ge::GRAPH_FAILED;
    }
    if (hasTp) {
        // TODO: OPS_LOG_E(nodeName, "elasticInfo is not supported with TP")
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * elasticInfo shape 精确值校验。
 * 阶段 G 调用（shape 精确值校验阶段）。
 * @param inputIndex  elasticInfo 在 context 中的输入索引
 * @param epWorldSize 从属性读取的 epWorldSize 值
 */
inline graphStatus CheckShape(TilingContext* ctx, uint32_t inputIndex, uint32_t epWorldSize) {
    // TODO: 获取 elasticInfo tensor desc
    // TODO: 校验 dim0 == 4 + 2 * epWorldSize
    return ge::GRAPH_SUCCESS;
}

/**
 * elastic 相关中间值计算（A 值和 dim0 调整）。
 * 阶段 D 调用（中间值计算阶段）。
 * @param localExpertNum   普通 rank 本地 expert 数
 * @param sharedA          shared rank 时的 A 值
 * @param moeA             非 shared rank 时的 A 值
 * @param epWorldSize      epWorldSize 属性值
 */
inline ElasticResult CalcAdjustments(uint32_t localExpertNum,
                                     uint32_t sharedA,
                                     uint32_t moeA,
                                     uint32_t epWorldSize) {
    ElasticResult result{};
    result.hasElastic = true;
    // TODO: 计算 adjustedA, adjustedExpertTokenDim0, adjustedEpRecvDim0
    // 逻辑来自原 v2 tiling 中 elastic 相关计算
    return result;
}

}  // namespace elastic_info_check

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_ELASTIC_INFO_CHECKER_H
