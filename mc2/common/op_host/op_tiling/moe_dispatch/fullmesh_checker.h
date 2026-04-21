/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - FullMeshV2 可选特性校验
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.4
 *
 * 支持情况（见 architecture_v0.md §六 可选特性矩阵）：
 *   v2-A3/A5 MTE: 支持（commAlg 属性驱动）
 *   arch35 CCU / setup / teardown: 不支持
 *
 * FullMesh 无独立 tensor，通过 commAlg 属性判断是否激活。
 * 此处仅做启用时的约束校验。
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_FULLMESH_CHECKER_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_FULLMESH_CHECKER_H

#include <cstdint>
// TODO: include framework headers

namespace fullmesh_check {

/**
 * FullMeshV2 启用时的约束校验。
 * 阶段 C 调用（isFullMeshV2 = true 时）。
 *
 * 约束（见原 v2 full_mesh 校验逻辑）：
 *   - BS <= 256
 *   - K  <= 12
 *   - tpWorldSize != 2（TP=2 与 fullmesh 不兼容）
 */
inline graphStatus CheckConstraints(const char* nodeName,
                                    uint32_t bs,
                                    uint32_t k,
                                    uint32_t tpWorldSize) {
    if (bs > 256U) {
        // TODO: OPS_LOG_E(nodeName, "fullmesh requires BS <= 256, got %u", bs)
        return ge::GRAPH_FAILED;
    }
    if (k > 12U) {
        // TODO: OPS_LOG_E(nodeName, "fullmesh requires K <= 12, got %u", k)
        return ge::GRAPH_FAILED;
    }
    if (tpWorldSize == 2U) {
        // TODO: OPS_LOG_E(nodeName, "fullmesh is incompatible with tpWorldSize=2")
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace fullmesh_check

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_FULLMESH_CHECKER_H
