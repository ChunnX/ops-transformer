/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - HCCL buffer 大小校验
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.7
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_HCCL_BUFFER_CHECKER_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_HCCL_BUFFER_CHECKER_H

#include <cstdint>
// TODO: include framework headers (TilingContext, graphStatus, etc.)

namespace hccl_buffer_check {

/**
 * 按参数计算最小 HCCL buffer 需求。
 * 各算子按自己的公式计算后传入 CheckBuffSizeFromContext/CheckBuffSizeFromAttr。
 */
inline uint64_t CalcMinBuffSize(/* TODO: 算子相关参数 */) {
    uint64_t minRequired = 0;
    // TODO: 根据 tokenMsgSize, epWorldSize, A 等计算最小 buffer 需求
    // 具体公式参照原各算子 CheckHcclBuffSize 实现
    return minRequired;
}

/**
 * 从 HCCL group 获取实际 buffer 大小并与最小需求比较。
 * v2/setup/teardown 通用路径（非 mc2Context 场景）。
 */
inline graphStatus CheckBuffSizeFromContext(TilingContext* ctx,
                                            uint32_t groupIndex,
                                            uint64_t minRequired) {
    // TODO: 从 ctx 获取 HCCL group groupIndex 的实际 buffer 大小
    // TODO: 实际大小 < minRequired 时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

/**
 * 从属性获取 buffer 大小并比较（V3/Mc2Context 场景）。
 * v3 通过 cclBufferSize 属性传入 buffer 大小。
 */
inline graphStatus CheckBuffSizeFromAttr(uint64_t cclBufferSize, uint64_t minRequired) {
    // TODO: cclBufferSize < minRequired 时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

}  // namespace hccl_buffer_check

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_HCCL_BUFFER_CHECKER_H
