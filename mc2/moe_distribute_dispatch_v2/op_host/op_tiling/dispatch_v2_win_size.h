/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - v2 win size 校验
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §七 阶段 H
 *
 * 仅含 v2 特有的复杂 WinSize/HCCL buffer 校验逻辑。
 * setup/teardown 的 HCCL buffer 校验通过公共层 hccl_buffer_checker.h 处理。
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_OP_HOST_OP_TILING_DISPATCH_V2_WIN_SIZE_H
#define MOE_DISTRIBUTE_DISPATCH_V2_OP_HOST_OP_TILING_DISPATCH_V2_WIN_SIZE_H

#include "dispatch_v2_layout.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/hccl_buffer_checker.h"
// TODO: include framework headers

/**
 * v2 WinSize/HCCL buffer 综合校验。
 * 阶段 H 调用。
 *
 * 复杂路径（见 architecture_v0.md §七 阶段 H）：
 *   - layered 公式
 *   - fullmesh 公式
 *   - TP 多路（tpWorldSize > 1）
 *   - isMc2Context（v3 通过 cclBufferSize 属性）
 *   与原 moe_distribute_check_win_size.h 中逻辑对应
 */
inline graphStatus CheckV2WinSize(TilingContext* ctx,
                                  const MoeDispatchTilingLayout& layout,
                                  const DispatchV2TilingCtx& tc) {
    // TODO: 根据 tc.isLayered / tc.isFullMeshV2 / tc.tpWorldSize 选择不同公式计算 minRequired
    // TODO: isMc2Context 时调用 hccl_buffer_check::CheckBuffSizeFromAttr(cclBufferSize, minRequired)
    // TODO: 否则调用 hccl_buffer_check::CheckBuffSizeFromContext(ctx, groupIndex, minRequired)
    return ge::GRAPH_SUCCESS;
}

#endif  // MOE_DISTRIBUTE_DISPATCH_V2_OP_HOST_OP_TILING_DISPATCH_V2_WIN_SIZE_H
