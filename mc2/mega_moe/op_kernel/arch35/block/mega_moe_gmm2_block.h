/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_gmm2_block.h
 * \brief Single-core entry for GMM2 matmul and post processing.
 */

#ifndef MEGA_MOE_GMM2_BLOCK_H
#define MEGA_MOE_GMM2_BLOCK_H

#include "kernel_operator.h"
#include "../mega_moe_impl.h"

using namespace AscendC;

namespace MegaMoeGmm2BlockImpl {
using Gmm2ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

// GMM2 block 与 epilogue 共用的问题描述，所有地址均由外部编排器准备。
struct Gmm2ProblemContext {
    const Gmm2ProblemShape& problemShape;
    const GMMAddrInfo& addrInfo;
    uint32_t expertBeforeCnt;
    uint32_t expertIdx;
};

// 每个核独立维护的逻辑调度状态；logicalTileId 不表示物理 block ID。
struct Gmm2RuntimeState {
    uint32_t& logicalTileId;
    int32_t& vecSetSyncCom;
    uint16_t& pingPongIdx;
};

// 纯 AIC GMM2 入口。AIV 编译实例不会执行或实例化矩阵计算分支。
template <int32_t CombineQuantMode, typename QuantOutType, typename SwigluQuantOutType, typename WeightType,
    typename QuantScaleType, bool EnableA8W4, bool EnableA4W4, bool IsShared = false>
__aicore__ inline void ProcessGmm2BlockSingleCore(const Params& params,
    const Gmm2ProblemContext& problemContext, Gmm2RuntimeState& runtimeState)
{
    if constexpr (EnableA8W4 || EnableA4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, WeightType, bfloat16_t,
            QuantScaleType, QuantScaleType, IsShared>(params, problemContext.problemShape, problemContext.addrInfo,
            runtimeState.logicalTileId, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
            runtimeState.pingPongIdx, problemContext.expertIdx);
    } else if (params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
        MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
            QuantScaleType, QuantScaleType, true, false, IsShared>(
            params, problemContext.problemShape, problemContext.addrInfo,
            runtimeState.logicalTileId, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
            runtimeState.pingPongIdx, problemContext.expertIdx);
    } else {
        MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
            QuantScaleType, QuantScaleType, false, false, IsShared>(
            params, problemContext.problemShape, problemContext.addrInfo,
            runtimeState.logicalTileId, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
            runtimeState.pingPongIdx, problemContext.expertIdx);
    }
}

template <int32_t CombineQuantMode>
__aicore__ inline void FinalizeGmm2SingleCore(Gmm2RuntimeState& runtimeState)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(runtimeState.vecSetSyncCom, runtimeState.pingPongIdx);
    }
}

} // namespace MegaMoeGmm2BlockImpl

#endif // MEGA_MOE_GMM2_BLOCK_H
