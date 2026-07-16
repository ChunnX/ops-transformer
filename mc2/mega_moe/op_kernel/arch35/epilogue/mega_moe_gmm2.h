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
 * \file mega_moe_gmm2.h
 * \brief GMM2 single-core orchestration shared by the AIC matmul and its AIV epilogue.
 */

#ifndef MEGA_MOE_GMM2_H
#define MEGA_MOE_GMM2_H

#include "kernel_operator.h"
#include "../mega_moe_impl.h"

using namespace AscendC;

namespace MegaMoeGmm2Impl {
using Gmm2ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

// 当前 expert 的 GMM2 shape、全局地址视图和 group 信息。
struct Gmm2ProblemContext {
    const Gmm2ProblemShape& problemShape;
    const GMMAddrInfo& addrInfo;
    uint32_t expertBeforeCnt;
    uint32_t expertIdx;
};

// GMM2 在多个 expert 之间需要延续的可变运行状态。
// logicalTileId 是连续逻辑调度 ID，不表示物理 block ID。
struct Gmm2RuntimeState {
    uint32_t& logicalTileId;
    int32_t& vecSetSyncCom;
    uint16_t& pingPongIdx;
};

// ==================================================================================
// RunGmm2MatmulSingleCore：按 A8W4/A4W4、NZ/ND 路径选择底层 GMM2 单核实现
// ==================================================================================
template <int32_t CombineQuantMode, typename QuantOutType, typename SwigluQuantOutType, typename WeightType,
    typename QuantScaleType, bool EnableA8W4, bool EnableA4W4>
__aicore__ inline void RunGmm2MatmulSingleCore(const Params& params, const Gmm2ProblemContext& problemContext,
    Gmm2RuntimeState& runtimeState)
{
    if constexpr (EnableA8W4 || EnableA4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, WeightType, bfloat16_t,
            QuantScaleType, QuantScaleType>(params, problemContext.problemShape, problemContext.addrInfo,
            runtimeState.logicalTileId, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
            runtimeState.pingPongIdx, problemContext.expertIdx);
    } else {
        // A8W8_NZ 和 generic ND 共用 GroupMatmul2，仅 IsWeightNZ 模板参数不同。
        if (params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleType, QuantScaleType, true>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.logicalTileId, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleType, QuantScaleType>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.logicalTileId, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx);
        }
    }
}

// ==================================================================================
// ProcessGmm2SingleCore：GMM2 单核总入口，只负责矩阵计算
// ==================================================================================
template <int32_t CombineQuantMode, typename QuantOutType, typename SwigluQuantOutType, typename WeightType,
    typename QuantScaleType, bool EnableA8W4, bool EnableA4W4>
__aicore__ inline void ProcessGmm2SingleCore(const Params& params, const Gmm2ProblemContext& problemContext,
    Gmm2RuntimeState& runtimeState)
{
    RunGmm2MatmulSingleCore<CombineQuantMode, QuantOutType, SwigluQuantOutType, WeightType, QuantScaleType,
        EnableA8W4, EnableA4W4>(params, problemContext, runtimeState);
}

// ==================================================================================
// FinalizeGmm2SingleCore：非量化 GMM2 路径等待 ping-pong 同步收尾
// ==================================================================================
template <int32_t CombineQuantMode>
__aicore__ inline void FinalizeGmm2SingleCore(Gmm2RuntimeState& runtimeState)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(runtimeState.vecSetSyncCom, runtimeState.pingPongIdx);
    }
}

} // namespace MegaMoeGmm2Impl

#endif // MEGA_MOE_GMM2_H
