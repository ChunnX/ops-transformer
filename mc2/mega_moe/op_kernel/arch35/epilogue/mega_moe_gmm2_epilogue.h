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
 * \file mega_moe_gmm2_epilogue.h
 * \brief GMM2 AIV prologue/post processing and AIC/AIV synchronization finalization.
 */

#ifndef MEGA_MOE_GMM2_EPILOGUE_H
#define MEGA_MOE_GMM2_EPILOGUE_H

#include "kernel_operator.h"
#include "../block/mega_moe_gmm2_block.h"

using namespace AscendC;

namespace MegaMoeGmm2EpilogueImpl {
using Gmm2ProblemContext = MegaMoeGmm2BlockImpl::Gmm2ProblemContext;
using Gmm2RuntimeState = MegaMoeGmm2BlockImpl::Gmm2RuntimeState;

// ==================================================================================
// Prototype: MegaMoe::GroupMatmulWithCombine. Runs one logical AIV side of GMM2.
// ==================================================================================
template <int32_t CombineQuantMode, typename QuantOutType, typename SwigluQuantOutType, typename WeightType,
    typename QuantScaleType, bool EnableA8W4, bool EnableA4W4>
__aicore__ inline void ProcessGmm2EpilogueOneCore(const Params& params,
    const Gmm2ProblemContext& problemContext, Gmm2RuntimeState& runtimeState)
{
    if (problemContext.threadBlock.threadBlockNum == 0U ||
        problemContext.threadBlock.threadBlockIdx >= problemContext.threadBlock.threadBlockNum) {
        return;
    }
    if constexpr (g_coreType == AIV) {
        if constexpr (EnableA8W4 || EnableA4W4) {
            MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, WeightType, bfloat16_t,
                QuantScaleType, QuantScaleType>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx, problemContext.threadBlock);
        } else if (params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleType, QuantScaleType, true>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx, problemContext.threadBlock);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleType, QuantScaleType>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx, problemContext.threadBlock);
        }
    }
}

// ==================================================================================
// Prototype: MegaMoe::GroupMatmulWithCombine. Completes non-quant GMM2 synchronization.
// ==================================================================================
template <int32_t CombineQuantMode>
__aicore__ inline void FinalizeGmm2OneCore(Gmm2RuntimeState& runtimeState)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(runtimeState.vecSetSyncCom, runtimeState.pingPongIdx);
    }
}

} // namespace MegaMoeGmm2EpilogueImpl

#endif // MEGA_MOE_GMM2_EPILOGUE_H
