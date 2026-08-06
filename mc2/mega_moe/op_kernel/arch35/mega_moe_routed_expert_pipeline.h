/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ROUTED_EXPERT_PIPELINE_H
#define MEGA_MOE_ROUTED_EXPERT_PIPELINE_H

#include "mega_moe_dispatch_gmm1_swiglu.h"
#include "mega_moe_gmm2_combine.h"

namespace MegaMoeImpl {

// Prototype: MegaMoe::Process routed-expert stages.
// Controls the regular template's complete Dispatch -> GMM1/SwiGLU -> GMM2/Combine pipeline.
template <uint8_t CombineMode, typename QuantOutType, typename ActivationType, typename WeightType,
          typename SwigluOutType, typename QuantScaleType, bool EnableA8W4, bool EnableA4W4,
          uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void ControlRoutedExpertPipeline(
    const DispatchGmm1SwigluContext &gmm1Context, const DispatchGmm1SwigluArgs &gmm1Args,
    const Gmm2Context &gmm2Context, const Gmm2Args &gmm2Args,
    const QuantCombineContext &combineContext, const QuantCombineBufferConfig &combineBufferConfig,
    const Params &params, BlockEpilogue &epilogueOp, TokenDispatchScratch<ActivationType> &dispatchScratch,
    Gmm1SwigluScratch &gmm1Scratch, Gmm2Scratch &gmm2Scratch, GM_ADDR expertTokenNumsOutPtr,
    int32_t &gmTileSequence, uint16_t &gmm2PingPongIdx)
{
    uint32_t gmm1StartBlockIdx = 0U;
    int32_t gmm1VecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    DispatchGmm1SwigluState gmm1RuntimeState{
        gmm1StartBlockIdx, gmm1VecSetSyncCom, gmTileSequence, gmm1PingPongIdx};
    ControlDispatchGmm1SwigluPipeline<QuantOutType, ActivationType, WeightType, SwigluOutType,
                                      QuantScaleType, EnableA8W4, Gmm1TileM, EpilogueTileM,
                                      TopkWeightsPrefetch>(
        gmm1Context, gmm1Args, params, epilogueOp, dispatchScratch, gmm1Scratch, gmm1RuntimeState);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            ExportExpertTokenCounts<ActivationType, EnableA8W4>(
                gmm1Context.dispatch, dispatchScratch, expertTokenNumsOutPtr);
        }
    }
    if constexpr (TopkWeightsPrefetch) {
        SyncAll<false>();
    }

    uint32_t gmm2StartBlockIdx = gmm1RuntimeState.startBlockIdx;
    int32_t gmm2VecSetSyncCom = 0;
    Gmm2RuntimeState gmm2RuntimeState{
        gmm2StartBlockIdx, gmm2VecSetSyncCom, gmTileSequence, gmm2PingPongIdx};
    ControlGmm2CombinePipeline<CombineMode, QuantOutType, SwigluOutType, WeightType,
                               QuantScaleType, ActivationType, EnableA8W4, EnableA4W4,
                               Gmm1TileM, TopkWeightsPrefetch>(
        gmm2Context, gmm2Args, gmm2Scratch, combineContext, combineBufferConfig, params, gmm2RuntimeState);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_ROUTED_EXPERT_PIPELINE_H
