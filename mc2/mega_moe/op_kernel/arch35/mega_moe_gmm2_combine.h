/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_H
#define MEGA_MOE_GMM2_COMBINE_H

#include "aiv_comm/mega_moe_token_combine.h"
#include "mega_moe_gmm2.h"

namespace MegaMoeImpl {

// 调度普通模板当前 MoE 专家的量化 Combine，并保持原有 GMM2-ready 等待顺序。
template <uint8_t CombineMode>
__aicore__ inline void ScheduleQuantizedMoeExpertCombine(
    const QuantCombineContext &context, const QuantCombineBufferConfig &bufferConfig,
    const Params &params, const GMMAddrInfo &gmmAddrInfo, uint32_t expertTokenCount,
    uint32_t expertBeforeCnt, uint32_t expertIdx)
{
    if constexpr (g_coreType == AIC || CombineMode == COMBINE_NO_QUANT) {
        return;
    }
    if (!context.participates || context.job.totalJobs == 0U ||
        context.job.jobIndex >= context.job.totalJobs || expertTokenCount == 0U) {
        return;
    }
    uint32_t groupCount = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);
    uint32_t firstGroup = 0U;
    uint32_t groupStride = 0U;
    uint32_t jobIndexWithinGroup = 0U;
    uint32_t jobsAssignedToGroup = 0U;
    ComputeCombineGroupsForCore(context.job.jobIndex, groupCount, context.job.totalJobs, firstGroup,
                                groupStride, jobIndexWithinGroup, jobsAssignedToGroup);
    for (uint32_t groupIndex = firstGroup; groupIndex < groupCount; groupIndex += groupStride) {
        uint32_t syncSlotIndex = groupCount <= context.job.totalJobs ? context.job.jobIndex : groupIndex;
        __gm__ int32_t *syncCounterAddress =
            GetCombineSyncCounterAddress(gmmAddrInfo.gmm2CombineSyncCounter, syncSlotIndex);
        while (AscendC::ReadGmByPassDCache(syncCounterAddress) !=
               static_cast<int32_t>(context.gmm2NTilesPerGroup)) {
            int64_t waitStartCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
            }
        }

        uint32_t groupTokenStart = groupIndex * COMBINE_TOKEN_GROUP_SIZE;
        uint32_t groupTokenCount =
            COMBINE_TOKEN_GROUP_SIZE < expertTokenCount - groupTokenStart ?
                COMBINE_TOKEN_GROUP_SIZE :
                expertTokenCount - groupTokenStart;
        uint32_t tokensPerJob = Ops::Base::CeilDiv(groupTokenCount, jobsAssignedToGroup);
        uint32_t tokenOffsetWithinGroup = jobIndexWithinGroup * tokensPerJob;
        if (tokenOffsetWithinGroup >= groupTokenCount) {
            continue;
        }
        uint32_t tokenCountForJob = groupTokenCount - tokenOffsetWithinGroup;
        tokenCountForJob = tokenCountForJob < tokensPerJob ? tokenCountForJob : tokensPerJob;

        QuantCombineTokenRange tokenRange{
            groupTokenStart + tokenOffsetWithinGroup,
            tokenCountForJob,
            static_cast<uint64_t>(expertBeforeCnt) + groupTokenStart + tokenOffsetWithinGroup,
            expertIdx};
        CombineQuantizedTokenRange<CombineMode>(context, bufferConfig, params, gmmAddrInfo, tokenRange);
    }
}

// 原型：MegaMoe 的 MoE 专家 GMM2 循环。执行 GMM2 及可选的量化 Combine。
template <uint8_t CombineMode, typename GenericElementA, typename A8W4ElementA, typename WeightType,
          typename QuantScaleType, typename WeightAddressType, bool EnableA8W4, bool EnableA4W4,
          uint32_t Gmm1TileM, bool TopkWeightsPrefetch>
__aicore__ inline void ProcessMoeExpertGmm2AndCombine(
    const Gmm2Context &gmm2Context, const Gmm2Args &gmm2Args, Gmm2Scratch &gmm2Scratch,
    const QuantCombineContext &combineContext, const QuantCombineBufferConfig &combineBufferConfig,
    const Params &params, Gmm2RuntimeState &runtimeState)
{
    if (gmm2Context.blockJob.totalJobs == 0U ||
        gmm2Context.blockJob.jobIndex >= gmm2Context.blockJob.totalJobs ||
        gmm2Context.countWorkspace.blockNum == 0U ||
        gmm2Context.countWorkspace.blockIdx >= gmm2Context.countWorkspace.blockNum) {
        return;
    }
    gmm2Scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(gmm2Args.expertRevTokenNumsPtr));
    Gmm2ExpertLoopState state = CreateGmm2ExpertLoopState(gmm2Context);
    GMMAddrInfo gmmAddrInfo{};
    for (uint32_t expertIdx = 0; expertIdx < gmm2Context.expertPerRank; ++expertIdx) {
        if (!UpdateMoeGmm2GroupParams<PackedElementTraits<GenericElementA>::ELEMENTS_PER_BYTE,
                                      PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE,
                                      PackedElementTraits<A8W4ElementA>::ELEMENTS_PER_BYTE>(
                gmm2Context, gmm2Scratch, state, expertIdx)) {
            continue;
        }
        constexpr bool configureGmm2Output =
            EnableA8W4 || EnableA4W4 || CombineMode != COMBINE_NO_QUANT;
        constexpr bool configureCombineCounter = CombineMode != COMBINE_NO_QUANT;
        UpdateMoeGmm2GlobalBuffer<WeightAddressType, A8W4ElementA, QuantScaleType,
                                  configureGmm2Output, configureCombineCounter, EnableA8W4 || EnableA4W4>(
            gmm2Context, gmm2Args, gmmAddrInfo, state, expertIdx);
        RunGmm2<CombineMode, GenericElementA, A8W4ElementA, WeightType, QuantScaleType,
                EnableA8W4, EnableA4W4, Gmm1TileM, TopkWeightsPrefetch, false>(
            gmm2Context, params, gmmAddrInfo, state, runtimeState);
        ScheduleQuantizedMoeExpertCombine<CombineMode>(
            combineContext, combineBufferConfig, params, gmmAddrInfo,
            static_cast<uint32_t>(Get<M_VALUE>(state.problemShape)), state.expertBeforeCnt, expertIdx);
    }
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_H
