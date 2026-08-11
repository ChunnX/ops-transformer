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
 * \file mega_moe_wave_a8w4.h
 * \brief MegaMoe A8W4 dynamic-wave orchestration
 */

#ifndef MEGA_MOE_WAVE_A8W4_H
#define MEGA_MOE_WAVE_A8W4_H

#include "mega_moe.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA8W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

// Each regular wave targets two complete scheduler rounds; a one-expert tail is merged into the current wave.
constexpr uint32_t A8W4_GMM1_TILES_PER_JOB_TARGET = 2U;
constexpr uint32_t MIN_EXPERTS_FOR_NEXT_A8W4_WAVE = 2U;

template <uint32_t TileM>
__aicore__ inline uint32_t GetA8W4Gmm1TileCount(uint32_t tokenCount, uint32_t nTileCount)
{
    return Ops::Base::CeilDiv(tokenCount, TileM) * nTileCount;
}

__aicore__ inline bool IsA8W4WaveComplete(uint32_t nextExpert, uint32_t expertCount,
                                          uint32_t accumulatedTileCount, uint32_t tileTarget)
{
    return nextExpert >= expertCount ||
           (accumulatedTileCount >= tileTarget &&
            expertCount - nextExpert >= MIN_EXPERTS_FOR_NEXT_A8W4_WAVE);
}

template <TemplateMegaMoeA8W4WaveTypeClass>
class MegaMoeA8W4Wave : public MegaMoe<TemplateMegaMoeA8W4WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA8W4WaveTypeFunc>;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using SwigluQuantOutType = typename MegaMoeBase::SwigluQuantOutType;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MegaMoeBase::A_ELEMS_PER_BYTE;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;

    static_assert(Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                      Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value,
                  "MegaMoeA8W4Wave requires fp8 activation and fp4 weight");

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::CrossRankSyncInWorldSize;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::InitTokenUnpermuteBuffers;
    using MegaMoeBase::ProcessSharedExpertGmm1;
    using MegaMoeBase::ProcessSharedExpertGmm2;
    using MegaMoeBase::SendAndQuantBuffInit;
    using MegaMoeBase::dispatchPrepareConfig_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::gmm1Config_;
    using MegaMoeBase::gmm2Config_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::quantProcessConfig_;
    using MegaMoeBase::quantScratch_;
    using MegaMoeBase::quantCombineBufferConfig_;
    using MegaMoeBase::quantCombineConfig_;
    using MegaMoeBase::resetTensor_;
    using MegaMoeBase::resetWorkspaceConfig_;
    using MegaMoeBase::sendMaskConfig_;
    using MegaMoeBase::sendMaskScratch_;
    using MegaMoeBase::sharedExpertNum_;
    using MegaMoeBase::sharedExpertPrepareConfig_;
    using MegaMoeBase::sharedExpertPrepareScratch_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::tokenUnpermuteConfig_;
    using MegaMoeBase::tokenUnpermuteScratch_;
    using MegaMoeBase::winRankAddr_;

    __aicore__ inline void RunGmm1SwigluForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                   uint32_t expertIdx);
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                   uint32_t expertIdx);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
};

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // Input preparation: quantize local tokens, publish route masks, and reset the shared synchronization state.
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, quantProcessConfig_, quantScratch_);
    GatherAndSendExpertMasks(dispatchPrepareConfig_, params_, winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    ResetDispatchWorkspace<TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, resetWorkspaceConfig_, resetTensor_);
    if (sharedExpertNum_ > 0U) {
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType, A_ELEMS_PER_BYTE>(
            dispatchPrepareConfig_, params_, sharedExpertPrepareConfig_, sharedExpertPrepareScratch_);
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();

    int32_t gmTileSequence = 0;
    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm1(gmTileSequence);
    }

    CrossRankSyncInWorldSize();

    // MoE expert pipeline: dynamic-wave Dispatch, GMM1/SwiGLU, GMM2, and Combine.
    ProcessMoeExpertStages(gmTileSequence);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && countWorkspace_.blockIdx == 0U) {
            ExportExpertTokenCounts<ActivationType, true>(tokenDispatchConfig_, tokenDispatchScratch_, params_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm2(gmTileSequence);
    }

    // Output aggregation starts after all ranks finish sending their Combine results.
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

/*
 * Dynamic Wave advances Dispatch and GMM1 independently. AIV1 publishes each expert's token count to GM;
 * AIC and AIV0 wait for the corresponding ready flag before reading it. The published count determines the
 * current expert's row range and whether its GMM1/SwiGLU stage needs to run.
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1SwigluForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t expertIdx)
{
    uint64_t countSlotIndex =
        static_cast<uint64_t>(expertIdx) * countWorkspace_.blockNum + countWorkspace_.blockIdx;
    if (GetSubBlockIdx() == 0U) {
        __gm__ int32_t *sendCountReadyFlagAddr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
            countSlotIndex * INT_CACHELINE;
        while (AscendC::ReadGmByPassDCache(sendCountReadyFlagAddr) == 0) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
    }

    if (!UpdateExpertLoopStateFromWorkspace(params_.workspaceInfo, countWorkspace_, state, expertIdx)) {
        return;
    }

    UpdateGmm1GlobalBuffer<ActivationType, Weight1Type, SwigluQuantOutType, QuantScaleOutType,
                           A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmm1Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_, gmmAddrInfo, state, expertIdx);
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, false, false>(
        epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence,
        gmm1Config_.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
}

// Execute GMM2 and subsequent Combine scheduling for one expert in the completed wave.
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t expertIdx)
{
    if (!UpdateExpertLoopStateFromWorkspace(params_.workspaceInfo, countWorkspace_, state, expertIdx)) {
        return;
    }

    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    UpdateGmm2GlobalBuffer<ActivationType, Weight1Type, SwigluQuantOutType, QuantScaleOutType,
                           true, configureCombineCounter, true>(
        gmm2Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo, state, expertIdx);
    RunGmm2A8W4<CombineQuantMode, SwigluQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                 QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch, false>(
        params_, state.problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, gmm2Config_.blockJob);
    ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
        quantCombineConfig_, quantCombineBufferConfig_, params_, gmmAddrInfo,
        static_cast<uint32_t>(Get<M_VALUE>(state.problemShape)), static_cast<uint32_t>(state.rowOffset), expertIdx);
}

/*
 * Divide experts into dynamic waves by GMM1 scheduler load. The startup phase dispatches the first complete
 * wave. During steady state AIV1 alternates one expert of the next wave's Dispatch with one expert of the
 * current wave's SwiGLU, while AIC/AIV0 execute current-wave GMM1. GMM2/Combine starts as soon as the current
 * wave finishes, instead of waiting for every expert's GMM1.
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages(
    int32_t &gmTileSequence)
{
    DispatchBuffInit();

    ExpertLoopState gmm1State = CreateExpertLoopState(gmm1Config_.common);
    ExpertLoopState gmm2State = CreateExpertLoopState(gmm2Config_.common);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // A8W4 GMM1 and GMM2 share the scheduler cursor, matching their execution order across wave boundaries.
    uint32_t startBlockIdx = 0U;

    const uint32_t expertCount = gmm1Config_.common.moeExpertPerRank;
    const uint32_t gmm1NTileCount = Ops::Base::CeilDiv(
        gmm1Config_.common.gmm1OutputDim / SWIGLU_N_HALF, static_cast<uint32_t>(L1_TILE_N));
    // Two scheduler tiles per logical AIC job provide the minimum load used by the validated wave policy.
    const uint32_t waveTileTarget =
        A8W4_GMM1_TILES_PER_JOB_TARGET * gmm1Config_.blockJob.totalJobs;

    uint32_t dispatchExpert = 0U;

    // Startup: AIV1 dispatches the first complete wave before its GMM1 begins consuming input.
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            uint32_t dispatchedWaveTileCount = 0U;
            while (!IsA8W4WaveComplete(dispatchExpert, expertCount, dispatchedWaveTileCount, waveTileTarget)) {
                uint64_t dispatchedTokenCount = 0U;
                RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, true, GMM1_TILE_M,
                                          TopkWeightsPrefetch>(
                    tokenDispatchConfig_, countWorkspace_, params_, winRankAddr_, tokenDispatchScratch_, dispatchExpert,
                    dispatchedTokenCount);
                dispatchedWaveTileCount += GetA8W4Gmm1TileCount<GMM1_TILE_M>(
                    static_cast<uint32_t>(dispatchedTokenCount), gmm1NTileCount);
                ++dispatchExpert;
            }
        }
    }

    uint32_t currentWaveBegin = 0U;
    while (currentWaveBegin < expertCount) {
        uint32_t currentGmm1Expert = currentWaveBegin;
        uint32_t currentWaveEnd = currentWaveBegin;
        uint32_t currentWaveTileCount = 0U;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveTileCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            nextWaveNeedsDispatch = GetSubBlockIdx() == 1U && dispatchExpert < expertCount;
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 alternates next-wave Dispatch with current-wave SwiGLU instead of dispatching in one burst.
            if constexpr (g_coreType == AIV) {
                if (nextWaveNeedsDispatch) {
                    uint64_t dispatchedTokenCount = 0U;
                    RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, true, GMM1_TILE_M,
                                              TopkWeightsPrefetch>(
                        tokenDispatchConfig_, countWorkspace_, params_, winRankAddr_, tokenDispatchScratch_,
                        dispatchExpert, dispatchedTokenCount);
                    nextDispatchWaveTileCount += GetA8W4Gmm1TileCount<GMM1_TILE_M>(
                        static_cast<uint32_t>(dispatchedTokenCount), gmm1NTileCount);
                    ++dispatchExpert;
                    nextWaveNeedsDispatch = !IsA8W4WaveComplete(
                        dispatchExpert, expertCount, nextDispatchWaveTileCount, waveTileTarget);
                }
            }

            if (currentWaveNeedsGmm1) {
                RunGmm1SwigluForExpert(
                    gmm1State, gmm1AddrInfo, startBlockIdx, gmTileSequence, currentGmm1Expert);
                currentWaveTileCount += GetA8W4Gmm1TileCount<GMM1_TILE_M>(
                    static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape)), gmm1NTileCount);
                ++currentGmm1Expert;
                currentWaveEnd = currentGmm1Expert;
                currentWaveNeedsGmm1 = !IsA8W4WaveComplete(
                    currentGmm1Expert, expertCount, currentWaveTileCount, waveTileTarget);
            }
        }
        // Consume the completed wave immediately so its GMM2/Combine overlaps the next wave's prefetched input.
        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx, gmTileSequence, expertIdx);
        }
        currentWaveBegin = currentWaveEnd;
    }
}

} // namespace MegaMoeImpl

#undef TemplateMegaMoeA8W4WaveTypeClass
#undef TemplateMegaMoeA8W4WaveTypeFunc

#endif
