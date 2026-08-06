/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_WAVE_ROUTED_EXPERT_PIPELINE_H
#define MEGA_MOE_WAVE_ROUTED_EXPERT_PIPELINE_H

#include "aiv_compute/mega_moe_expert_token_count.h"
#include "aiv_comm/mega_moe_token_dispatch.h"
#include "mega_moe_gmm1_swiglu.h"
#include "mega_moe_gmm2.h"
#include "mega_moe_utils.h"
#include "mega_moe_wave_gmm2_combine.h"

namespace MegaMoeImpl {

// Prototype: MegaMoeWave::DispatchExpert / DispatchExpertsUntil.
// Advances the AIV1 Dispatch cursor and prefetches every newly exposed expert exactly once.
template <typename ActivationType, typename QuantScaleType, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchWaveExpertsUntil(
    const TokenDispatchContext &context, const TokenDispatchArgs &args,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t &nextDispatchExpert,
    uint32_t dispatchEnd)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 1U) {
            return;
        }
        for (; nextDispatchExpert < dispatchEnd; ++nextDispatchExpert) {
            uint64_t sendCnt = 0U;
            ComputeExpertTokenCountAndNotify<ActivationType, false, TopkWeightsPrefetch>(
                context, args, scratch, nextDispatchExpert, sendCnt);
            if (sendCnt != 0U) {
                DispatchExpertTokens<ActivationType, QuantScaleType, false, Gmm1TileM,
                                     TopkWeightsPrefetch>(
                    context, args, scratch, nextDispatchExpert);
            }
        }
    }
}

// Prototype: MegaMoeWave::ProcessGmm1Wave.
// Runs one expert wave on AIC/AIV0 while AIV1 remains dedicated to Dispatch and Combine.
template <typename QuantOutType, typename ActivationType, typename WeightType, typename QuantScaleType,
          uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved, typename BlockEpilogue>
__aicore__ inline void RunWaveGmm1Stage(
    const Gmm1SwigluContext &context, const Gmm1SwigluArgs &args, const Params &params,
    BlockEpilogue &epilogueOp, Gmm1SwigluScratch &scratch, uint32_t batchBegin,
    uint32_t batchEnd, Gmm1ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
    Gmm1SwigluState &runtimeState)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }
    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
        if (!WaitAndUpdateGmm1GroupParams<false>(context, args, scratch, state, expertIdx, 0U)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, WeightType, ActivationType, QuantScaleType, false,
                               TopkWeightsPrefetch>(
            context, args, epilogueOp, gmmAddrInfo, state, expertIdx);
        ControlGmm1SwigluPipelineByMode<QuantOutType, WeightType, ActivationType, QuantScaleType,
                                        false, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                        IsGmm1Interleaved, true>(
            context, params, epilogueOp, gmmAddrInfo, state, runtimeState, expertIdx);
    }
}

// Prototype: MegaMoeWave::ProcessGmm2Wave.
// Runs one expert wave on AIC and advances the matching virtual GMM2 cursor on AIV0.
template <typename QuantOutType, typename ActivationType, typename WeightType, typename QuantScaleType,
          uint32_t Gmm1TileM, bool TopkWeightsPrefetch, bool IsGmm1Interleaved>
__aicore__ inline void RunWaveGmm2Stage(
    const Gmm2Context &context, const Gmm2Args &args, Gmm2Scratch &scratch,
    const WaveCombineContext &combineContext, const WaveGmm2ReadyArgs &readyArgs,
    const Params &params, uint32_t batchBegin, uint32_t batchEnd, Gmm2ExpertLoopState &state,
    GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom,
    uint16_t &gmm2PingPongIdx)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }
    int32_t gmTileSequence = 0;
    Gmm2RuntimeState runtimeState{startBlockIdx, vecSetSyncCom, gmTileSequence, gmm2PingPongIdx};
    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
        if (!UpdateRoutedGmm2GroupParams<1U, 1U, 1U>(context, scratch, state, expertIdx)) {
            continue;
        }
        UpdateRoutedGmm2GlobalBuffer<ActivationType, ActivationType, QuantScaleType,
                                     true, false, false>(
            context, args, gmmAddrInfo, state, expertIdx);
        RunGmm2<COMBINE_NO_QUANT, QuantOutType, ActivationType, WeightType,
                QuantScaleType, false, false, Gmm1TileM, TopkWeightsPrefetch, false,
                IsGmm1Interleaved, true>(
            context, params, gmmAddrInfo, state, runtimeState);
        NotifyWaveGmm2Ready(combineContext, readyArgs, expertIdx);
    }
}

// Prototype: MegaMoeWave::ProcessCombineExperts.
// Waits for one wave's GMM2 fan-in and sends the token ranges owned by AIV1 jobs.
template <uint8_t CombineMode, typename ActivationType, typename QuantScaleType>
__aicore__ inline void RunWaveCombineStage(
    const Gmm2Context &gmm2Context, const Gmm2Args &gmm2Args, Gmm2Scratch &gmm2Scratch,
    const WaveCombineContext &combineContext, const WaveCombineArgs &combineArgs,
    const WaveGmm2ReadyArgs &readyArgs, const WaveCombineBufferConfig &bufferConfig,
    WaveCombineScratch &combineScratch, const Params &params, uint32_t batchBegin,
    uint32_t batchEnd, Gmm2ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    uint32_t rowSequence = 0U;
    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
        if (!UpdateRoutedGmm2GroupParams<1U, 1U, 1U>(
                gmm2Context, gmm2Scratch, state, expertIdx)) {
            continue;
        }
        UpdateRoutedGmm2GlobalBuffer<ActivationType, ActivationType, QuantScaleType,
                                     true, false, false>(
            gmm2Context, gmm2Args, gmmAddrInfo, state, expertIdx);
        uint32_t currentExpertTokenNum = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
        TokenRange currentCoreTokenRange =
            GetWaveCombineOwnedRange(combineContext, currentExpertTokenNum);
        if (currentCoreTokenRange.count != 0U) {
            WaitWaveGmm2Ready(combineContext, readyArgs, expertIdx);
        }

        for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
            uint32_t remainingTokenNum = currentCoreTokenRange.count - processedTokenNum;
            uint32_t chunkTokenNum =
                remainingTokenNum < WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ?
                    remainingTokenNum :
                    WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
            PreloadWaveCombineMetaInfo(
                combineArgs, combineScratch,
                static_cast<uint64_t>(state.expertBeforeCnt) + currentCoreTokenRange.start + processedTokenNum,
                chunkTokenNum, 0U);
            CombineWaveTokenRange<CombineMode>(
                combineContext, bufferConfig, combineScratch, params, gmmAddrInfo.gmm2OutGlobal,
                currentCoreTokenRange.start + processedTokenNum, chunkTokenNum, 0U, rowSequence);
            processedTokenNum += chunkTokenNum;
        }
    }
    DrainWaveCombineRowRing<CombineMode>(rowSequence);
}

// Prototype: MegaMoeWave::ProcessRoutedExpertWaves.
// Controls the rolling Dispatch -> GMM1/SwiGLU -> GMM2 -> Combine pipeline:
//   AIC  : GMM1(B) -> GMM2(B) -> GMM1(B+1)
//   AIV0 : SwiGLU(B) -> SwiGLU(B+1)
//   AIV1 : Dispatch(B+1) -> Combine(B)
template <uint8_t CombineMode, typename QuantOutType, typename ActivationType, typename WeightType,
          typename QuantScaleType, uint32_t Gmm1TileM, uint32_t EpilogueTileM,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved, typename BlockEpilogue>
__aicore__ inline void ControlWaveRoutedExpertPipeline(
    const TokenDispatchContext &dispatchContext, const TokenDispatchArgs &dispatchArgs,
    const Gmm1SwigluContext &gmm1Context, const Gmm1SwigluArgs &gmm1Args,
    const Gmm2Context &gmm2Context, const Gmm2Args &gmm2Args,
    const WaveCombineContext &combineContext, const WaveCombineArgs &combineArgs,
    const WaveGmm2ReadyArgs &readyArgs, const ExpertTokenCountExportContext &exportContext,
    const ExpertTokenCountExportArgs &exportArgs, const Params &params, BlockEpilogue &epilogueOp,
    TokenDispatchScratch<ActivationType> &dispatchScratch, Gmm1SwigluScratch &gmm1Scratch,
    Gmm2Scratch &gmm2Scratch, WaveCombineScratch &combineScratch,
    ExpertTokenCountExportScratch &exportScratch, const WaveCombineBufferConfig &combineBufferConfig,
    uint32_t expertsPerWave, uint32_t &startBlockIdx, uint16_t &gmm1PingPongIdx,
    uint16_t &gmm2PingPongIdx)
{
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    GMMAddrInfo combineAddrInfo{};
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(gmm1Context);
    Gmm2ExpertLoopState gmm2State = CreateGmm2ExpertLoopState(gmm2Context);
    Gmm2ExpertLoopState combineState = CreateGmm2ExpertLoopState(gmm2Context);
    int32_t vecSetSyncCom = 0;
    int32_t gmm1TileSequence = 0;
    Gmm1SwigluState gmm1RuntimeState{
        startBlockIdx, vecSetSyncCom, gmm1TileSequence, gmm1PingPongIdx};

    uint32_t nextDispatchExpert = 0U;
    uint32_t currentWaveExpertNum = 0U;
    for (uint32_t batchBegin = 0U; batchBegin < gmm2Context.expertPerRank;
         batchBegin += currentWaveExpertNum) {
        currentWaveExpertNum =
            GetWaveExpertCount(batchBegin, gmm2Context.expertPerRank, expertsPerWave);
        const uint32_t batchEnd = batchBegin + currentWaveExpertNum;
        const uint32_t waveBeginCursor = startBlockIdx;

        const uint32_t nextWaveExpertNum =
            GetWaveExpertCount(batchEnd, gmm2Context.expertPerRank, expertsPerWave);
        const uint32_t nextBatchEnd = batchEnd + nextWaveExpertNum;
        // Dispatch and Combine reuse UB/event resources, so AIV1 prefetches the next wave before combining this one.
        DispatchWaveExpertsUntil<ActivationType, QuantScaleType, Gmm1TileM, TopkWeightsPrefetch>(
            dispatchContext, dispatchArgs, dispatchScratch, nextDispatchExpert, nextBatchEnd);

        RunWaveGmm1Stage<QuantOutType, ActivationType, WeightType, QuantScaleType,
                         Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, IsGmm1Interleaved>(
            gmm1Context, gmm1Args, params, epilogueOp, gmm1Scratch, batchBegin, batchEnd,
            gmm1State, gmm1AddrInfo, gmm1RuntimeState);
        const uint32_t gmm1EndCursor = startBlockIdx;

        RunWaveGmm2Stage<QuantOutType, ActivationType, WeightType, QuantScaleType,
                         Gmm1TileM, TopkWeightsPrefetch, IsGmm1Interleaved>(
            gmm2Context, gmm2Args, gmm2Scratch, combineContext, readyArgs, params,
            batchBegin, batchEnd, gmm2State, gmm2AddrInfo, startBlockIdx, vecSetSyncCom,
            gmm2PingPongIdx);
        // Avoid a fixed GMM1/GMM2 physical-core role when both cursors cancel over one complete wave.
        const bool hasNextWave = batchEnd < gmm2Context.expertPerRank;
        const bool fixedRoleResonance =
            startBlockIdx == waveBeginCursor && gmm1EndCursor != waveBeginCursor;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx = gmm1EndCursor;
        }

        RunWaveCombineStage<CombineMode, ActivationType, QuantScaleType>(
            gmm2Context, gmm2Args, gmm2Scratch, combineContext, combineArgs, readyArgs,
            combineBufferConfig, combineScratch, params, batchBegin, batchEnd, combineState,
            combineAddrInfo);
    }

    if constexpr (!TopkWeightsPrefetch) {
        EndSync<IsGmm1Interleaved>(vecSetSyncCom, gmm1PingPongIdx);
    }
    vecSetSyncCom = 0;
    gmm1PingPongIdx = 0;
    ExportExpertTokenCountsFromWorkspace(exportContext, exportArgs, exportScratch);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WAVE_ROUTED_EXPERT_PIPELINE_H
