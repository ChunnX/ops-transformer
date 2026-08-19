/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH92_A8W8_WAVE_H
#define MEGA_MOE_ARCH92_A8W8_WAVE_H

#include "mega_moe.h"

namespace MegaMoeImpl {

// Keep the first arch92 version aligned with the proven A8W4 expert-wave policy.
constexpr uint32_t ARCH92_GMM1_TILES_PER_JOB_TARGET = 2U;
constexpr uint32_t ARCH92_MIN_EXPERTS_FOR_NEXT_WAVE = 2U;

template <uint32_t TileM>
__aicore__ inline uint32_t GetArch92Gmm1TileCount(uint32_t tokenCount, uint32_t nTileCount)
{
    return Ops::Base::CeilDiv(tokenCount, TileM) * nTileCount;
}

__aicore__ inline bool IsArch92WaveComplete(uint32_t nextExpert, uint32_t expertCount,
                                            uint32_t accumulatedTileCount, uint32_t tileTarget)
{
    return nextExpert >= expertCount ||
           (accumulatedTileCount >= tileTarget &&
            expertCount - nextExpert >= ARCH92_MIN_EXPERTS_FOR_NEXT_WAVE);
}

#define TemplateMegaMoeArch92TypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeArch92TypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, \
        IsGmm1Interleaved

template <TemplateMegaMoeArch92TypeClass>
class MegaMoeArch92A8W8Wave
    : public MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode,
                     TopkWeightsPrefetch, MegaMoeArch92ArchitecturePolicy<IsGmm1Interleaved>> {
private:
    using ArchitecturePolicy = MegaMoeArch92ArchitecturePolicy<IsGmm1Interleaved>;
    using MegaMoeBase = MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode,
                                TopkWeightsPrefetch, ArchitecturePolicy>;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaMoeBase::ActivationQuantOutType;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MegaMoeBase::A_ELEMS_PER_BYTE;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;

    template <bool IsWeightNz>
    using Gmm1PersistentConfig = GmmKernel::Config<
        false, COMBINE_NO_QUANT, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
        QuantScaleOutType, IsWeightNz, TopkWeightsPrefetch, false, false,
        IsGmm1Interleaved, false>;
    template <bool IsWeightNz>
    using Gmm2PersistentConfig = GmmKernel::Config<
        false, CombineQuantMode, ActivationQuantOutType, ActivationQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, IsWeightNz, TopkWeightsPrefetch, false, false,
        IsGmm1Interleaved, false>;
    using Gmm1NdBlockMmad = typename Gmm1PersistentConfig<false>::BlockMmad;
    using Gmm1NzBlockMmad = typename Gmm1PersistentConfig<true>::BlockMmad;
    using Gmm2NdBlockMmad = typename Gmm2PersistentConfig<false>::BlockMmad;
    using Gmm2NzBlockMmad = typename Gmm2PersistentConfig<true>::BlockMmad;
    struct EmptyMmadContext {};
    template <typename BlockMmad>
    using MmadContextStorage = std::conditional_t<
        g_coreType == AIC, GmmKernel::BlockMmadContext<BlockMmad>, EmptyMmadContext>;

    static_assert(Std::IsSame<Weight1Type, fp8_e5m2_t>::value ||
                      Std::IsSame<Weight1Type, fp8_e4m3fn_t>::value,
                  "MegaMoe arch92 first version only supports A8W8 weights");
    static_assert(Std::IsSame<QuantOutType, fp8_e5m2_t>::value ||
                      Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value,
                  "MegaMoe arch92 first version only supports A8W8 activation");
    static_assert(Std::IsSame<QuantOutType, Weight1Type>::value,
                  "MegaMoe arch92 requires matching A8W8 activation and weight dtypes");

    using MegaMoeBase::CrossRankSyncInWorldSize;
    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::InitTokenUnpermuteBuffers;
    using MegaMoeBase::ProcessSharedExpertGmm1;
    using MegaMoeBase::ProcessSharedExpertGmm2;
    using MegaMoeBase::SendAndQuantBuffInit;
    using MegaMoeBase::dispatchPrepareConfig_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::gmm1ActivationScratch_;
    using MegaMoeBase::gmm1Config_;
    using MegaMoeBase::gmm2Config_;
    using MegaMoeBase::gmm2PingPongIdx_;
    using MegaMoeBase::gmm2Scratch_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::quantCombineBufferConfig_;
    using MegaMoeBase::quantCombineConfig_;
    using MegaMoeBase::quantProcessConfig_;
    using MegaMoeBase::quantScratch_;
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

    MmadContextStorage<Gmm1NdBlockMmad> gmm1NdMmadContext_;
    MmadContextStorage<Gmm1NzBlockMmad> gmm1NzMmadContext_;
    MmadContextStorage<Gmm2NdBlockMmad> gmm2NdMmadContext_;
    MmadContextStorage<Gmm2NzBlockMmad> gmm2NzMmadContext_;

    __aicore__ inline void RunGmm1ActivationForExpert(
        ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
        int32_t &gmTileSequence, uint16_t &pingpongIdx, uint32_t expertIdx);
    __aicore__ inline void RunGmm2CombineForExpert(
        ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
        int32_t &gmTileSequence, uint32_t expertIdx);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
};

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
    int32_t &gmTileSequence, uint16_t &pingpongIdx, uint32_t expertIdx)
{
    const BlockWorkspaceContext &countWorkspace = gmm1Config_.countWorkspace;
    if constexpr (g_coreType == AIC) {
        uint64_t countSlotIndex =
            static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum + countWorkspace.blockIdx;
        __gm__ int32_t *sendCountReadyFlagAddr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
            countSlotIndex * INT_CACHELINE;
        WaitUntilGmFlagIsNonZero(sendCountReadyFlagAddr);
    }

    if (!UpdateExpertLoopStateFromWorkspace(params_.workspaceInfo, countWorkspace, state, expertIdx)) {
        return;
    }

    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType,
                                    QuantScaleOutType, A_ELEMS_PER_BYTE, false,
                                    TopkWeightsPrefetch, ArchitecturePolicy::USE_GM_HANDOFF>(
        gmm1Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_, gmmAddrInfo,
        state, expertIdx);

    int32_t unusedVecSync = 0;
    Gmm1ActivationState runtimeState{startBlockIdx, unusedVecSync, gmTileSequence, pingpongIdx};
    void *blockMmadContext = nullptr;
    if constexpr (g_coreType == AIC) {
        blockMmadContext = gmm1Config_.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ?
                               static_cast<void *>(&gmm1NzMmadContext_) :
                               static_cast<void *>(&gmm1NdMmadContext_);
    }
    RunGmm1ActivationByMode<QuantOutType, Weight1Type, ActivationQuantOutType,
                            QuantScaleOutType, false, GMM1_TILE_M, EPILOGUE_TILE_M,
                            TopkWeightsPrefetch, IsGmm1Interleaved, false,
                            ArchitecturePolicy::USE_GM_HANDOFF>(
        gmm1Config_, params_, epilogueOp_, gmmAddrInfo, state, runtimeState, expertIdx,
        blockMmadContext);
}

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
    int32_t &gmTileSequence, uint32_t expertIdx)
{
    if (!UpdateExpertLoopStateFromWorkspace(
            params_.workspaceInfo, gmm2Config_.countWorkspace, state, expertIdx)) {
        return;
    }

    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    UpdateWaveGmm2GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType,
                               QuantScaleOutType, configureCombineCounter>(
        gmm2Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo, state, expertIdx);

    int32_t unusedVecSync = 0;
    Gmm2RuntimeState runtimeState{startBlockIdx, unusedVecSync, gmTileSequence, gmm2PingPongIdx_};
    Gmm2ExpertLoopState runtimeProblemState = CreateGmm2ExpertLoopState(gmm2Config_);
    Get<M_VALUE>(runtimeProblemState.problemShape) = Get<M_VALUE>(state.problemShape);
    Get<N_VALUE>(runtimeProblemState.problemShape) = Get<N_VALUE>(state.problemShape);
    Get<K_VALUE>(runtimeProblemState.problemShape) = Get<K_VALUE>(state.problemShape);
    runtimeProblemState.expertBeforeCnt = static_cast<uint32_t>(state.rowOffset);
    void *blockMmadContext = nullptr;
    if constexpr (g_coreType == AIC) {
        blockMmadContext = gmm2Config_.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ?
                               static_cast<void *>(&gmm2NzMmadContext_) :
                               static_cast<void *>(&gmm2NdMmadContext_);
    }
    RunGmm2ByMode<CombineQuantMode, ActivationQuantOutType, ActivationQuantOutType, Weight1Type,
                  QuantScaleOutType, false, false, GMM1_TILE_M, TopkWeightsPrefetch, false,
                  IsGmm1Interleaved, false, ArchitecturePolicy::USE_GM_HANDOFF>(
        gmm2Config_, params_, gmmAddrInfo, runtimeProblemState, runtimeState, blockMmadContext, false,
        gmm2Config_.blockJob.totalJobs);

    ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
        quantCombineConfig_, quantCombineBufferConfig_, params_, gmmAddrInfo,
        static_cast<uint32_t>(Get<M_VALUE>(state.problemShape)),
        static_cast<uint32_t>(state.rowOffset), expertIdx);
}

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::ProcessMoeExpertStages(
    int32_t &gmTileSequence)
{
    DispatchBuffInit();
    gmm1ActivationScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    gmm2Scratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));

    ExpertLoopState gmm1State = CreateExpertLoopState(gmm1Config_.common);
    ExpertLoopState gmm2State = CreateExpertLoopState(gmm2Config_.common);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    uint32_t startBlockIdx = 0U;
    uint16_t gmm1PingpongIdx = 0U;

    const uint32_t expertCount = gmm1Config_.common.moeExpertPerRank;
    const uint32_t schedulerN = IsGmm1Interleaved ?
                                    gmm1Config_.common.gmm1OutputDim :
                                    gmm1Config_.common.gmm1OutputDim / ACTIVATION_N_HALF;
    const uint32_t gmm1NTileCount =
        Ops::Base::CeilDiv(schedulerN, static_cast<uint32_t>(L1_TILE_N));
    const uint32_t waveTileTarget =
        ARCH92_GMM1_TILES_PER_JOB_TARGET * gmm1Config_.blockJob.totalJobs;

    uint32_t dispatchExpert = 0U;
    if constexpr (g_coreType == AIV) {
        uint32_t dispatchedWaveTileCount = 0U;
        while (!IsArch92WaveComplete(dispatchExpert, expertCount,
                                     dispatchedWaveTileCount, waveTileTarget)) {
            uint64_t dispatchedTokenCount = 0U;
            RunMoeExpertDispatchJob<ActivationType, QuantScaleOutType,
                                    ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM, true,
                                    GMM1_TILE_M, TopkWeightsPrefetch>(
                tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_, dispatchExpert,
                dispatchedTokenCount);
            dispatchedWaveTileCount += GetArch92Gmm1TileCount<GMM1_TILE_M>(
                static_cast<uint32_t>(dispatchedTokenCount), gmm1NTileCount);
            ++dispatchExpert;
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
            nextWaveNeedsDispatch = dispatchExpert < expertCount;
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            if constexpr (g_coreType == AIV) {
                if (nextWaveNeedsDispatch) {
                    uint64_t dispatchedTokenCount = 0U;
                    RunMoeExpertDispatchJob<ActivationType, QuantScaleOutType,
                                            ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM, true,
                                            GMM1_TILE_M, TopkWeightsPrefetch>(
                        tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_, dispatchExpert,
                        dispatchedTokenCount);
                    nextDispatchWaveTileCount += GetArch92Gmm1TileCount<GMM1_TILE_M>(
                        static_cast<uint32_t>(dispatchedTokenCount), gmm1NTileCount);
                    ++dispatchExpert;
                    nextWaveNeedsDispatch = !IsArch92WaveComplete(
                        dispatchExpert, expertCount, nextDispatchWaveTileCount, waveTileTarget);
                }
            }

            if (currentWaveNeedsGmm1) {
                RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, startBlockIdx,
                                           gmTileSequence, gmm1PingpongIdx, currentGmm1Expert);
                currentWaveTileCount += GetArch92Gmm1TileCount<GMM1_TILE_M>(
                    static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape)), gmm1NTileCount);
                ++currentGmm1Expert;
                currentWaveEnd = currentGmm1Expert;
                currentWaveNeedsGmm1 = !IsArch92WaveComplete(
                    currentGmm1Expert, expertCount, currentWaveTileCount, waveTileTarget);
            }
        }

        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm2CombineForExpert(
                gmm2State, gmm2AddrInfo, startBlockIdx, gmTileSequence, expertIdx);
        }
        currentWaveBegin = currentWaveEnd;
    }
}

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, quantProcessConfig_, quantScratch_);
    GatherAndSendExpertMasks(
        dispatchPrepareConfig_, params_, winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    ResetSyncStatus<TopkWeightsPrefetch>(
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
        // FIXME: CANN 9.1.0 CopyL0C2UB parameter mismatch, skip for now.
        // ProcessSharedExpertGmm1(gmTileSequence);
    }

    CrossRankSyncInWorldSize();
    ProcessMoeExpertStages(gmTileSequence);

    if constexpr (g_coreType == AIV) {
        if (tokenDispatchConfig_.blockJob.jobIndex == 0U) {
            ExportExpertTokenCounts<ActivationType, ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM>(
                tokenDispatchConfig_, tokenDispatchScratch_, params_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm2(gmTileSequence);
    }

    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        auto unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#undef TemplateMegaMoeArch92TypeClass
#undef TemplateMegaMoeArch92TypeFunc

} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH92_A8W8_WAVE_H
