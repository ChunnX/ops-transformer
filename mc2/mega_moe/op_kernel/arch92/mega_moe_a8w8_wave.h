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

// Match the A5 A8W4 token-grained Wave target while using arch92 A8W8 Generic MMAD.
constexpr uint32_t ARCH92_GMM1_TILES_PER_JOB_TARGET = 2U;

template <uint32_t TileM>
__aicore__ inline uint32_t  AdvanceArch92ExpertTokenPositionInWave(
    uint32_t expertTokenCount, uint32_t waveMGroupTarget, uint32_t &waveMGroupCount,
    ExpertTokenPosition &position)
{
    uint32_t expertRemainingTokenCount = expertTokenCount - position.tokenIndexInExpert;
    uint32_t waveRemainingTokenCapacity = (waveMGroupTarget - waveMGroupCount) * TileM;
    uint32_t sliceTokenCount = expertRemainingTokenCount < waveRemainingTokenCapacity ?
                                   expertRemainingTokenCount :
                                   waveRemainingTokenCapacity;
    waveMGroupCount += Ops::Base::CeilDiv(sliceTokenCount, TileM);
    position.tokenIndexInExpert += sliceTokenCount;
    if (position.tokenIndexInExpert >= expertTokenCount) {
        ++position.expertIdx;
        position.tokenIndexInExpert = 0U;
    }
    return sliceTokenCount;
}

#define TemplateMegaMoeArch92TypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsWeightNz
#define TemplateMegaMoeArch92TypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, IsWeightNz

template <TemplateMegaMoeArch92TypeClass>
class MegaMoeArch92A8W8Wave
    : public MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode,
                     TopkWeightsPrefetch, MegaMoeArch92ArchitecturePolicy> {
private:
    using ArchitecturePolicy = MegaMoeArch92ArchitecturePolicy;
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

    using Gmm1PersistentConfig = GmmKernel::Config<
        false, COMBINE_NO_QUANT, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
        QuantScaleOutType, IsWeightNz, TopkWeightsPrefetch, false, false, false>;
    using Gmm2PersistentConfig = GmmKernel::Config<
        false, CombineQuantMode, ActivationQuantOutType, ActivationQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, IsWeightNz, TopkWeightsPrefetch, false, false, false>;
    using Gmm1BlockMmad = typename Gmm1PersistentConfig::BlockMmad;
    using Gmm2BlockMmad = typename Gmm2PersistentConfig::BlockMmad;
    static_assert(Std::IsSame<Gmm1BlockMmad, Gmm2BlockMmad>::value,
                  "arch92 A8W8 wave requires GMM1 and GMM2 to share one BlockMmad type");
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
    using MegaMoeBase::gmm1Config_;
    using MegaMoeBase::gmm2Config_;
    using MegaMoeBase::gmm2PingPongIdx_;
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

    MmadContextStorage<Gmm1BlockMmad> mmadContext_;
    uint32_t waveMGroupTarget_ = 0U;

    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position,
                                                uint32_t waveMGroupCount) const
    {
        return position.expertIdx < gmm1Config_.common.moeExpertPerRank &&
               waveMGroupCount < waveMGroupTarget_;
    }

    __aicore__ inline uint32_t DispatchMoeExpert(uint32_t expertIdx);
    template <bool WaitForTokenCountReady>
    __aicore__ inline void PrepareGmmExpertState(ExpertLoopState &state, uint32_t expertIdx);
    __aicore__ inline void RunGmm1ActivationForExpert(
        ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
        int32_t &gmTileSequence, uint32_t expertIdx, uint32_t tokenStartIndexInExpert,
        uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void RunGmm2CombineForExpert(
        ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
        int32_t &gmTileSequence, uint32_t expertIdx, uint32_t tokenStartIndexInExpert,
        uint32_t sliceTokenCount);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
};

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline uint32_t MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::DispatchMoeExpert(
    uint32_t expertIdx)
{
    uint64_t expertTokenCount = 0U;
    RunMoeExpertDispatchJob<ActivationType, QuantScaleOutType,
                            ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM, true,
                            GMM1_TILE_M, TopkWeightsPrefetch>(
        tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_, expertIdx,
        expertTokenCount);
    return static_cast<uint32_t>(expertTokenCount);
}

template <TemplateMegaMoeArch92TypeClass>
template <bool WaitForTokenCountReady>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::PrepareGmmExpertState(
    ExpertLoopState &state, uint32_t expertIdx)
{
    const BlockWorkspaceContext &countWorkspace = gmm1Config_.countWorkspace;
    if constexpr (WaitForTokenCountReady && g_coreType == AIC) {
        WaitForMoeExpertTokenCountReady(
            params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace, expertIdx);
    }
    uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
        params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace, expertIdx);
    UpdateExpertLoopState(state, expertIdx, expertTokenCount);
}

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
    int32_t &gmTileSequence, uint32_t expertIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup)
{
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType,
                                    QuantScaleOutType, A_ELEMS_PER_BYTE, false,
                                    TopkWeightsPrefetch, ArchitecturePolicy::USE_GM_HANDOFF>(
        gmm1Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_, gmmAddrInfo,
        state, expertIdx, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    gmmAddrInfo.debugRankId = gmm1Config_.common.rankId;

    int32_t unusedVecSync = 0;
    Gmm1ActivationState runtimeState{startBlockIdx, unusedVecSync, gmTileSequence};
    void *blockMmadContext = nullptr;
    if constexpr (g_coreType == AIC) {
        blockMmadContext = static_cast<void *>(&mmadContext_);
    }
    RunGmm1ActivationByMode<QuantOutType, Weight1Type, ActivationQuantOutType,
                            QuantScaleOutType, false, GMM1_TILE_M, EPILOGUE_TILE_M,
                            TopkWeightsPrefetch, true,
                            ArchitecturePolicy::USE_GM_HANDOFF>(
        gmm1Config_, params_, epilogueOp_, gmmAddrInfo, state, runtimeState, expertIdx,
        blockMmadContext, false, tokenStartIndexInExpert, sliceTokenCount);
}

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
    int32_t &gmTileSequence, uint32_t expertIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    UpdateWaveGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                               configureCombineCounter>(
        gmm2Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo, state,
        expertIdx, tokenStartIndexInExpert);
    gmmAddrInfo.debugRankId = gmm2Config_.common.rankId;

    int32_t unusedVecSync = 0;
    Gmm2RuntimeState runtimeState{startBlockIdx, unusedVecSync, gmTileSequence, gmm2PingPongIdx_};
    Gmm2ExpertLoopState runtimeProblemState = CreateGmm2ExpertLoopState(gmm2Config_);
    Get<M_VALUE>(runtimeProblemState.problemShape) = sliceTokenCount;
    Get<N_VALUE>(runtimeProblemState.problemShape) = Get<N_VALUE>(state.problemShape);
    Get<K_VALUE>(runtimeProblemState.problemShape) = Get<K_VALUE>(state.problemShape);
    runtimeProblemState.expertBeforeCnt =
        static_cast<uint32_t>(state.rowOffset) + tokenStartIndexInExpert;
    void *blockMmadContext = nullptr;
    if constexpr (g_coreType == AIC) {
        blockMmadContext = static_cast<void *>(&mmadContext_);
    }
    RunGmm2ByMode<CombineQuantMode, ActivationQuantOutType, ActivationQuantOutType, Weight1Type,
                  QuantScaleOutType, false, false, GMM1_TILE_M, TopkWeightsPrefetch, false,
                  true, ArchitecturePolicy::USE_GM_HANDOFF>(
        gmm2Config_, params_, gmmAddrInfo, runtimeProblemState, runtimeState, blockMmadContext, false,
        gmm2Config_.blockJob.totalJobs, expertTokenCount, tokenStartIndexInExpert);

    ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
        quantCombineConfig_, quantCombineBufferConfig_, params_, gmmAddrInfo,
        expertTokenCount, static_cast<uint32_t>(state.rowOffset), expertIdx,
        tokenStartIndexInExpert, sliceTokenCount);
}

template <TemplateMegaMoeArch92TypeClass>
__aicore__ inline void MegaMoeArch92A8W8Wave<TemplateMegaMoeArch92TypeFunc>::ProcessMoeExpertStages(
    int32_t &gmTileSequence)
{
    DispatchBuffInit();

    ExpertLoopState gmm1State = CreateExpertLoopState(gmm1Config_.common);
    ExpertLoopState gmm2State = CreateExpertLoopState(gmm2Config_.common);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    uint32_t startBlockIdx = 0U;

    const uint32_t schedulerN = gmm1Config_.common.gmm1OutputDim / ACTIVATION_N_HALF;
    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(schedulerN, static_cast<uint32_t>(L1_TILE_N));
    waveMGroupTarget_ = Ops::Base::CeilDiv(
        ARCH92_GMM1_TILES_PER_JOB_TARGET * gmm1Config_.blockJob.totalJobs,
        gmm1TilesPerMGroup);

    ExpertTokenPosition dispatchPosition{};
    if constexpr (g_coreType == AIV) {
        uint32_t dispatchedWaveMGroupCount = 0U;
        while (IsPositionWithinWave(dispatchPosition, dispatchedWaveMGroupCount)) {
            uint32_t expertTokenCount = dispatchPosition.tokenIndexInExpert == 0U ?
                                            DispatchMoeExpert(dispatchPosition.expertIdx) :
                                            GetExpertTokenCountFromWorkspace(
                                                params_.workspaceInfo.expertRevTokenNumsPtr,
                                                gmm1Config_.countWorkspace,
                                                dispatchPosition.expertIdx);
            AdvanceArch92ExpertTokenPositionInWave<GMM1_TILE_M>(
                expertTokenCount, waveMGroupTarget_, dispatchedWaveMGroupCount,
                dispatchPosition);
        }
    }

    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < gmm1Config_.common.moeExpertPerRank) {
        ExpertTokenPosition waveBeginPosition = gmm1Position;
        ExpertTokenPosition waveEndPosition = gmm1Position;
        uint32_t currentWaveMGroupCount = 0U;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            nextWaveNeedsDispatch =
                IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            if constexpr (g_coreType == AIV) {
                if (nextWaveNeedsDispatch) {
                    uint32_t expertTokenCount = dispatchPosition.tokenIndexInExpert == 0U ?
                                                    DispatchMoeExpert(dispatchPosition.expertIdx) :
                                                    GetExpertTokenCountFromWorkspace(
                                                        params_.workspaceInfo.expertRevTokenNumsPtr,
                                                        gmm1Config_.countWorkspace,
                                                        dispatchPosition.expertIdx);
                    AdvanceArch92ExpertTokenPositionInWave<GMM1_TILE_M>(
                        expertTokenCount, waveMGroupTarget_, nextDispatchWaveMGroupCount,
                        dispatchPosition);
                    nextWaveNeedsDispatch =
                        IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                uint32_t expertTokenCount =
                    static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                uint32_t currentExpertIdx = gmm1Position.expertIdx;
                uint32_t sliceTokenStartIndexInExpert = gmm1Position.tokenIndexInExpert;
                uint32_t sliceTokenCount = AdvanceArch92ExpertTokenPositionInWave<GMM1_TILE_M>(
                    expertTokenCount, waveMGroupTarget_, currentWaveMGroupCount, gmm1Position);
                if (sliceTokenCount != 0U) {
                    RunGmm1ActivationForExpert(
                        gmm1State, gmm1AddrInfo, startBlockIdx, gmTileSequence, currentExpertIdx,
                        sliceTokenStartIndexInExpert, sliceTokenCount, gmm1TilesPerMGroup);
                }
                waveEndPosition = gmm1Position;
                currentWaveNeedsGmm1 =
                    IsPositionWithinWave(gmm1Position, currentWaveMGroupCount);
            }
        }

        uint32_t waveGmm2ExpertEndExclusive =
            waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
        for (uint32_t expertIdx = waveBeginPosition.expertIdx;
             expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
            uint32_t sliceTokenStartIndexInExpert =
                expertIdx == waveBeginPosition.expertIdx ?
                    waveBeginPosition.tokenIndexInExpert :
                    0U;
            if (sliceTokenStartIndexInExpert == 0U) {
                PrepareGmmExpertState<false>(gmm2State, expertIdx);
            }
            uint32_t expertTokenCount =
                static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
            uint32_t sliceTokenEndIndexInExpert =
                expertIdx == waveEndPosition.expertIdx ?
                    waveEndPosition.tokenIndexInExpert :
                    expertTokenCount;
            uint32_t sliceTokenCount =
                sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
            if (sliceTokenCount != 0U) {
                RunGmm2CombineForExpert(
                    gmm2State, gmm2AddrInfo, startBlockIdx, gmTileSequence, expertIdx,
                    sliceTokenStartIndexInExpert, sliceTokenCount);
            }
        }
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
        ProcessSharedExpertGmm1(gmTileSequence);
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
