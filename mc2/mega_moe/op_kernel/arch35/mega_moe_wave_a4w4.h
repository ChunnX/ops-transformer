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
 * \file mega_moe_wave_a4w4.h
 * \brief MegaMoe A4W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A4W4_H
#define MEGA_MOE_WAVE_A4W4_H

#include "common/mega_moe_utils.h"
#include "mega_moe_arch35.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA4W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA4W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

// 每个逻辑 AIC 任务以两个 GMM1 scheduler tile 作为 Wave 目标负载。
constexpr uint32_t A4W4_GMM1_TILES_PER_JOB_TARGET = 2U;

template <TemplateMegaMoeA4W4WaveTypeClass>
class MegaMoeA4W4Wave : public MegaMoe<TemplateMegaMoeA4W4WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA4W4WaveTypeFunc>;

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

    static_assert(Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value,
                  "MegaMoeA4W4Wave requires fp4 activation and fp4 weight");

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::DispatchMoeExpert;
    using MegaMoeBase::InitTokenUnpermuteBuffers;
    using MegaMoeBase::ProcessInputPreparationStage;
    using MegaMoeBase::ProcessSharedExpertGmm1;
    using MegaMoeBase::ProcessSharedExpertGmm2;
    using MegaMoeBase::PrepareGmmExpertState;
    using MegaMoeBase::aivJob_;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::quantCombineBufferConfig_;
    using MegaMoeBase::quantCombineConfig_;
    using MegaMoeBase::rankId_;
    using MegaMoeBase::sharedExpertNum_;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::tokenUnpermuteConfig_;
    using MegaMoeBase::tokenUnpermuteScratch_;
    using MegaMoeBase::worldSize_;

    uint32_t waveMGroupTarget_ = 0U;

    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < waveMGroupTarget_;
    }

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      GmmRuntimeState &runtimeState, uint32_t tokenStartIndexInExpert,
                                                      uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                   uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
};

// A4W4 的 GMM1 使用 generic kernel，Activation 将中间结果提升为 FP8 后供 GMM2 使用。
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, GmmRuntimeState &runtimeState, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup)
{
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, false, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    RunGmm1GenericByWeightFormat<QuantOutType, ActivationQuantOutType, QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                 TopkWeightsPrefetch, false, true>(
        gmmExecutionConfig_, params_, epilogueOp_, gmmAddrInfo, sliceProblemShape,
        static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert, runtimeState, state.expertIdx);
}

// A4W4 的 GMM2 复用 A8W4 prologue，将 FP4 weight 解包为 FP8 后执行矩阵乘与 Combine。
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType, true,
                                    configureCombineCounter, true>(gmmExecutionConfig_, syncWorkspaceLayout_,
                                                                   params_.workspaceInfo, moeWeightTensorListAddrs_,
                                                                   gmmAddrInfo, state, tokenStartIndexInExpert);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    RunGmm2A8W4<CombineQuantMode, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                GMM1_TILE_M, TopkWeightsPrefetch, false, false, true>(
        params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, gmmExecutionConfig_.blockJob,
        expertTokenCount, tokenStartIndexInExpert);
    ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
        quantCombineConfig_, quantCombineBufferConfig_, params_, gmmAddrInfo, expertTokenCount,
        static_cast<uint32_t>(state.globalTokenStartIndex), state.expertIdx, tokenStartIndexInExpert, sliceTokenCount);
}

/*
 * 按 GMM1 调度负载将专家划分为动态 Wave。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段由 AIV1
 * 逐专家 Dispatch 下一 Wave，同时 AIC/AIV0 执行当前 Wave 的 GMM1/Activation。当前 Wave 完成后
 * 立即启动 GMM2/Combine，无需等待所有专家的 GMM1。
 */
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::ProcessMoeExpertStages(int32_t &gmTileSequence)
{
    DispatchBuffInit();

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // 同一 Block 内绑定的 1C2V 各自持有分核游标，必须按相同调用顺序推进并始终保持一致。
    // 某个角色即使不参与当前 GMM 计算，也必须使用该 problem 的 tile 数更新游标。
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    GmmRuntimeState gmm1RuntimeState{startBlockIdx, vecSetSyncCom, gmTileSequence, gmm1PingPongIdx};

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));
    waveMGroupTarget_ =
        Ops::Base::CeilDiv(A4W4_GMM1_TILES_PER_JOB_TARGET * gmmExecutionConfig_.blockJob.totalJobs, gmm1TilesPerMGroup);

    ExpertTokenPosition dispatchPosition{};

    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            uint32_t dispatchedWaveMGroupCount = 0U;
            while (IsPositionWithinWave(dispatchPosition, dispatchedWaveMGroupCount)) {
                uint32_t expertTokenCount =
                    dispatchPosition.tokenIndexInExpert == 0U ?
                        DispatchMoeExpert(dispatchPosition.expertIdx) :
                        GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                                                         dispatchPosition.expertIdx);
                AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, waveMGroupTarget_,
                                                              dispatchedWaveMGroupCount, dispatchPosition);
            }
        }
    }

    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertTokenPosition waveBeginPosition = gmm1Position;
        ExpertTokenPosition waveEndPosition = gmm1Position;
        uint32_t currentWaveMGroupCount = 0U;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
            }
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 逐专家准备下一 Wave；AIC/AIV0 同时推进当前 Wave 的 GMM1/Activation。
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    uint32_t expertTokenCount =
                        dispatchPosition.tokenIndexInExpert == 0U ?
                            DispatchMoeExpert(dispatchPosition.expertIdx) :
                            GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                             countWorkspace_, dispatchPosition.expertIdx);
                    AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, waveMGroupTarget_,
                                                                  nextDispatchWaveMGroupCount, dispatchPosition);
                    nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                uint32_t sliceTokenStartIndexInExpert = gmm1Position.tokenIndexInExpert;
                uint32_t sliceTokenCount = AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(
                    expertTokenCount, waveMGroupTarget_, currentWaveMGroupCount, gmm1Position);
                if (sliceTokenCount != 0U) {
                    RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, gmm1RuntimeState, sliceTokenStartIndexInExpert,
                                               sliceTokenCount, gmm1TilesPerMGroup);
                }
                waveEndPosition = gmm1Position;
                currentWaveNeedsGmm1 = IsPositionWithinWave(gmm1Position, currentWaveMGroupCount);
            }
        }

        // Wave 在专家内结束时需包含该专家；在专家边界结束时，waveEndPosition 已指向下一专家。
        uint32_t waveGmm2ExpertEndExclusive =
            waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
        for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
            uint32_t sliceTokenStartIndexInExpert =
                expertIdx == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
            if (sliceTokenStartIndexInExpert == 0U) {
                PrepareGmmExpertState<false>(gmm2State, expertIdx);
            }
            uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
            uint32_t sliceTokenEndIndexInExpert =
                expertIdx == waveEndPosition.expertIdx ? waveEndPosition.tokenIndexInExpert : expertTokenCount;
            uint32_t sliceTokenCount = sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
            if (sliceTokenCount != 0U) {
                RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx, gmTileSequence,
                                        sliceTokenStartIndexInExpert, sliceTokenCount);
            }
        }
    }

    if constexpr (!TopkWeightsPrefetch) {
        EndSync(gmm1RuntimeState.vecSetSyncCom);
    }
}

template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    ProcessInputPreparationStage();
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();

    int32_t gmTileSequence = 0;
    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm1(gmTileSequence);
    }

    CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
    ProcessMoeExpertStages(gmTileSequence);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && countWorkspace_.blockIdx == 0U) {
            ExportExpertTokenCounts<ActivationType, true>(commonConfig_, tokenDispatchScratch_, params_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm2(gmTileSequence);
    }
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
        MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }

    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl

#undef TemplateMegaMoeA4W4WaveTypeClass
#undef TemplateMegaMoeA4W4WaveTypeFunc

#endif
