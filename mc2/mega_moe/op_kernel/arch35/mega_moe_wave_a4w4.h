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
#include "mega_moe_trace.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA4W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA4W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaMoeA4W4WaveTypeClass>
class MegaMoeA4W4Wave : public MegaMoe<TemplateMegaMoeA4W4WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA4W4WaveTypeFunc>;
    friend MegaMoeBase;

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

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineJob_;
    using MegaMoeBase::waveCombineScratch_;
    using MegaMoeBase::syncWorkspaceLayout_;

#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
    A4W4WaveTrace trace_{};
#endif

    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < mGroupsPerWave_;
    }

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      GmmRuntimeState &runtimeState, uint32_t tokenStartIndexInExpert,
                                                      uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup,
                                                      uint32_t waveIndex);
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
                                                   uint32_t sliceTokenCount,
                                                   WaveCombineBufferConfig &combineBufferConfig,
                                                   uint32_t &combineRowSequence, bool useAllAivCores,
                                                   uint32_t waveIndex);
    __aicore__ inline void ProcessMoeExpertStages();
};

// A4W4 的 GMM1 使用 generic kernel，Activation 将中间结果提升为 FP8 后供 GMM2 使用。
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, GmmRuntimeState &runtimeState, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup, uint32_t waveIndex)
{
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm1TilesPerMGroup;
    if (SkipUnownedWaveAicProblem(problemTileCount, gmmExecutionConfig_.blockJob, runtimeState.startBlockIdx)) {
        return;
    }
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, false, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    bool coversFullExpert =
        tokenStartIndexInExpert == 0U && sliceTokenCount == static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
    A4W4WaveTraceEvent stageEvent = A4W4WaveTraceEvent::INVALID;
    A4W4WaveTraceEvent waitEvent = A4W4WaveTraceEvent::INVALID;
    if constexpr (g_coreType == AIC) {
        stageEvent = A4W4WaveTraceEvent::GMM1;
        waitEvent = A4W4WaveTraceEvent::WAIT_FOR_DISPATCH;
    } else if (GetSubBlockIdx() == 0U) {
        stageEvent = A4W4WaveTraceEvent::ACTIVATION;
        waitEvent = A4W4WaveTraceEvent::WAIT_FOR_GMM1;
    }
    CriticalPathTileTraceContext traceContext =
        trace_.CreateTileTraceContext(stageEvent, waitEvent, waveIndex, state.expertIdx, tokenStartIndexInExpert);
    CriticalPathTileTraceContext *traceContextPtr = stageEvent == A4W4WaveTraceEvent::INVALID ? nullptr : &traceContext;
#else
    CriticalPathTileTraceContext *traceContextPtr = nullptr;
#endif
    RunGmm1GenericByWeightFormat<QuantOutType, ActivationQuantOutType, QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                 TopkWeightsPrefetch, false, true, false, true>(
        gmmExecutionConfig_, params_, epilogueOp_, gmmAddrInfo, sliceProblemShape,
        static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert, runtimeState, state.expertIdx,
        nullptr, coversFullExpert, traceContextPtr);
}

// A4W4 的 GMM2 复用 A8W4 prologue，将 FP4 weight 解包为 FP8 后执行矩阵乘与 Combine。
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, WaveCombineBufferConfig &combineBufferConfig, uint32_t &combineRowSequence,
    bool useAllAivCores, uint32_t waveIndex)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    uint32_t gmm2NTileCount = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm2NTileCount;
#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
    CriticalPathTileTraceContext gmm2TraceContext{};
    CriticalPathTileTraceContext combineTraceContext{};
    CriticalPathTileTraceContext combineMetaTraceContext{};
    CriticalPathTileTraceContext *gmm2TraceContextPtr = nullptr;
    CriticalPathTileTraceContext *combineTraceContextPtr = nullptr;
    CriticalPathTileTraceContext *combineMetaTraceContextPtr = nullptr;
    if constexpr (g_coreType == AIC) {
        gmm2TraceContext =
            trace_.CreateTileTraceContext(A4W4WaveTraceEvent::GMM2, A4W4WaveTraceEvent::WAIT_FOR_ACTIVATION, waveIndex,
                                          state.expertIdx, tokenStartIndexInExpert);
        gmm2TraceContextPtr = &gmm2TraceContext;
    } else {
        if (GetSubBlockIdx() == 0U) {
            gmm2TraceContext =
                trace_.CreateTileTraceContext(A4W4WaveTraceEvent::GMM2_PROLOGUE, A4W4WaveTraceEvent::INVALID, waveIndex,
                                              state.expertIdx, tokenStartIndexInExpert);
            gmm2TraceContextPtr = &gmm2TraceContext;
        }
        if (GetSubBlockIdx() == 1U || useAllAivCores) {
            // Combine 在专家完整输出上重新分核，其 owned range 已经是专家内坐标。
            combineTraceContext = trace_.CreateTileTraceContext(
                A4W4WaveTraceEvent::COMBINE, A4W4WaveTraceEvent::WAIT_FOR_GMM2, waveIndex, state.expertIdx, 0U);
            combineTraceContextPtr = &combineTraceContext;
            combineMetaTraceContext = trace_.CreateTileTraceContext(
                A4W4WaveTraceEvent::COMBINE_META_PREFETCH, A4W4WaveTraceEvent::INVALID, waveIndex, state.expertIdx, 0U);
            combineMetaTraceContextPtr = &combineMetaTraceContext;
        }
    }
#else
    CriticalPathTileTraceContext *gmm2TraceContextPtr = nullptr;
    CriticalPathTileTraceContext *combineTraceContextPtr = nullptr;
    CriticalPathTileTraceContext *combineMetaTraceContextPtr = nullptr;
#endif
    bool ownsGmm2Tile = !SkipUnownedWaveAicProblem(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx);
    if (ownsGmm2Tile) {
        UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType, true, false>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo,
            state, tokenStartIndexInExpert);
        ProblemShape sliceProblemShape = state.problemShape;
        Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    TopkWeightsPrefetch, false, false, true>(
            sliceProblemShape, gmmAddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, expertTokenCount,
            tokenStartIndexInExpert, tokenStartIndexInExpert == 0U && sliceTokenCount == expertTokenCount,
            gmm2TraceContextPtr);
    }
    if (tokenStartIndexInExpert + sliceTokenCount < expertTokenCount) {
        return;
    }
    NotifyWaveGmm2Ready(waveCombineJob_, params_, state.expertIdx);
    UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, gmmAddrInfo, state);
    if (useAllAivCores) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                /* A4W4 复用同一 W4 prologue UB；Final Combine 覆盖前等待该 MTE3 搬运完成。 */
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        combineBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, combineBufferConfig, waveCombineScratch_);
        RunWaveExpertCombineStage<CombineQuantMode, true>(
            commonConfig_, waveCombineJob_, combineBufferConfig, waveCombineScratch_, params_, gmmAddrInfo, state,
            state.expertIdx, combineRowSequence, combineTraceContextPtr, combineMetaTraceContextPtr);
    } else {
        RunWaveExpertCombineStage<CombineQuantMode>(
            commonConfig_, waveCombineJob_, combineBufferConfig, waveCombineScratch_, params_, gmmAddrInfo, state,
            state.expertIdx, combineRowSequence, combineTraceContextPtr, combineMetaTraceContextPtr);
    }
}

/*
 * 按专家顺序将 token 划分为连续 Wave。Wave 以 256 行 M group 为单位，根据 GMM1 的 N tile 数
 * 控制总负载；空间不足时可在专家内部切分，但同一 Wave 的 GMM1 和 GMM2 始终处理相同的 token 范围。
 *
 * 启动阶段先 Dispatch 第一个 Wave；稳态阶段由 AIV1 预取下一 Wave，同时 AIC/AIV0 独立推进当前
 * Wave 的 GMM1/Activation/GMM2。AIV1 完成预取后再处理当前 Wave 的 Combine。
 */
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::ProcessMoeExpertStages()
{
    DispatchBuffInit();
    PrepareMoeExpertTokenCountTable(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
    WaveCombineBufferConfig combineBufferConfig =
        InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
    uint32_t combineRowSequence = 0U;

    ExpertTokenPosition dispatchPosition{};
    uint32_t dispatchExpertTokenCount = 0U;

    // GMM1 开始消费输入前，由 AIV1 先准备第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            uint32_t dispatchedWaveMGroupCount = 0U;
            while (IsPositionWithinWave(dispatchPosition, dispatchedWaveMGroupCount)) {
                if (dispatchPosition.tokenIndexInExpert == 0U) {
#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
                    CriticalPathTileTraceContext dispatchTraceContext = trace_.CreateTileTraceContext(
                        A4W4WaveTraceEvent::DISPATCH, A4W4WaveTraceEvent::INVALID, 0U, dispatchPosition.expertIdx, 0U);
                    CriticalPathTileTraceContext *dispatchTraceContextPtr = &dispatchTraceContext;
#else
                    CriticalPathTileTraceContext *dispatchTraceContextPtr = nullptr;
#endif
                    dispatchExpertTokenCount =
                        RunWaveExpertDispatchStage<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
                            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_,
                            params_, winRankAddr_, tokenDispatchScratch_, dispatchPosition.expertIdx,
                            dispatchTraceContextPtr);
                }
                AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(dispatchExpertTokenCount, mGroupsPerWave_,
                                                              dispatchedWaveMGroupCount, dispatchPosition);
            }
        }
    }

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // 同一 Block 内绑定的 1C2V 各自持有分核游标，必须按相同调用顺序推进并始终保持一致。
    // 某个角色即使不参与当前 GMM 计算，也必须使用该 problem 的 tile 数更新游标。
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    GmmRuntimeState gmm1RuntimeState{startBlockIdx, vecSetSyncCom, gmm1PingPongIdx};

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));

    ExpertTokenPosition gmm1Position{};
    uint32_t waveIndex = 0U;
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertTokenPosition waveBeginPosition = gmm1Position;
        ExpertTokenPosition waveEndPosition = gmm1Position;
        uint32_t currentWaveMGroupCount = 0U;
        uint32_t waveLastActiveExpertIdx = commonConfig_.moeExpertPerRank;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
            }
        }

        // AIV1 先预取下一 Wave 再进入当前 Combine；AIC/AIV0 不等待预取，独立推进当前 Wave。
        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    if (dispatchPosition.tokenIndexInExpert == 0U) {
#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
                        CriticalPathTileTraceContext dispatchTraceContext =
                            trace_.CreateTileTraceContext(A4W4WaveTraceEvent::DISPATCH, A4W4WaveTraceEvent::INVALID,
                                                          waveIndex + 1U, dispatchPosition.expertIdx, 0U);
                        CriticalPathTileTraceContext *dispatchTraceContextPtr = &dispatchTraceContext;
#else
                        CriticalPathTileTraceContext *dispatchTraceContextPtr = nullptr;
#endif
                        dispatchExpertTokenCount = RunWaveExpertDispatchStage<ActivationType, QuantScaleOutType,
                                                                              GMM1_TILE_M, TopkWeightsPrefetch>(
                            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_,
                            params_, winRankAddr_, tokenDispatchScratch_, dispatchPosition.expertIdx,
                            dispatchTraceContextPtr);
                    }
                    AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(dispatchExpertTokenCount, mGroupsPerWave_,
                                                                  nextDispatchWaveMGroupCount, dispatchPosition);
                    nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    this->template PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                uint32_t sliceTokenStartIndexInExpert = gmm1Position.tokenIndexInExpert;
                uint32_t sliceTokenCount = AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(
                    expertTokenCount, mGroupsPerWave_, currentWaveMGroupCount, gmm1Position);
                if (sliceTokenCount != 0U) {
                    waveLastActiveExpertIdx = gmm1State.expertIdx;
                    RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, gmm1RuntimeState, sliceTokenStartIndexInExpert,
                                               sliceTokenCount, gmm1TilesPerMGroup, waveIndex);
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
                this->template PrepareGmmExpertState<false>(gmm2State, expertIdx);
            }
            uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
            uint32_t sliceTokenEndIndexInExpert =
                expertIdx == waveEndPosition.expertIdx ? waveEndPosition.tokenIndexInExpert : expertTokenCount;
            uint32_t sliceTokenCount = sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
            if (sliceTokenCount != 0U) {
                bool useAllAivCores = waveEndPosition.expertIdx >= commonConfig_.moeExpertPerRank &&
                                      expertIdx == waveLastActiveExpertIdx &&
                                      sliceTokenEndIndexInExpert >= expertTokenCount;
                RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx, sliceTokenStartIndexInExpert,
                                        sliceTokenCount, combineBufferConfig, combineRowSequence, useAllAivCores,
                                        waveIndex);
            }
        }
        DrainWaveCombineRowRing(combineRowSequence, combineBufferConfig.rowBufferCount);
        ++waveIndex;
    }

    if constexpr (!TopkWeightsPrefetch) {
        EndSync(gmm1RuntimeState.vecSetSyncCom);
    }
}

template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::Process()
{
#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
    A4W4WaveTraceRole traceRole = A4W4WaveTraceRole::AIC;
    if constexpr (g_coreType == AIV) {
        traceRole = GetSubBlockIdx() == 0U ? A4W4WaveTraceRole::AIV0 : A4W4WaveTraceRole::AIV1;
    }
    trace_.Init(params_.workspaceInfo.dispatchRevDataPtr, params_.workspaceInfo.workspaceSize, params_.y2GmAddr,
                A4W4_WAVE_TRACE_OUTPUT_BYTES, gmmExecutionConfig_.blockJob.totalJobs, traceRole,
                gmmExecutionConfig_.blockJob.jobIndex, commonConfig_.tokenNum, commonConfig_.tokenHiddenDim,
                commonConfig_.gmm1OutputDim, commonConfig_.topK, commonConfig_.moeExpertPerRank,
                commonConfig_.worldSize);
    trace_.SetWaveMGroupTarget(mGroupsPerWave_);
#endif
    this->ProcessWave(*this);
#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0
    trace_.Finish();
#endif
}

} // namespace MegaMoeImpl

#undef TemplateMegaMoeA4W4WaveTypeClass
#undef TemplateMegaMoeA4W4WaveTypeFunc

#endif
