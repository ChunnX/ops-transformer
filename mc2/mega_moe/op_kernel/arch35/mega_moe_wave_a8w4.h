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
 * \brief MegaMoe A8W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A8W4_H
#define MEGA_MOE_WAVE_A8W4_H

#include "common/mega_moe_utils.h"
#include "mega_moe_arch35.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA8W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaMoeA8W4WaveTypeClass>
class MegaMoeA8W4Wave : public MegaMoe<TemplateMegaMoeA8W4WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA8W4WaveTypeFunc>;
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
    using MegaMoeBase::DispatchMoeExpert;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineJob_;
    using MegaMoeBase::waveCombineScratch_;
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
    using MegaMoeBase::trace_;
    __aicore__ inline uint32_t TraceDispatchMoeExpert(uint32_t expertIdx, uint32_t waveIndex)
    {
        A8W4TraceScope scope{waveIndex, expertIdx, A8W4_TRACE_NO_SCOPE, 0U, 0U, 0U};
        uint64_t begin = trace_.Now();
        uint32_t count = DispatchMoeExpert(expertIdx);
        scope.tokenCount = count;
        scope.workCount = GetMGroupCountForRows(count, GMM1_TILE_M);
        trace_.Record(A8W4TraceEvent::DISPATCH_WORK, A8W4TraceEvent::INVALID, A8W4_TRACE_END_COMPLETE, scope, begin,
                      trace_.Now());
        return count;
    }
#endif
    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < mGroupsPerWave_;
    }

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      uint32_t &startBlockIdx, int32_t &gmm1TileReadySequence,
                                                      uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount,
                                                      uint32_t gmm1TilesPerMGroup, uint32_t waveIndex);
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
                                                   uint32_t sliceTokenCount,
                                                   WaveCombineBufferConfig &combineBufferConfig,
                                                   uint32_t &combineRowSequence, bool useAllAivCores,
                                                   uint32_t waveIndex);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmm1TileReadySequence);
};

// 更新当前专家切片的输入、输出及同步地址，并执行 GMM1 和 Activation。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmm1TileReadySequence,
    uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup, uint32_t waveIndex)
{
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm1TilesPerMGroup;
    if (HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        return;
    }
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
    gmmAddrInfo.criticalPathTrace = &trace_;
    gmmAddrInfo.traceWave = waveIndex;
    gmmAddrInfo.traceExpert = state.expertIdx;
    gmmAddrInfo.traceTile = tokenStartIndexInExpert / GMM1_TILE_M;
    gmmAddrInfo.traceTokenStart = tokenStartIndexInExpert;
    gmmAddrInfo.traceTokenCount = sliceTokenCount;
    gmmAddrInfo.traceWorkCount = problemTileCount;
#endif
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                EPILOGUE_TILE_M, TopkWeightsPrefetch, false, true>(
        epilogueOp_, params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmm1TileReadySequence,
        gmmExecutionConfig_.blockJob, static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert,
        state.expertIdx);
}

// 对已完成 Wave 中的一个专家执行 GMM2 及后续 Combine 调度。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, WaveCombineBufferConfig &combineBufferConfig, uint32_t &combineRowSequence,
    bool useAllAivCores, uint32_t waveIndex)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    uint32_t gmm2NTileCount = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm2NTileCount;
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
    A8W4TraceScope traceScope{
        waveIndex,       state.expertIdx, tokenStartIndexInExpert / GMM1_TILE_M, tokenStartIndexInExpert,
        sliceTokenCount, problemTileCount};
    gmmAddrInfo.criticalPathTrace = &trace_;
    gmmAddrInfo.traceWave = waveIndex;
    gmmAddrInfo.traceExpert = state.expertIdx;
    gmmAddrInfo.traceTile = tokenStartIndexInExpert / GMM1_TILE_M;
    gmmAddrInfo.traceTokenStart = tokenStartIndexInExpert;
    gmmAddrInfo.traceTokenCount = sliceTokenCount;
    gmmAddrInfo.traceWorkCount = problemTileCount;
#endif
    if (!HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType, true, false>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo,
            state, tokenStartIndexInExpert);
        ProblemShape sliceProblemShape = state.problemShape;
        Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    TopkWeightsPrefetch, false, false, true>(sliceProblemShape, gmmAddrInfo, startBlockIdx,
                                                             gmmExecutionConfig_.blockJob, expertTokenCount,
                                                             tokenStartIndexInExpert);
    }
    if (tokenStartIndexInExpert + sliceTokenCount < expertTokenCount) {
        return;
    }
    NotifyWaveGmm2Ready(waveCombineJob_, params_, state.expertIdx);
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
    uint64_t combineBegin = trace_.Now();
#endif
    UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, gmmAddrInfo, state);
    if (useAllAivCores) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                /*
                 * AIV0 的 W4 prologue 使用 MTE3 将 [64 KiB, 192 KiB) 的反量化权重从 UB 搬到 L1，
                 * Final Combine 随后使用 MTE2 覆盖同一 UB 区域。补齐 MTE3 -> MTE2 依赖即可安全复用，
                 * 无需排空其他流水，也无需等待其余 AIV。
                 */
                SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
            }
        }
        combineBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, combineBufferConfig, waveCombineScratch_);
        RunWaveExpertCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                          waveCombineScratch_, params_, gmmAddrInfo, state,
                                                          state.expertIdx, combineRowSequence);
    } else {
        RunWaveExpertCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                    waveCombineScratch_, params_, gmmAddrInfo, state, state.expertIdx,
                                                    combineRowSequence);
    }
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
    if constexpr (g_coreType == AIV) {
        A8W4TraceEvent combineEvent =
            useAllAivCores ? A8W4TraceEvent::FINAL_COMBINE_OR_DRAIN : A8W4TraceEvent::COMBINE_WORK;
        trace_.Record(combineEvent, A8W4TraceEvent::INVALID, A8W4_TRACE_END_SUBMIT, traceScope, combineBegin,
                      trace_.Now());
    }
#endif
}

/*
 * 按 GMM1 调度负载将专家划分为动态 Wave。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段由 AIV1
 * 交替执行下一 Wave 的一个专家 Dispatch 和当前 Wave 的一个专家 Activation，同时 AIC/AIV0 执行当前
 * Wave 的 GMM1。当前 Wave 完成后立即启动 GMM2/Combine，无需等待所有专家的 GMM1。
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages(
    int32_t &gmm1TileReadySequence)
{
    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    DispatchBuffInit();
    PrepareMoeExpertTokenCountTable<true>(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
    WaveCombineBufferConfig combineBufferConfig =
        InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
    uint32_t combineRowSequence = 0U;

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // A8W4 GMM1 与 GMM2 按相同的跨 Wave tile 顺序执行，因此共用分核起点。
    uint32_t startBlockIdx = 0U;

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));

    ExpertTokenPosition dispatchPosition{};
    uint32_t waveIndex = 0U;

    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            uint32_t dispatchedWaveMGroupCount = 0U;
            while (IsPositionWithinWave(dispatchPosition, dispatchedWaveMGroupCount)) {
                uint32_t expertTokenCount = 0U;
                if (dispatchPosition.tokenIndexInExpert == 0U) {
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
                    expertTokenCount = TraceDispatchMoeExpert(dispatchPosition.expertIdx, 0U);
#else
                    expertTokenCount = DispatchMoeExpert(dispatchPosition.expertIdx);
#endif
                } else {
                    expertTokenCount =
                        GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                                                         commonConfig_.moeExpertPerRank, dispatchPosition.expertIdx);
                }
                AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, mGroupsPerWave_,
                                                              dispatchedWaveMGroupCount, dispatchPosition);
            }
        }
    }

    ExpertTokenPosition gmm1Position{};
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

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 交替执行下一 Wave 的 Dispatch 与当前 Wave 的 Activation，避免集中突发 Dispatch。
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    uint32_t expertTokenCount = 0U;
                    if (dispatchPosition.tokenIndexInExpert == 0U) {
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
                        expertTokenCount = TraceDispatchMoeExpert(dispatchPosition.expertIdx, waveIndex + 1U);
#else
                        expertTokenCount = DispatchMoeExpert(dispatchPosition.expertIdx);
#endif
                    } else {
                        expertTokenCount = GetExpertTokenCountFromWorkspace(
                            params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                            commonConfig_.moeExpertPerRank, dispatchPosition.expertIdx);
                    }
                    AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, mGroupsPerWave_,
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
                    RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, startBlockIdx, gmm1TileReadySequence,
                                               sliceTokenStartIndexInExpert, sliceTokenCount, gmm1TilesPerMGroup,
                                               waveIndex);
                }
                waveEndPosition = gmm1Position;
                currentWaveNeedsGmm1 = IsPositionWithinWave(gmm1Position, currentWaveMGroupCount);
            }
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        // 当前 Wave 完成后立即消费，使其 GMM2/Combine 与下一 Wave 的输入预取重叠。
        // Wave 在专家内结束时需包含该专家；在专家边界结束时，waveEndPosition 已指向下一专家。
        uint32_t waveGmm2ExpertEndExclusive =
            waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
        for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
            uint32_t sliceTokenStartIndexInExpert =
                expertIdx == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
            // Wave 从专家起点开始时，准备该专家的 GMM2 状态。
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
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
        uint64_t drainBegin = trace_.Now();
#endif
        DrainCombineRowBuffers(combineRowSequence, combineBufferConfig.rowBufferCount);
#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
        if constexpr (g_coreType == AIV) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
            A8W4TraceScope drainScope{waveIndex, waveLastActiveExpertIdx, A8W4_TRACE_NO_SCOPE, 0U, 0U, 0U};
            trace_.Record(A8W4TraceEvent::FINAL_COMBINE_OR_DRAIN, A8W4TraceEvent::INVALID,
                          A8W4_TRACE_END_COMPLETE | A8W4_TRACE_ONLY_COMPLETION, drainScope, drainBegin, trace_.Now());
        }
#endif
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
        ++waveIndex;
    }
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

} // namespace MegaMoeImpl

#undef TemplateMegaMoeA8W4WaveTypeClass
#undef TemplateMegaMoeA8W4WaveTypeFunc

#endif
