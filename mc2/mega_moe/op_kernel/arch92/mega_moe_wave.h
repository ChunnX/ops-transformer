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
 * \file mega_moe_wave.h
 * \brief MegaMoe arch92 1C1V wave 流水
 */

#ifndef MEGA_MOE_ARCH92_WAVE_H
#define MEGA_MOE_ARCH92_WAVE_H

#include <type_traits>

#include "../arch35_92/common/mega_moe_utils.h"
#include "../arch35_92/mega_moe.h"

namespace MegaMoeImpl {

constexpr uint32_t GMM2_LAG_MIN_TOKEN_NUM = 4096U;

#define TemplateMegaMoeArch92WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch, \
        typename TopkIndexType
#define TemplateMegaMoeArch92WaveTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format, CombineQuantMode, \
        TopkWeightsPrefetch, TopkIndexType

/*
 * Init、输入准备、共享专家和 Unpermute 由 MegaMoe 基类统一实现；本类保留 arch92 的
 * 双 Wave lookahead、GMM1/GMM2 交错调度及量化 Combine 编排。
 */
template <TemplateMegaMoeArch92WaveTypeClass>
class MegaMoeArch92Wave
    : public MegaMoe<TemplateMegaMoeArch92WaveTypeFunc, MegaMoeArch92ExecutionPolicy> {
private:
    using ExecutionPolicy = MegaMoeArch92ExecutionPolicy;
    using MegaMoeBase = MegaMoe<TemplateMegaMoeArch92WaveTypeFunc, ExecutionPolicy>;
    friend MegaMoeBase;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using CombineBufferConfig = WaveCombineBufferConfig;
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaMoeBase::ActivationQuantOutType;
    using SharedQuantConfig = typename MegaMoeBase::SharedQuantConfig;
    using MoeQuantConfig = typename MegaMoeBase::MoeQuantConfig;
    using SharedActivationOutType = typename MegaMoeBase::SharedActivationOutType;
    using SharedQuantScaleType = typename MegaMoeBase::SharedQuantScaleType;

    using Gmm1PersistentConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, QuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                          QuantScaleOutType, MoeWeight1Format != FORMAT_ND, TopkWeightsPrefetch, false, false, true>;
    using Gmm2PersistentConfig =
        GmmKernel::Config<false, CombineQuantMode, ActivationQuantOutType, MoeWeightType, bfloat16_t,
                          QuantScaleOutType, QuantScaleOutType, MoeWeight2Format != FORMAT_ND, TopkWeightsPrefetch,
                          false, false, true>;
    using SharedGmm1PersistentConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, typename SharedQuantConfig::QuantOutType, SharedWeightType,
                          bfloat16_t, SharedQuantScaleType, SharedQuantScaleType,
                          SharedWeight1Format != FORMAT_ND, false, true, false, true>;
    using SharedGmm2PersistentConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, SharedActivationOutType, SharedWeightType, bfloat16_t,
                          SharedQuantScaleType, SharedQuantScaleType, SharedWeight2Format != FORMAT_ND, false, true,
                          false, true>;
    using Gmm1BlockMmad = typename Gmm1PersistentConfig::BlockMmad;
    using Gmm2BlockMmad = typename Gmm2PersistentConfig::BlockMmad;
    using SharedGmm1BlockMmad = typename SharedGmm1PersistentConfig::BlockMmad;
    using SharedGmm2BlockMmad = typename SharedGmm2PersistentConfig::BlockMmad;
    static_assert(MoeQuantConfig::AXW_MODE != AxWMode::A8W4 && SharedQuantConfig::AXW_MODE != AxWMode::A8W4,
                  "arch92 A8W4 requires the direct mixed FP8/FP4 Tensor BlockMmad interface");
    struct EmptyMmadContext {};
    template <typename BlockMmad>
    using MmadContextStorage =
        std::conditional_t<g_coreType == AIC, GmmKernel::BlockMmadContext<BlockMmad>, EmptyMmadContext>;
    static constexpr bool ROUTED_AND_SHARED_USE_SAME_MMAD =
        std::is_same_v<Gmm1BlockMmad, SharedGmm1BlockMmad>;
    static constexpr bool SHARED_GMM_STAGES_USE_SAME_MMAD =
        std::is_same_v<SharedGmm1BlockMmad, SharedGmm2BlockMmad>;
    using SharedMmadContextStorage =
        std::conditional_t<g_coreType == AIC && SHARED_GMM_STAGES_USE_SAME_MMAD &&
                               !ROUTED_AND_SHARED_USE_SAME_MMAD,
                           GmmKernel::BlockMmadContext<SharedGmm1BlockMmad>, EmptyMmadContext>;

    __aicore__ inline void *GetRoutedMmadContext()
    {
        if constexpr (g_coreType == AIC) {
            return static_cast<void *>(&routedMmadContext_);
        }
        return nullptr;
    }

    __aicore__ inline void *GetRoutedGmm2MmadContext()
    {
        if constexpr (std::is_same_v<Gmm1BlockMmad, Gmm2BlockMmad>) {
            return GetRoutedMmadContext();
        }
        return nullptr;
    }

    __aicore__ inline void *GetSharedMmadContext()
    {
        if constexpr (g_coreType == AIC && SHARED_GMM_STAGES_USE_SAME_MMAD) {
            if constexpr (ROUTED_AND_SHARED_USE_SAME_MMAD) {
                return static_cast<void *>(&routedMmadContext_);
            }
            return static_cast<void *>(&sharedMmadContext_);
        }
        return nullptr;
    }

    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::EnterSteadyDispatch;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmm1PingPongIdx_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::moeExpertPerRank_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::startBlockIdx_;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineJob_;
    using MegaMoeBase::waveCombineScratch_;
    using MegaMoeBase::waveCombineDedupScratch_;

    struct DispatchWaveState {
        ExpertTokenPosition position{};
        ExpertTokenPosition preparedEnd{};
        bool hasPreparedWave = false;
    };

    struct Gmm2WaveState {
        ExpertTokenPosition position{};
        ExpertLoopState expertState{};
        GMMAddrInfo addrInfo{};
        ExpertLoopState combineExpertState{};
        GMMAddrInfo combineAddrInfo{};
        uint32_t combineBeginExpertIndex = 0U;
        uint32_t finalCombineExpertIndex = 0U;
        uint64_t loopCount = 0U;
    };

    __aicore__ inline uint32_t FindFinalCombineExpert();
    __aicore__ inline void PrepareDispatchWave(DispatchWaveState &state, bool isFirstWave);
    __aicore__ inline void ProcessGmm2AndCombineWave(const ExpertTokenPosition &waveEndPosition, Gmm2WaveState &state,
                                                     const CombineBufferConfig &combineBufferConfig);
    /*
     * 唯一 AIV 按阶段复用 UB；Dispatch cumsum 经 GM 备份恢复，不是全流程常驻 UB。
     * 量化 Combine 沿用公共布局：行环与量化 scratch 位于 [64, 184 KiB)，ready 扫描从
     * 184 KiB 开始，metadata 从 200 KiB 开始。稳态使用一个行槽，最后一个非空专家可扩容
     * 至自适应的最多六槽，但必须先排空旧行环。这些地址不代表 A6 的 UB 总容量。
     */
    __aicore__ inline CombineBufferConfig InitCombineBuffers();
    __aicore__ inline void ProcessMoeExpertStages(Gmm1ActivationSync &gmm1ActivationSync,
                                                  Gmm2CombineSync &gmm2CombineSync);
    __aicore__ inline bool IsSameExpertTokenPosition(const ExpertTokenPosition &currentPosition,
                                                     const ExpertTokenPosition &targetPosition) const;
    __aicore__ inline ExpertTokenPosition DispatchNextWave(ExpertTokenPosition &dispatchPosition,
                                                           bool reloadCumsum);
    __aicore__ inline ExpertTokenPosition ProcessGmm1Wave(ExpertTokenPosition &gmm1Position,
                                                          ExpertLoopState &gmm1ExpertState, GMMAddrInfo &gmm1AddrInfo,
                                                          GmmRuntimeState &runtimeState);
    __aicore__ inline void ProcessGmm2Wave(ExpertTokenPosition &gmm2Position,
                                           const ExpertTokenPosition &waveEndPosition, ExpertLoopState &gmm2ExpertState,
                                           GMMAddrInfo &gmm2AddrInfo, uint32_t &startBlockIdx);
    __aicore__ inline void ProcessGmm2WaveProblem(const ExpertTokenPosition &gmm2Position,
                                                  uint32_t waveEndTokenIndexInExpert,
                                                  const ExpertLoopState &gmm2ExpertState, GMMAddrInfo &gmm2AddrInfo,
                                                  uint32_t &startBlockIdx);
    __aicore__ inline void AdvanceGmm2WavePosition(ExpertTokenPosition &gmm2Position,
                                                   uint32_t waveEndTokenIndexInExpert,
                                                   const ExpertLoopState &gmm2ExpertState);
    __aicore__ inline void ProcessCombineExperts(uint32_t expertBegin, uint32_t expertEnd,
                                                 ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
                                                 const CombineBufferConfig &bufferConfig,
                                                 uint32_t finalCombineExpertIndex);

    uint32_t gmm1TilesPerMGroup_ = 1U;
    uint32_t gmm2TilesPerMGroup_ = 1U;
    MmadContextStorage<Gmm1BlockMmad> routedMmadContext_;
    SharedMmadContextStorage sharedMmadContext_;
};

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline typename MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::CombineBufferConfig
MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::InitCombineBuffers()
{
    return InitWaveCombineBuffers<CombineQuantMode, false, WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT>(commonConfig_,
                                                                                                 waveCombineScratch_);
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline bool MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::IsSameExpertTokenPosition(
    const ExpertTokenPosition &currentPosition, const ExpertTokenPosition &targetPosition) const
{
    return currentPosition.expertIdx == targetPosition.expertIdx &&
           currentPosition.tokenIndexInExpert == targetPosition.tokenIndexInExpert;
}

// 规划并 Dispatch 紧接着的一个完整 WAVE，返回更新后的全局专家位置。
template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::DispatchNextWave(
    ExpertTokenPosition &dispatchPosition, bool reloadCumsum)
{
    ExpertTokenRange dispatchRange{dispatchPosition, dispatchPosition};
    ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
    uint32_t dispatchWaveMGroupCount = 0U;
    while (plannedDispatchPosition.expertIdx < moeExpertPerRank_ && dispatchWaveMGroupCount < mGroupsPerWave_) {
        ExpertTokenRange nextDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
            params_.workspaceInfo.expertRecvTokenCountPtr, countWorkspace_, moeExpertPerRank_, mGroupsPerWave_,
            dispatchWaveMGroupCount, plannedDispatchPosition);
        plannedDispatchPosition = nextDispatchRange.end;
        dispatchRange.end = nextDispatchRange.end;
    }
    if (reloadCumsum && dispatchRange.end.globalTokenIndex > dispatchRange.begin.globalTokenIndex) {
        uint32_t lastExpertIdx = dispatchRange.end.tokenIndexInExpert == 0U ? dispatchRange.end.expertIdx - 1U :
                                                                           dispatchRange.end.expertIdx;
        ReloadDispatchCumsumRange(commonConfig_, tokenDispatchScratch_, dispatchRange.begin.expertIdx, lastExpertIdx);
    }
    DispatchTokenRange<TopkIndexType, ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
        tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
        g_winRankAddr_, tokenDispatchScratch_, dispatchRange);
    // 整 WAVE 的数据和 ready flag 发布完成后，再提交 Dispatch 进度。
    dispatchPosition = plannedDispatchPosition;
    return dispatchPosition;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm1Wave(
    ExpertTokenPosition &gmm1Position, ExpertLoopState &gmm1ExpertState, GMMAddrInfo &gmm1AddrInfo,
    GmmRuntimeState &runtimeState)
{
    uint32_t processedMGroupCount = 0U;
    while (gmm1Position.expertIdx < moeExpertPerRank_ && processedMGroupCount < mGroupsPerWave_) {
        if (gmm1Position.tokenIndexInExpert == 0U) {
            if (!gmm1ExpertState.expertCountTableReady) {
                WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_,
                                                0U);
                gmm1ExpertState.expertCountTableReady = true;
            }
            uint32_t expertTokenCount =
                GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRecvTokenCountPtr, countWorkspace_,
                                                 moeExpertPerRank_, gmm1Position.expertIdx);
            UpdateExpertLoopState(gmm1ExpertState, gmm1Position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(gmm1ExpertState.problemShape);
        if (expertRowCount == 0U || gmm1Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm1Position.expertIdx;
            gmm1Position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t remainingMGroupCount = mGroupsPerWave_ - processedMGroupCount;
        uint32_t waveEndTokenIndexInExpert = GetWaveEndRowOffsetInExpert(
            expertRowCount, gmm1Position.tokenIndexInExpert, remainingMGroupCount, GMM1_TILE_M);
        uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm1Position.tokenIndexInExpert;
        uint32_t problemMGroupCount = GetMGroupCountForRows(waveRowCount, GMM1_TILE_M);
        bool skipGmm1Problem = false;
        if constexpr (g_coreType == AIC) {
            uint32_t problemTileCount = problemMGroupCount * gmm1TilesPerMGroup_;
            skipGmm1Problem = HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob,
                                                           runtimeState.startBlockIdx);
        }
        if (skipGmm1Problem) {
            processedMGroupCount += problemMGroupCount;
            gmm1Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
            gmm1Position.globalTokenIndex += waveRowCount;
            if (gmm1Position.tokenIndexInExpert >= expertRowCount) {
                ++gmm1Position.expertIdx;
                gmm1Position.tokenIndexInExpert = 0U;
            }
            continue;
        }
        ProblemShape gmm1WaveProblemShape = gmm1ExpertState.problemShape;
        Get<M_VALUE>(gmm1WaveProblemShape) = waveEndTokenIndexInExpert - gmm1Position.tokenIndexInExpert;
        UpdateMoeExpertGmm1GlobalBuffer<ActivationType, MoeWeightType, ActivationQuantOutType, QuantScaleOutType,
                                        PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE, false,
                                        TopkWeightsPrefetch, ExecutionPolicy::USE_GM_HANDOFF>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
            gmm1AddrInfo, gmm1ExpertState, gmm1Position.tokenIndexInExpert, gmm1TilesPerMGroup_);
        /*
         * 只有该 problem 从第 0 行覆盖到专家尾，后续 Wave 才不会再次读取同一份 GMM1 权重。
         * 热点专家的任一 Wave problem 都为 false，即使本次不足 256 行也保留正常 L2 cache。
         */
        bool isWholeExpert =
            gmm1Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
        uint32_t waveTokenStartIndex =
            static_cast<uint32_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
        void *blockMmadContext = GetRoutedMmadContext();
        RunGmm1Generic<QuantOutType, ActivationType, MoeWeightType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                       MoeWeight1Format != FORMAT_ND, GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, false, true,
                       ExecutionPolicy::USE_GM_HANDOFF>(
            epilogueOp_, params_, gmm1WaveProblemShape, gmm1AddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, gmmExecutionConfig_.blockJob, waveTokenStartIndex, gmm1Position.expertIdx,
            runtimeState.pingpongIdx, blockMmadContext, isWholeExpert);

        processedMGroupCount += problemMGroupCount;
        gmm1Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
        gmm1Position.globalTokenIndex += waveRowCount;
        if (gmm1Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm1Position.expertIdx;
            gmm1Position.tokenIndexInExpert = 0U;
        }
    }
    return gmm1Position;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm2WaveProblem(
    const ExpertTokenPosition &gmm2Position, uint32_t waveEndTokenIndexInExpert, const ExpertLoopState &gmm2ExpertState,
    GMMAddrInfo &gmm2AddrInfo, uint32_t &startBlockIdx)
{
    uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
    uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
    uint32_t problemMGroupCount = GetMGroupCountForRows(waveRowCount, GMM1_TILE_M);
    bool skipGmm2Problem = false;
    if constexpr (g_coreType == AIC) {
        uint32_t problemTileCount = problemMGroupCount * gmm2TilesPerMGroup_;
        skipGmm2Problem = HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx);
    }
    if (skipGmm2Problem) {
        return;
    }

    ProblemShape gmm2WaveProblemShape = gmm2ExpertState.problemShape;
    Get<M_VALUE>(gmm2WaveProblemShape) = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
    UpdateMoeExpertGmm2GlobalBuffer<MoeWeightType, ActivationQuantOutType, QuantScaleOutType>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmm2AddrInfo,
        gmm2ExpertState, gmm2Position.tokenIndexInExpert);
    // GMM2 与 GMM1 使用相同保护：只有完整专家 problem 才允许进一步判断是否绕过 L2。
    bool isWholeExpert =
        gmm2Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
    void *blockMmadContext = GetRoutedGmm2MmadContext();
    if (IsCombineDedupOn(params_.tilingData->dedupMode)) {
        RunGmm2Generic<COMBINE_NO_QUANT, ActivationQuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                       QuantScaleOutType,
                       MoeWeight2Format != FORMAT_ND, false, GMM1_TILE_M, TopkWeightsPrefetch, false, true,
                       false>(
            gmm2WaveProblemShape, gmm2AddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, blockMmadContext,
            isWholeExpert, gmm2Position.tokenIndexInExpert, &params_);
    } else {
        RunGmm2Generic<COMBINE_NO_QUANT, ActivationQuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                       QuantScaleOutType,
                       MoeWeight2Format != FORMAT_ND, false, GMM1_TILE_M, TopkWeightsPrefetch, false, true,
                       CombineQuantMode == COMBINE_NO_QUANT>(
            gmm2WaveProblemShape, gmm2AddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, blockMmadContext,
            isWholeExpert, gmm2Position.tokenIndexInExpert, &params_);
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::AdvanceGmm2WavePosition(
    ExpertTokenPosition &gmm2Position, uint32_t waveEndTokenIndexInExpert, const ExpertLoopState &gmm2ExpertState)
{
    uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
    uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
    gmm2Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
    gmm2Position.globalTokenIndex += waveRowCount;
    if (gmm2Position.tokenIndexInExpert >= expertRowCount) {
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
        } else if (IsCombineDedupOn(params_.tilingData->dedupMode)) {
            NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
        }
        ++gmm2Position.expertIdx;
        gmm2Position.tokenIndexInExpert = 0U;
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm2Wave(
    ExpertTokenPosition &gmm2Position, const ExpertTokenPosition &waveEndPosition, ExpertLoopState &gmm2ExpertState,
    GMMAddrInfo &gmm2AddrInfo, uint32_t &startBlockIdx)
{
    while (!IsSameExpertTokenPosition(gmm2Position, waveEndPosition) && gmm2Position.expertIdx < moeExpertPerRank_) {
        if (gmm2Position.tokenIndexInExpert == 0U) {
            uint32_t expertTokenCount =
                GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRecvTokenCountPtr, countWorkspace_,
                                                 moeExpertPerRank_, gmm2Position.expertIdx);
            UpdateExpertLoopState(gmm2ExpertState, gmm2Position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
        if (expertRowCount == 0U || gmm2Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm2Position.expertIdx;
            gmm2Position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t waveEndTokenIndexInExpert = static_cast<uint32_t>(expertRowCount);
        if (gmm2Position.expertIdx == waveEndPosition.expertIdx) {
            waveEndTokenIndexInExpert = waveEndPosition.tokenIndexInExpert;
        }
        ProcessGmm2WaveProblem(gmm2Position, waveEndTokenIndexInExpert, gmm2ExpertState, gmm2AddrInfo, startBlockIdx);
        AdvanceGmm2WavePosition(gmm2Position, waveEndTokenIndexInExpert, gmm2ExpertState);
    }
}

// arch92 只有一个 AIV 角色：同一 AIV 在 GMM2 epilogue 后继续执行当前 WAVE 的 Combine。
template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessCombineExperts(
    uint32_t expertBegin, uint32_t expertEnd, ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
    const CombineBufferConfig &bufferConfig, uint32_t finalCombineExpertIndex)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    CombineBufferConfig activeBufferConfig = bufferConfig;
    uint32_t rowSequence = 0U;
    const bool dedupCombine = IsCombineDedupOn(params_.tilingData->dedupMode);
    if (dedupCombine) {
        ArmWaveCombineDedupEvents();
    }
    for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRecvTokenCountPtr,
                                                                     countWorkspace_, moeExpertPerRank_, expertIdx);
        UpdateExpertLoopState(combineState, expertIdx, expertTokenCount);
        if (expertTokenCount == 0U) {
            continue;
        }
        UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, combineAddrInfo, combineState);
        if (dedupCombine) {
            RunWaveCombineStageDedup(commonConfig_, waveCombineJob_, activeBufferConfig, waveCombineScratch_,
                                     waveCombineDedupScratch_, params_, combineState, expertIdx, rowSequence);
            continue;
        }
        if (expertIdx == finalCombineExpertIndex) {
            // 扩容前按旧槽数消费全部在途事件，并将 rowSequence 归零后重新开始填充。
            DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
            activeBufferConfig =
                InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
        }
        RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, activeBufferConfig,
                                              waveCombineScratch_, params_, combineAddrInfo, combineState,
                                              combineState.expertIdx, rowSequence);
    }
    if (dedupCombine) {
        DrainCombineRowBuffersDedup(rowSequence, activeBufferConfig.rowBufferCount);
    } else {
        DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline uint32_t MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::FindFinalCombineExpert()
{
    if constexpr (g_coreType == AIV) {
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            for (uint32_t expertEnd = moeExpertPerRank_; expertEnd > 0U; --expertEnd) {
                uint32_t expertIdx = expertEnd - 1U;
                uint64_t countOffset =
                    GetExpertCountWorkspaceOffset(countWorkspace_, moeExpertPerRank_, expertIdx, true);
                __gm__ int32_t *expertTokenCountAddr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRecvTokenCountPtr) + countOffset;
                if (AscendC::ReadGmBypassDCache(expertTokenCountAddr) != 0) {
                    return expertIdx;
                }
            }
        }
    }

    return moeExpertPerRank_;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::PrepareDispatchWave(
    DispatchWaveState &state, bool isFirstWave)
{
    ExpertTokenPosition waveEndPosition =
        state.hasPreparedWave ? state.preparedEnd : DispatchNextWave(state.position, false);
    if (waveEndPosition.expertIdx < moeExpertPerRank_) {
        state.preparedEnd = DispatchNextWave(state.position, !isFirstWave);
        state.hasPreparedWave = true;
    } else {
        state.hasPreparedWave = false;
    }
    if (isFirstWave) {
        EnterSteadyDispatch();
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm2AndCombineWave(
    const ExpertTokenPosition &waveEndPosition, Gmm2WaveState &state, const CombineBufferConfig &combineBufferConfig)
{
    ProcessGmm2Wave(state.position, waveEndPosition, state.expertState, state.addrInfo, startBlockIdx_);
    UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++state.loopCount);
    if (CombineQuantMode != COMBINE_NO_QUANT || IsCombineDedupOn(params_.tilingData->dedupMode)) {
        ProcessCombineExperts(state.combineBeginExpertIndex, waveEndPosition.expertIdx, state.combineExpertState,
                              state.combineAddrInfo, combineBufferConfig, state.finalCombineExpertIndex);
        state.combineBeginExpertIndex = waveEndPosition.expertIdx;
    }
}

/*
 * 按动态 WAVE 边界滚动执行 Arch92 MoE 流水。唯一 AIV 启动时连续准备 W0/W1，稳态消费已准备 WAVE 的同时
 * Dispatch 下一 WAVE，始终保持一轮 lookahead。每次先规划完整 WAVE 的 [begin, end) 范围，再通过统一的
 * DispatchTokenRange 执行搬运和 ready 发布；GMM1、Activation、GMM2 与 Combine 复用同一 WAVE 终点。
 */
template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessMoeExpertStages(
    Gmm1ActivationSync &gmm1ActivationSync, Gmm2CombineSync &gmm2CombineSync)
{
    const uint32_t gmm1SchedulerWidth = commonConfig_.gmm1OutputDim;
    gmm1TilesPerMGroup_ = Ops::Base::CeilDiv(gmm1SchedulerWidth, static_cast<uint32_t>(L1_TILE_N));
    gmm2TilesPerMGroup_ = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));

    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    DispatchBuffInit();
    CombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitCombineBuffers();
    } else if (IsCombineDedupOn(params_.tilingData->dedupMode)) {
        combineBufferConfig = InitWaveCombineBuffersDedup(commonConfig_, waveCombineScratch_, waveCombineDedupScratch_);
    }
    PrepareMoeExpertTokenCountTable<ExecutionPolicy::PERSIST_DISPATCH_CUMSUM>(
        commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
    if (tokenDispatchConfig_.dedup.dedupMode != 0) {
        // Arch92 has no separate AIV0 worker or high-half done counter.
        BuildDedupRowDescTable<TopkIndexType>(tokenDispatchConfig_.dedup, commonConfig_, gmmExecutionConfig_.blockJob,
                                             syncWorkspaceLayout_, params_, tokenDispatchScratch_);
    }

    GMMAddrInfo gmm1AddrInfo{};
    ExpertLoopState gmm1ExpertState = CreateExpertLoopState(commonConfig_);
    Gmm2WaveState gmm2State{};
    if constexpr (TopkWeightsPrefetch) {
        gmm1AddrInfo.gmm1ActivationSync = &gmm1ActivationSync;
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (!IsCombineDedupOn(params_.tilingData->dedupMode)) {
            gmm2State.addrInfo.gmm2CombineSync = &gmm2CombineSync;
        }
    }

    gmm2State.expertState = CreateExpertLoopState(commonConfig_);
    gmm2State.combineExpertState = CreateExpertLoopState(commonConfig_);
    DispatchWaveState dispatchState{};
    int32_t gmm1TileSequence = 0;
    GmmRuntimeState gmm1RuntimeState{startBlockIdx_, gmm1TileSequence, gmm1PingPongIdx_};

    ExpertTokenPosition waveBeginPosition{};
    ExpertTokenPosition gmm1Position{};
    // GMM2 的滞后消费目标（见循环内注释）；首轮为零位置，GMM2 空转一轮。
    // 沿用按 token 数开启 lag 的门控；A6 上的性能收益仍需单独测量。
    const bool gmm2LagActive = commonConfig_.tokenNum >= GMM2_LAG_MIN_TOKEN_NUM;
    // AIC 与唯一 AIV 整体滞后一拍的缓存：上一 wave 的边界（见循环内注释）。
    ExpertTokenPosition gmm2PendingWaveEnd{};
    bool hasPendingGmm2Wave = false;

    if constexpr (g_coreType == AIV) {
        WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_, 0U);
    }
    gmm2State.finalCombineExpertIndex = FindFinalCombineExpert();

    while (waveBeginPosition.expertIdx < moeExpertPerRank_) {
        const uint32_t waveStartBlockIndex = startBlockIdx_;
        ExpertTokenPosition waveEndPosition{};
        if constexpr (g_coreType == AIV) {
            // 唯一 AIV 保留 A8W4 的 Dispatch/Activation 职责，分核游标只由实际 GMM 调度推进。
            PrepareDispatchWave(dispatchState, gmm1Count == 0U);
            waveEndPosition = ProcessGmm1Wave(gmm1Position, gmm1ExpertState, gmm1AddrInfo, gmm1RuntimeState);
        } else {
            waveEndPosition = ProcessGmm1Wave(gmm1Position, gmm1ExpertState, gmm1AddrInfo, gmm1RuntimeState);
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        const uint32_t gmm1EndBlockIndex = startBlockIdx_;

        /*
         * GMM2/Combine 的 AIC/AIV 整体滞后一拍（大 bs 门控）：wave w 延至 GMM1(w+1)
         * 之后执行，用下一 wave 的 GMM1 覆盖 AIC 等激活发布的自旋（唯一 AIV 在末批 MMAD
         * 后仍需一段向量时间才发布 activationToGmm2Flag，紧跟消费同一 wave 会让 AIC 反复自旋）。
         * AIC 与唯一 AIV 以相同调用序同拍延后：ProcessGmm2Wave 内部的 pairwise tile
         * 序号、分核游标与 combine 状态在 C/V 两侧保持一致推进，不增加跳过 GMM1 的游标补偿。
         */
        if (gmm2LagActive) {
            if (hasPendingGmm2Wave) {
                ProcessGmm2AndCombineWave(gmm2PendingWaveEnd, gmm2State, combineBufferConfig);
            }
            gmm2PendingWaveEnd = waveEndPosition;
            hasPendingGmm2Wave = true;
        } else {
            ProcessGmm2AndCombineWave(waveEndPosition, gmm2State, combineBufferConfig);
        }

        const bool hasNextWave = waveEndPosition.expertIdx < moeExpertPerRank_;
        const bool fixedRoleResonance =
            startBlockIdx_ == waveStartBlockIndex && gmm1EndBlockIndex != waveStartBlockIndex;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx_ = gmm1EndBlockIndex;
        }

        waveBeginPosition = waveEndPosition;
    }

    // 滞后流水收尾：AIC 与唯一 AIV 共同补跑最后一个 wave 的 GMM2/Combine。
    if (hasPendingGmm2Wave) {
        ProcessGmm2AndCombineWave(gmm2PendingWaveEnd, gmm2State, combineBufferConfig);
    }

    gmm1PingPongIdx_ = 0;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::Process()
{
    this->ProcessWave(*this, GetSharedMmadContext());
}

#undef TemplateMegaMoeArch92WaveTypeClass
#undef TemplateMegaMoeArch92WaveTypeFunc

} // namespace MegaMoeImpl
#endif
