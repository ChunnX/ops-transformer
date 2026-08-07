/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_PIPELINE_H
#define MEGA_MOE_GMM2_COMBINE_PIPELINE_H

#include "adv_api/reduce/reduce.h"
#include "aiv_comm/mega_moe_token_combine.h"
#include "mega_moe_impl.h"

namespace MegaMoeImpl {

using Gmm2ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using Gmm2BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t>;

struct Gmm2ExpertLoopState {
    Gmm2ProblemShape problemShape;
    Gmm2BlockOffset baseOffset;
    uint32_t expertBeforeCnt = 0;
};

struct Gmm2Context {
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    uint32_t tokenNum;
    uint32_t expertPerRank;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
    int32_t swigluFlagSlotsPerExpert;
    int32_t dispatchFlagSlotsPerExpert;
    uint64_t combineSyncSlotCountPerExpert;
    int32_t groupedMatmulMode;
    bool isPerExpertWeightTensor;
};

struct Gmm2Args {
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR swigluDataPtr;
    GM_ADDR swigluScalePtr;
    GM_ADDR gmm2OutputPtr;
    GM_ADDR weight2TensorListAddr;
    GM_ADDR weight2ScaleTensorListAddr;
    GM_ADDR swigluToGmm2FlagPtr;
    GM_ADDR dispatchToGmm1FlagPtr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    GM_ADDR gmmToEpilogueFlagPtr;
};

struct SharedGmm2Args {
    GM_ADDR swigluDataPtr;
    GM_ADDR swigluScalePtr;
    GM_ADDR resultPtr;
    GM_ADDR weight2TensorListAddr;
    GM_ADDR weight2ScaleTensorListAddr;
    GM_ADDR tileCounterPtr;
    GM_ADDR gmmToEpilogueFlagPtr;
};

struct Gmm2Scratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
};

struct Gmm2RuntimeState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

__aicore__ inline Gmm2ExpertLoopState CreateGmm2ExpertLoopState(const Gmm2Context &context)
{
    Gmm2ProblemShape shape;
    Get<N_VALUE>(shape) = context.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.tokenHiddenDim;
    Gmm2BlockOffset offset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return {shape, offset, 0};
}

template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t SwigluElementsPerByte>
__aicore__ inline void AdvanceGmm2ExpertOffsets(Gmm2ExpertLoopState &state)
{
    uint64_t m = Get<M_VALUE>(state.problemShape);
    uint64_t n = Get<N_VALUE>(state.problemShape);
    uint64_t k = Get<K_VALUE>(state.problemShape);
    state.expertBeforeCnt += m;
    Get<IDX_A_OFFSET>(state.baseOffset) += m * k / ActivationElementsPerByte;
    Get<IDX_B_OFFSET>(state.baseOffset) += n * k / WeightElementsPerByte;
    uint64_t scaleK =
        Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
    Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
    Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / SwigluElementsPerByte;
    Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
        m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
    Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / WeightElementsPerByte;
    Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
        k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
    Get<IDX_M_OFFSET>(state.baseOffset) += m;
    Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
    Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
}

// 原型：MegaMoe::UpdateGmm2GroupParams。使用调用方选择的 workspace 槽更新一个 MoE 专家游标。
template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t SwigluElementsPerByte>
__aicore__ inline bool UpdateMoeGmm2GroupParams(const Gmm2Context &context, Gmm2Scratch &scratch,
                                                Gmm2ExpertLoopState &state, uint32_t expertIdx)
{
    if (expertIdx != 0U) {
        AdvanceGmm2ExpertOffsets<ActivationElementsPerByte, WeightElementsPerByte,
                                 SwigluElementsPerByte>(state);
    }
    uint64_t countOffset = static_cast<uint64_t>(expertIdx) * INT32_PER_256B * context.countWorkspace.blockNum +
                           static_cast<uint64_t>(INT32_PER_256B) * context.countWorkspace.blockIdx;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        scratch.expertRevNumsGlobalTensor[countOffset]);
    Get<M_VALUE>(state.problemShape) = scratch.expertRevNumsGlobalTensor.GetValue(countOffset);
    return Get<M_VALUE>(state.problemShape) != 0;
}

// 原型：MegaMoe::UpdateSharedGroupParams。使用固定 token 数推进一个共享专家。
template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t SwigluElementsPerByte>
__aicore__ inline bool UpdateSharedGmm2GroupParams(const Gmm2Context &context,
                                                   Gmm2ExpertLoopState &state, uint32_t expertIdx)
{
    if (expertIdx != 0U) {
        AdvanceGmm2ExpertOffsets<ActivationElementsPerByte, WeightElementsPerByte,
                                 SwigluElementsPerByte>(state);
    }
    Get<M_VALUE>(state.problemShape) = context.tokenNum;
    return Get<M_VALUE>(state.problemShape) != 0;
}

template <typename ActivationType, typename SwigluOutType, typename QuantScaleType,
          bool ConfigureGmm2Output, bool ConfigureCombineCounter, bool ConfigureGmmToEpilogue>
__aicore__ inline void UpdateMoeGmm2GlobalBuffer(const Gmm2Context &context, const Gmm2Args &args,
                                                 GMMAddrInfo &gmmAddrInfo,
                                                 const Gmm2ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (ConfigureGmm2Output) {
        gmmAddrInfo.gmm2OutGlobal =
            args.gmm2OutputPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal =
        args.swigluDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(SwigluOutType);
    gmmAddrInfo.aScaleGlobal =
        args.swigluScalePtr + Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        args.weight2TensorListAddr, context.isPerExpertWeightTensor, expertIdx,
        Get<IDX_B2_OFFSET>(state.baseOffset));
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        args.weight2ScaleTensorListAddr, context.isPerExpertWeightTensor, expertIdx,
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    if constexpr (ConfigureCombineCounter) {
        uint64_t expertSyncSlotOffset =
            static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) * context.combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter =
            reinterpret_cast<__gm__ int32_t *>(args.gmm2CombineSyncCounterPtr) +
            expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
    }
    gmmAddrInfo.swigluToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(args.swigluToGmm2FlagPtr) +
                                   Get<IDX_FLAG_OFFSET>(state.baseOffset) * context.swigluFlagSlotsPerExpert;
    gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(args.dispatchToGmm1FlagPtr) +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * context.dispatchFlagSlotsPerExpert;
    if constexpr (ConfigureGmmToEpilogue) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }
}

template <typename ActivationType, typename QuantScaleType, uint32_t Gmm1TileM,
          bool ConfigureGmmToEpilogue>
__aicore__ inline void UpdateSharedGmm2GlobalBuffer(const Gmm2Context &context, const SharedGmm2Args &args,
                                                    GMMAddrInfo &gmmAddrInfo,
                                                    const Gmm2ExpertLoopState &state, uint32_t sharedExpertIdx)
{
    gmmAddrInfo.gmm2OutGlobal =
        args.resultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    gmmAddrInfo.aGlobal = args.swigluDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal =
        args.swigluScalePtr + Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        args.weight2TensorListAddr, context.isPerExpertWeightTensor, sharedExpertIdx,
        Get<IDX_B2_OFFSET>(state.baseOffset));
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        args.weight2ScaleTensorListAddr, context.isPerExpertWeightTensor, sharedExpertIdx,
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(context.tokenNum, Gmm1TileM);
    gmmAddrInfo.sharedExpertGmm2TileCounter =
        reinterpret_cast<__gm__ int32_t *>(args.tileCounterPtr) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
    if constexpr (ConfigureGmmToEpilogue) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }
}

// 原型：MegaMoe::GroupMatmulWithCombine / MegaMoeWave::GroupMatmulGmm2。
// 使用调用方传入的 block 任务执行 GMM2，并保持所选同步约定不变。
template <uint8_t CombineMode, typename GenericElementA, typename A8W4ElementA, typename WeightType,
          typename QuantScaleType, bool EnableA8W4, bool EnableA4W4, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch, bool IsShared, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm2(const Gmm2Context &context, const Params &params,
                               const GMMAddrInfo &gmmAddrInfo, const Gmm2ExpertLoopState &state,
                               Gmm2RuntimeState &runtimeState)
{
    if constexpr (EnableA8W4 || EnableA4W4) {
        GroupMatmul2CombineA8W4<CombineMode, A8W4ElementA, WeightType, bfloat16_t, QuantScaleType,
                                QuantScaleType, Gmm1TileM, TopkWeightsPrefetch, IsShared>(
            params, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.gmTileSequence,
            context.blockJob, state.expertBeforeCnt, runtimeState.pingpongIdx);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        GroupMatmul2<CombineMode, GenericElementA, GenericElementA, bfloat16_t, QuantScaleType,
                     QuantScaleType, true, false, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                     IsGmm1Interleaved, IsWaveFlagGrained>(
            params, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            context.blockJob, state.expertBeforeCnt, runtimeState.pingpongIdx);
    } else {
        GroupMatmul2<CombineMode, GenericElementA, GenericElementA, bfloat16_t, QuantScaleType,
                     QuantScaleType, false, false, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                     IsGmm1Interleaved, IsWaveFlagGrained>(
            params, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            context.blockJob, state.expertBeforeCnt, runtimeState.pingpongIdx);
    }
}

// 原型：MegaMoe::ProcessSharedExpertGmm2 / MegaMoeWave::ProcessSharedExpertGmm2。
// 使用可复用的 GMM2 block 任务处理全部共享专家。
template <typename GenericElementA, typename A8W4ElementA, typename WeightType, typename QuantScaleType,
          bool EnableA8W4, bool EnableA4W4, uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void ProcessSharedExpertGmm2(const Gmm2Context &context, const SharedGmm2Args &args,
                                              const Params &params, Gmm2RuntimeState &runtimeState)
{
    Gmm2ExpertLoopState state = CreateGmm2ExpertLoopState(context);
    GMMAddrInfo gmmAddrInfo{};
    for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < context.expertPerRank; ++sharedExpertIdx) {
        if (!UpdateSharedGmm2GroupParams<PackedElementTraits<GenericElementA>::ELEMENTS_PER_BYTE,
                                         PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE,
                                         PackedElementTraits<A8W4ElementA>::ELEMENTS_PER_BYTE>(
                context, state, sharedExpertIdx)) {
            continue;
        }
        UpdateSharedGmm2GlobalBuffer<A8W4ElementA, QuantScaleType, Gmm1TileM,
                                     EnableA8W4 || EnableA4W4>(
            context, args, gmmAddrInfo, state, sharedExpertIdx);
        RunGmm2<COMBINE_NO_QUANT, GenericElementA, A8W4ElementA, WeightType, QuantScaleType,
                EnableA8W4, EnableA4W4, Gmm1TileM, TopkWeightsPrefetch, true, IsGmm1Interleaved,
                IsWaveFlagGrained>(
            context, params, gmmAddrInfo, state, runtimeState);
    }
}

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

constexpr uint32_t WAVE_GMM2_READY_SCAN_UB_ADDR = 184U * 1024U;
constexpr uint32_t WAVE_GMM2_READY_MAX_SCAN_BYTES =
    MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512 ?
        MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) :
        ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR =
    WAVE_GMM2_READY_SCAN_UB_ADDR +
    ((WAVE_GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_SUM_UB_ADDR = WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;

struct WaveGmm2ReadyArgs {
    GM_ADDR gmm2ReadyPtr;
};

__aicore__ inline uint64_t GetWaveGmm2ReadySlotStride(const WaveCombineContext &context)
{
    return static_cast<uint64_t>(context.job.totalJobs) * INT_CACHELINE;
}

// 每个 AIC 在 GMM2 写入可见后发布一个独占 cache line 的完成标记。
__aicore__ inline void NotifyWaveGmm2Ready(const WaveCombineContext &context,
                                           const WaveGmm2ReadyArgs &args, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(args.gmm2ReadyPtr);
        uint64_t slotOffset =
            static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(context);
        AscendC::WriteGmByPassDCache(
            readyBase + slotOffset + static_cast<uint64_t>(context.job.jobIndex) * INT_CACHELINE,
            int32_t(1));
    }
}

// 一个 AIV Combine 任务等待全部 AIC 完成指定专家。
__aicore__ inline void WaitWaveGmm2Ready(const WaveCombineContext &context,
                                         const WaveGmm2ReadyArgs &args, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!context.participates || context.job.totalJobs == 0U) {
        return;
    }
    uint32_t readyElements = context.job.totalJobs * static_cast<uint32_t>(INT_CACHELINE);
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(args.gmm2ReadyPtr);
    uint64_t readySlotOffset =
        static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(context);
    GlobalTensor<int32_t> readyGm;
    readyGm.SetGlobalBuffer(readyBase);
    LocalTensor<int32_t> scanUb(TPosition::VECCALC, WAVE_GMM2_READY_SCAN_UB_ADDR, readyElements);
    LocalTensor<uint8_t> reduceTmpUb(
        TPosition::VECCALC, WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR, ALIGN_512);
    LocalTensor<int32_t> sumUb(
        TPosition::VECCALC, WAVE_GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);
    const uint32_t readyShape[] = {1U, readyElements};
    while (true) {
        DataCopy(scanUb, readyGm[readySlotOffset], readyElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ReduceSum<int32_t, AscendC::Pattern::Reduce::AR, true>(
            sumUb, scanUb, reduceTmpUb, readyShape, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (sumUb.GetValue(0) >= static_cast<int32_t>(context.job.totalJobs)) {
            return;
        }
        int64_t waitStart = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStart < 100) {
        }
    }
}

// 原型：MegaMoeWave::ProcessGmm2Wave。
// AIC 执行一个专家 wave，同时 AIV0 推进对应的虚拟 GMM2 游标。
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
        if (!UpdateMoeGmm2GroupParams<1U, 1U, 1U>(context, scratch, state, expertIdx)) {
            continue;
        }
        UpdateMoeGmm2GlobalBuffer<ActivationType, ActivationType, QuantScaleType,
                                  true, false, false>(
            context, args, gmmAddrInfo, state, expertIdx);
        RunGmm2<COMBINE_NO_QUANT, QuantOutType, ActivationType, WeightType,
                QuantScaleType, false, false, Gmm1TileM, TopkWeightsPrefetch, false,
                IsGmm1Interleaved, true>(
            context, params, gmmAddrInfo, state, runtimeState);
        NotifyWaveGmm2Ready(combineContext, readyArgs, expertIdx);
    }
}

// 原型：MegaMoeWave::ProcessCombineExperts。
// 等待一个 wave 的 GMM2 汇聚完成，并发送 AIV1 任务负责的 token 范围。
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
        if (!UpdateMoeGmm2GroupParams<1U, 1U, 1U>(
                gmm2Context, gmm2Scratch, state, expertIdx)) {
            continue;
        }
        UpdateMoeGmm2GlobalBuffer<ActivationType, ActivationType, QuantScaleType,
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

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_PIPELINE_H
