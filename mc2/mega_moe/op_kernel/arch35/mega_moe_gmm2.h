/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_H
#define MEGA_MOE_GMM2_H

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

// Prototype: MegaMoe::UpdateGmm2GroupParams. Updates one routed expert cursor using the caller-selected workspace slot.
template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t SwigluElementsPerByte>
__aicore__ inline bool UpdateRoutedGmm2GroupParams(const Gmm2Context &context, Gmm2Scratch &scratch,
                                                   Gmm2ExpertLoopState &state,
    uint32_t expertIdx)
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

// Prototype: MegaMoe::UpdateSharedGroupParams. Advances one shared expert with its fixed token count.
template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t SwigluElementsPerByte>
__aicore__ inline bool UpdateSharedGmm2GroupParams(const Gmm2Context &context,
                                                   Gmm2ExpertLoopState &state,
    uint32_t expertIdx)
{
    if (expertIdx != 0U) {
        AdvanceGmm2ExpertOffsets<ActivationElementsPerByte, WeightElementsPerByte,
                                 SwigluElementsPerByte>(state);
    }
    Get<M_VALUE>(state.problemShape) = context.tokenNum;
    return true;
}

template <typename ActivationType, typename SwigluOutType, typename QuantScaleType,
          bool StoreGmm2Output, bool ConfigureCombineCounter, bool ConfigureGmmToEpilogue>
__aicore__ inline void UpdateRoutedGmm2GlobalBuffer(const Gmm2Context &context, const Gmm2Args &args,
                                                    GMMAddrInfo &gmmAddrInfo,
                                                    const Gmm2ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (StoreGmm2Output) {
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
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (ConfigureGmmToEpilogue) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }
}

// Prototype: MegaMoe::GroupMatmulWithCombine / MegaMoeWave::GroupMatmulGmm2.
// Runs GMM2 with the block job supplied by the caller while preserving the selected synchronization contract.
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

// Prototype: MegaMoe::ProcessSharedExpertGmm2 / MegaMoeWave::ProcessSharedExpertGmm2.
// Runs all shared experts through the reusable GMM2 block job.
template <typename GenericElementA, typename A8W4ElementA, typename WeightType, typename QuantScaleType,
          bool EnableA8W4, bool EnableA4W4, uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void ControlSharedGmm2Pipeline(const Gmm2Context &context, const SharedGmm2Args &args,
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

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_H
