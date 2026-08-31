/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_ACTIVATION_H
#define MEGA_MOE_GMM1_ACTIVATION_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "../epilogue/block_epilogue_activation_mx_quant.h"
#include "../common/mega_moe_types.h"
#include "../moe_gmm_sync_policy.h"
#include "apace/kernel/matmul/grouped_matmul/run_gmm1_generic.h"
#include "apace/kernel/matmul/grouped_matmul/run_gmm1_a8w4.h"

namespace MegaMoeImpl {

using namespace AscendC;

using Apace::Gemm::Kernel::Gmm1GenericKernel;
using Apace::Gemm::Kernel::Gmm1A8W4Kernel;


template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, bool EnableA8W4, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBuffer(
    const GmmExecutionConfig &gmmConfig, const MoeSyncWorkspaceLayout &syncLayout, const WorkspaceInfo &workspace,
    const ExpertWeightTensorListAddrs &weights, BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state, uint32_t rowOffsetInExpert = 0U, uint32_t gmm1TilesPerMGroup = 0U)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = state.expertIdx;
    int64_t globalTokenStartIndex = state.globalTokenStartIndex + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = workspace.gmm1MmadResPtr + globalTokenStartIndex * n * sizeof(bfloat16_t);
    }
    if constexpr (TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1TileStatus = reinterpret_cast<__gm__ int32_t *>(workspace.gmm1TileStatusPtr) +
                                     (static_cast<uint64_t>(state.expertIdx) * syncLayout.gmm1TileStatusCountPerExpert +
                                      static_cast<uint64_t>(expertMGroupOffset) * gmm1TilesPerMGroup) *
                                         INT_CACHELINE;
    }
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr;
    gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
                                       expertOffset * syncLayout.activationFlagSlotCountPerExpert +
                                       static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal =
        workspace.dispatchRevDataPtr + globalTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.dispatchRevScalePtr + globalTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleType>(weights.weightScales1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsActivation = true;
        if constexpr (EnableA8W4) {
            runsActivation = GetSubBlockIdx() == 1;
        }
        if (runsActivation) {
            int64_t scaleN = Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                globalTokenStartIndex * n / ACTIVATION_N_HALF / outputElementsPerByte,
                globalTokenStartIndex * scaleN,
                expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
                                         expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
                                         static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }
}

template <typename QuantOutType, typename QuantScaleType, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, 
          bool IsShared = false, typename BlockEpilogue>
__aicore__ inline void RunGmm1GenericByWeightFormat(const GmmExecutionConfig &gmmConfig, const Params &gmmParams,
                                                     BlockEpilogue &epilogueOp, const GMMAddrInfo &gmmAddrInfo,
                                                     const ProblemShape &problemShape, uint32_t globalTokenStartIndex,
                                                     GmmRuntimeState &runtimeState, uint32_t expertIdx,
                                                     void *persistentBlockMmadContext = nullptr,
                                                     bool allowWeightL2Bypass = false)
{
    uint32_t m = static_cast<uint32_t>(Get<M_VALUE>(problemShape));
    if constexpr (TopkWeightsPrefetch) {
        epilogueOp.SetMetaInfo(gmmAddrInfo.metaInfoGlobal, globalTokenStartIndex);
    }
    Gmm1GenericSyncPolicy<TopkWeightsPrefetch, IsShared, IsWaveFlagGrained, IsGmm1Interleaved> sync(
        gmmAddrInfo.dispatchToGmm1Flag, gmmAddrInfo.activationToGmm2Flag,
        gmmAddrInfo.gmm1TileStatus, expertIdx, runtimeState.vecSetSyncCom, runtimeState.pingpongIdx, m, Gmm1TileM);
    if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
        gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        using GmmKernel = Gmm1GenericKernel<decltype(sync), BlockEpilogue, QuantOutType,
                                            QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, true,
                                            Gmm1TileM, TopkWeightsPrefetch, IsGmm1Interleaved, IsWaveFlagGrained>;
        typename GmmKernel::Params params{gmmAddrInfo, problemShape, &runtimeState.startBlockIdx,
                                 &runtimeState.pingpongIdx, persistentBlockMmadContext, allowWeightL2Bypass};
        GmmKernel kernel;
        kernel(sync, epilogueOp, params);
    } else {
        using GmmKernel = Gmm1GenericKernel<decltype(sync), BlockEpilogue, QuantOutType,
                                            QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, false,
                                            Gmm1TileM, TopkWeightsPrefetch, IsGmm1Interleaved, IsWaveFlagGrained>;
        typename GmmKernel::Params params{gmmAddrInfo, problemShape, &runtimeState.startBlockIdx,
                                 &runtimeState.pingpongIdx, persistentBlockMmadContext, allowWeightL2Bypass};
        GmmKernel kernel;
        kernel(sync, epilogueOp, params);
    }
}

// 原型：MegaMoe::UpdateSharedGlobalBuffer<GMM1>。构造一个共享专家的 GMM1 和 SwiGLU GM 视图。
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const MoeStageCommonConfig &commonConfig,
                                                          const GmmExecutionConfig &gmmConfig,
                                                          const WorkspaceInfo &workspace,
                                                          const ExpertWeightTensorListAddrs &weights,
                                                          BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = commonConfig.tokenNum;
    uint64_t n = commonConfig.gmm1OutputDim;
    uint64_t k = commonConfig.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t activationN = n / ACTIVATION_N_HALF;
    uint64_t scaleN = Ops::Base::CeilDiv(activationN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    gmmAddrInfo.aGlobal = workspace.sharedExpertInputDataPtr;
    gmmAddrInfo.aScaleGlobal = workspace.sharedExpertInputScalePtr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = workspace.sharedExpertGmm1OutPtr + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor,
                                                              sharedExpertIdx, expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales1, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx, expertIdx * n * scaleK);
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.metaInfoGlobal = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * activationN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

// 执行一个专家的 GMM1/SwiGLU 阶段。参数化 TopkWeightsPrefetch 和 IsShared，
// 供共享专家和 MoE 专家统一调用。
template <typename QuantOutType, typename WeightType, typename QuantScaleType,
          bool EnableA8W4, uint32_t Gmm1TileM, bool TopkWeightsPrefetch, bool IsShared,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void RunGmm1ActivationStage(const GmmExecutionConfig &gmmConfig, const Params &gmmParams,
                                              BlockEpilogue &epilogueOp, const GMMAddrInfo &gmmAddrInfo,
                                              const ProblemShape &problemShape,
                                              GmmRuntimeState &runtimeState, uint32_t expertBeforeCnt,
                                              uint32_t expertIdx, void *persistentBlockMmadContext = nullptr,
                                              bool allowWeightL2Bypass = false)
{
    if constexpr (EnableA8W4) {
        uint32_t m = static_cast<uint32_t>(Get<M_VALUE>(problemShape));
        epilogueOp.SetMetaInfo(gmmAddrInfo.metaInfoGlobal, expertBeforeCnt);
        Gmm1A8W4SyncPolicy<TopkWeightsPrefetch, IsShared, IsWaveFlagGrained> sync(gmmAddrInfo.dispatchToGmm1Flag,
            gmmAddrInfo.activationToGmm2Flag, gmmAddrInfo.gmm1TileStatus, expertIdx,
            gmmAddrInfo.gmmToEpilogueFlag, runtimeState.gmTileSequence, m, Gmm1TileM);
        using GmmKernel = Gmm1A8W4Kernel<decltype(sync), BlockEpilogue, QuantOutType, WeightType, bfloat16_t,
                                         QuantScaleType, QuantScaleType, Gmm1TileM>;
        typename GmmKernel::Params params{gmmAddrInfo, problemShape, &runtimeState.startBlockIdx};
        GmmKernel kernel;
        kernel(sync, epilogueOp, params);
    } else {
        RunGmm1GenericByWeightFormat<QuantOutType, QuantScaleType, Gmm1TileM,
                                     TopkWeightsPrefetch, IsGmm1Interleaved, IsWaveFlagGrained, IsShared>(
            gmmConfig, gmmParams, epilogueOp, gmmAddrInfo, problemShape, expertBeforeCnt, runtimeState, expertIdx,
            persistentBlockMmadContext, allowWeightL2Bypass);
    }
}

// 共享专家的 GMM1/SwiGLU 阶段。固定 TopkWeightsPrefetch=false, IsShared=true。
template <typename QuantOutType, typename WeightType, typename QuantScaleType,
          bool EnableA8W4, uint32_t Gmm1TileM, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename BlockEpilogue>
__aicore__ inline void RunSharedExpertGmm1ActivationStage(const MoeStageCommonConfig &commonConfig,
                                                          const GmmExecutionConfig &gmmConfig, const Params &gmmParams,
                                                          BlockEpilogue &epilogueOp, const GMMAddrInfo &gmmAddrInfo,
                                                          const ProblemShape &problemShape,
                                                          GmmRuntimeState &runtimeState, uint32_t sharedExpertIdx,
                                                          void *persistentBlockMmadContext = nullptr,
                                                          bool allowWeightL2Bypass = false)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * commonConfig.tokenNum;
    RunGmm1ActivationStage<QuantOutType, WeightType, QuantScaleType,
                           EnableA8W4, Gmm1TileM, false, true, IsGmm1Interleaved, IsWaveFlagGrained>(
        gmmConfig, gmmParams, epilogueOp, gmmAddrInfo, problemShape, runtimeState, expertBeforeCnt, sharedExpertIdx,
        persistentBlockMmadContext, allowWeightL2Bypass);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_ACTIVATION_H
