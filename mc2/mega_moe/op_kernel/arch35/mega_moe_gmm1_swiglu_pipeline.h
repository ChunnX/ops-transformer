/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_SWIGLU_PIPELINE_H
#define MEGA_MOE_GMM1_SWIGLU_PIPELINE_H

#include "aiv_comm/mega_moe_token_dispatch.h"
#include "aiv_compute/mega_moe_expert_token_count.h"
#include "mega_moe_impl.h"

namespace MegaMoeImpl {

using Gmm1ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

struct Gmm1ExpertLoopState {
    Gmm1ProblemShape problemShape;
    int64_t rowOffset;
};

struct Gmm1SwigluContext {
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    uint32_t expertCapacityPerRank;
    uint32_t expertPerRank;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
    int32_t dispatchFlagSlotsPerExpert;
    int32_t swigluFlagSlotsPerExpert;
    uint32_t maxTilesPerExpert;
    int32_t groupedMatmulMode;
    uint32_t activationElementsPerByte;
    bool isPerExpertWeightTensor;
};

struct Gmm1SwigluArgs {
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR gmm1MmadResPtr;
    GM_ADDR weight1TensorListAddr;
    GM_ADDR weight1ScaleTensorListAddr;
    GM_ADDR dispatchToGmm1FlagPtr;
    GM_ADDR sendCntFlagPtr;
    GM_ADDR gmm1TileStatusPtr;
    GM_ADDR gmmToEpilogueFlagPtr;
};

struct Gmm1SwigluScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
};

struct Gmm1SwigluState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

__aicore__ inline Gmm1ExpertLoopState CreateGmm1ExpertLoopState(const Gmm1SwigluContext &context)
{
    Gmm1ProblemShape shape;
    Get<N_VALUE>(shape) = context.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.tokenHiddenDim;
    return Gmm1ExpertLoopState{shape, 0};
}

// 必要时等待 token 计数生产完成，并更新当前专家 GMM1 的 M 维度。
template <bool EnableA8W4>
__aicore__ inline bool WaitAndUpdateGmm1GroupParams(const Gmm1SwigluContext &context,
                                                    const Gmm1SwigluArgs &args, Gmm1SwigluScratch &scratch,
                                                    Gmm1ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            Get<M_VALUE>(state.problemShape) = sendCnt;
            return sendCnt != 0;
        }
    }

    if (expertIdx != 0) {
        state.rowOffset += Get<M_VALUE>(state.problemShape);
    }

    if (GetSubBlockIdx() == 0) {
        __gm__ int32_t *sendCntFlag =
            reinterpret_cast<__gm__ int32_t *>(args.sendCntFlagPtr) +
            static_cast<uint64_t>(expertIdx) * context.countWorkspace.blockNum * INT_CACHELINE +
            static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
        while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
        uint64_t offset = expertIdx * INT32_PER_256B * context.countWorkspace.blockNum +
                          INT32_PER_256B * context.countWorkspace.blockIdx;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            scratch.expertRevNumsGlobalTensor[offset]);
        Get<M_VALUE>(state.problemShape) = scratch.expertRevNumsGlobalTensor.GetValue(offset);
    } else {
        Get<M_VALUE>(state.problemShape) = sendCnt;
    }
    return Get<M_VALUE>(state.problemShape) != 0;
}

template <typename ActivationType, typename WeightType, typename SwigluOutType, typename QuantScaleType,
          bool EnableA8W4, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void UpdateGmm1GlobalBuffer(
    const Gmm1SwigluContext &context, const Gmm1SwigluArgs &args, BlockEpilogue &epilogueOp,
    GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<SwigluOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = expertIdx;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResPtr + state.rowOffset * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal =
        args.dispatchRevDataPtr + state.rowOffset * k / context.activationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = args.dispatchRevScalePtr + state.rowOffset * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        args.weight1TensorListAddr, context.isPerExpertWeightTensor, expertIdx,
        static_cast<uint64_t>(expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        args.weight1ScaleTensorListAddr, context.isPerExpertWeightTensor, expertIdx,
        static_cast<uint64_t>(expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsSwiglu = true;
        if constexpr (EnableA8W4) {
            runsSwiglu = GetSubBlockIdx() == 1;
        }
        if (runsSwiglu) {
            int64_t scaleN =
                Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                state.rowOffset * n / SWIGLU_N_HALF / outputElementsPerByte,
                state.rowOffset * scaleN,
                expertOffset * context.swigluFlagSlotsPerExpert / INT_CACHELINE,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(args.dispatchToGmm1FlagPtr) +
                                         expertOffset * context.dispatchFlagSlotsPerExpert;
    }
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }
}

template <typename QuantOutType, typename WeightType, typename SwigluOutType, typename QuantScaleType, bool EnableA8W4,
          uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void Gmm1SwigluPipelineByMode(
    const Gmm1SwigluContext &context, const Params &gmmParams, BlockEpilogue &epilogueOp,
    const GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state, Gmm1SwigluState &runtimeState,
    uint32_t expertIdx)
{
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType,
                                                Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, false>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.gmTileSequence, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, true, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                            false, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx,
            runtimeState.pingpongIdx);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, false, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                            false, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx,
            runtimeState.pingpongIdx);
    }
}

using SharedExpertGmm1ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

struct SharedExpertGmm1SwigluContext {
    BlockJobContext blockJob;
    uint32_t sharedExpertNum;
    uint32_t tokenNum;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
    int32_t groupedMatmulMode;
    bool isPerExpertWeightTensor;
};

struct SharedExpertGmm1SwigluArgs {
    GM_ADDR inputDataPtr;
    GM_ADDR inputScalePtr;
    GM_ADDR gmm1MmadResultPtr;
    GM_ADDR weight1TensorListAddr;
    GM_ADDR weight1ScaleTensorListAddr;
    GM_ADDR swigluOutputDataPtr;
    GM_ADDR swigluOutputScalePtr;
    GM_ADDR gmmToEpilogueFlagPtr;
    float clampLimit;
    uint8_t actMode;
    uint8_t actSubMode;
    float activationAlpha;
    float activationBeta;
};

struct SharedExpertGmm1SwigluState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

// 原型：MegaMoe::UpdateSharedGlobalBuffer<GMM1>。构造一个共享专家的 GMM1 和 SwiGLU GM 视图。
template <typename ActivationType, typename WeightType, typename SwigluOutType, typename QuantScaleType,
          bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const SharedExpertGmm1SwigluContext &context,
                                                          const SharedExpertGmm1SwigluArgs &args,
                                                          BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<SwigluOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = context.tokenNum;
    uint64_t n = context.gmm1OutputDim;
    uint64_t k = context.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t swigluN = n / SWIGLU_N_HALF;
    uint64_t scaleN = Ops::Base::CeilDiv(swigluN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    gmmAddrInfo.aGlobal = args.inputDataPtr;
    gmmAddrInfo.aScaleGlobal = args.inputScalePtr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResultPtr + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        args.weight1TensorListAddr, context.isPerExpertWeightTensor, sharedExpertIdx,
        expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        args.weight1ScaleTensorListAddr, context.isPerExpertWeightTensor, sharedExpertIdx,
        expertIdx * n * scaleK);
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * swigluN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

// 原型：MegaMoe::GroupMatmulWithSwigluQuant<true>。执行一个共享专家的 GMM1/SwiGLU 阶段。
template <typename QuantOutType, typename WeightType, typename SwigluOutType, typename QuantScaleType, bool EnableA8W4,
          uint32_t Gmm1TileM, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename BlockEpilogue>
__aicore__ inline void RunSharedExpertGmm1SwigluStage(const SharedExpertGmm1SwigluContext &context,
                                                      const Params &gmmParams, BlockEpilogue &epilogueOp,
                                                      const GMMAddrInfo &gmmAddrInfo,
                                                      const SharedExpertGmm1ProblemShape &problemShape,
                                                      SharedExpertGmm1SwigluState &runtimeState,
                                                      uint32_t sharedExpertIdx)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * context.tokenNum;
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType,
                                                Gmm1TileM, MegaMoeImpl::L1_TILE_M_256, false, true>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.gmTileSequence,
            context.blockJob, expertBeforeCnt, sharedExpertIdx);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, true, Gmm1TileM, MegaMoeImpl::L1_TILE_M_256, false, true,
                                            IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            context.blockJob, expertBeforeCnt, sharedExpertIdx, runtimeState.pingpongIdx);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, false, Gmm1TileM, MegaMoeImpl::L1_TILE_M_256, false, true,
                                            IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            context.blockJob, expertBeforeCnt, sharedExpertIdx, runtimeState.pingpongIdx);
    }
}

// 原型：MegaMoe::ProcessSharedExpertGmm1。处理一个逻辑 MIX block 任务上的共享专家 GMM1 和 SwiGLU。
template <typename QuantOutType, typename ActivationType, typename WeightType, typename SwigluOutType,
          typename QuantScaleType, bool EnableA8W4, uint32_t Gmm1TileM, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void SharedExpertGmm1SwigluPipeline(
    const SharedExpertGmm1SwigluContext &context, const SharedExpertGmm1SwigluArgs &args,
    const Params &gmmParams, BlockEpilogue &epilogueOp, SharedExpertGmm1SwigluState &runtimeState)
{
    if (context.blockJob.totalJobs == 0U || context.blockJob.jobIndex >= context.blockJob.totalJobs ||
        context.sharedExpertNum == 0U) {
        return;
    }

    typename BlockEpilogue::Params epilogueParams{
        args.swigluOutputDataPtr, args.swigluOutputScalePtr, nullptr, nullptr, nullptr, nullptr, nullptr,
        args.clampLimit, args.actMode, args.actSubMode, args.activationAlpha, args.activationBeta};
    epilogueOp.Init(epilogueParams);

    SharedExpertGmm1ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = context.tokenNum;
    Get<N_VALUE>(problemShape) = context.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = context.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};

    for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < context.sharedExpertNum; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4>(
            context, args, epilogueOp, gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm1SwigluStage<QuantOutType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                                       Gmm1TileM, IsGmm1Interleaved, IsWaveFlagGrained>(
            context, gmmParams, epilogueOp, gmmAddrInfo, problemShape, runtimeState, sharedExpertIdx);
    }
    EndSync<IsGmm1Interleaved>(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
}

// 组合结构只负责把当前普通模板使用的 Token Dispatch 与 GMM1/SwiGLU 接口组织在一起。
struct DispatchGmm1SwigluContext {
    TokenDispatchContext dispatch;
    Gmm1SwigluContext gmm1;
};

struct DispatchGmm1SwigluArgs {
    TokenDispatchArgs dispatch;
    Gmm1SwigluArgs gmm1;
};

using DispatchGmm1SwigluState = Gmm1SwigluState;

// 调度普通模板的 SendCount -> Dispatch -> GMM1 -> SwiGLU 流水。
// 同一个 MIX_AIC_1_2 block 的物理核接收相同逻辑 block 任务，各阶段只保留自身约定的同步。
template <typename QuantOutType, typename ActivationType, typename WeightType, typename SwigluOutType,
          typename QuantScaleType, bool EnableA8W4, uint32_t PipelineTileM, uint32_t EpilogueTileM,
          bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void DispatchGmm1SwigluPipeline(
    const DispatchGmm1SwigluContext &context, const DispatchGmm1SwigluArgs &args, const Params &gmmParams,
    BlockEpilogue &epilogueOp, TokenDispatchScratch<ActivationType> &dispatchScratch,
    Gmm1SwigluScratch &gmm1Scratch, DispatchGmm1SwigluState &runtimeState)
{
    const TokenDispatchContext &dispatchContext = context.dispatch;
    const Gmm1SwigluContext &gmm1Context = context.gmm1;
    if (dispatchContext.blockJob.totalJobs == 0U ||
        dispatchContext.blockJob.jobIndex >= dispatchContext.blockJob.totalJobs ||
        dispatchContext.countWorkspace.blockNum == 0U ||
        dispatchContext.countWorkspace.blockIdx >= dispatchContext.countWorkspace.blockNum ||
        gmm1Context.expertPerRank == 0U || dispatchContext.blockNumPerRank == 0U) {
        return;
    }

    InitTokenDispatchBuffers<ActivationType, EnableA8W4>(dispatchContext, args.dispatch, dispatchScratch);
    gmm1Scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(args.gmm1.expertRevTokenNumsPtr));
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(gmm1Context);
    GMMAddrInfo gmm1AddrInfo;
    uint64_t currentSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1) {
            ComputeExpertTokenCountAndNotify<ActivationType, EnableA8W4, TopkWeightsPrefetch>(
                dispatchContext, args.dispatch, dispatchScratch, 0, nextSendCnt);
            if (nextSendCnt != 0) {
                DispatchExpertTokens<ActivationType, QuantScaleType, EnableA8W4, PipelineTileM, TopkWeightsPrefetch>(
                    dispatchContext, args.dispatch, dispatchScratch, 0);
            }
        }
    }

    for (uint32_t expertIdx = 0; expertIdx < gmm1Context.expertPerRank; ++expertIdx) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1) {
                currentSendCnt = nextSendCnt;
                if (expertIdx + 1 < gmm1Context.expertPerRank) {
                    ComputeExpertTokenCountAndNotify<ActivationType, EnableA8W4, TopkWeightsPrefetch>(
                        dispatchContext, args.dispatch, dispatchScratch, expertIdx + 1, nextSendCnt);
                    if (nextSendCnt != 0) {
                        DispatchExpertTokens<ActivationType, QuantScaleType, EnableA8W4, PipelineTileM,
                                             TopkWeightsPrefetch>(
                            dispatchContext, args.dispatch, dispatchScratch, expertIdx + 1);
                    }
                }
            }
        }

        if (!WaitAndUpdateGmm1GroupParams<EnableA8W4>(
                gmm1Context, args.gmm1, gmm1Scratch, gmm1State, expertIdx, currentSendCnt)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                               TopkWeightsPrefetch>(
            gmm1Context, args.gmm1, epilogueOp, gmm1AddrInfo, gmm1State, expertIdx);
        Gmm1SwigluPipelineByMode<QuantOutType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                                        PipelineTileM, EpilogueTileM, TopkWeightsPrefetch>(
            gmm1Context, gmmParams, epilogueOp, gmm1AddrInfo, gmm1State, runtimeState, expertIdx);
    }

    if constexpr (TopkWeightsPrefetch) {
        int32_t allDoneTag = static_cast<int32_t>(gmm1Context.expertCapacityPerRank + 1U);
        __gm__ int32_t *allDoneAddr =
            reinterpret_cast<__gm__ int32_t *>(args.gmm1.gmm1TileStatusPtr) +
            static_cast<uint64_t>(gmm1Context.expertCapacityPerRank) * gmm1Context.maxTilesPerExpert * INT_CACHELINE;
        if constexpr (g_coreType == AIV) {
            constexpr uint32_t epilogueSubBlockIdx = EnableA8W4 ? 1U : 0U;
            if (GetSubBlockIdx() == epilogueSubBlockIdx) {
                AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
            }
        } else {
            while (AscendC::ReadGmByPassDCache(allDoneAddr) != allDoneTag) {
                int64_t startCycle = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - startCycle < 100) {
                }
            }
        }
    } else {
        EndSync(runtimeState.vecSetSyncCom);
    }
}

// 原型：MegaMoeWave::DispatchExpert / DispatchExpertsUntil。
// 推进 AIV1 Dispatch 游标，并确保每个新暴露的专家只预取一次。
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

// 原型：MegaMoeWave::ProcessGmm1Wave。
// AIC/AIV0 执行一个专家 wave，AIV1 继续专用于 Dispatch 和 Combine。
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
        Gmm1SwigluPipelineByMode<QuantOutType, WeightType, ActivationType, QuantScaleType,
                                        false, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                        IsGmm1Interleaved, true>(
            context, params, epilogueOp, gmmAddrInfo, state, runtimeState, expertIdx);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_SWIGLU_PIPELINE_H
