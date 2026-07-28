/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_DISPATCH_GMM1_SWIGLU_H
#define MEGA_MOE_DISPATCH_GMM1_SWIGLU_H

#include "../aiv_comm/token_dispatch.h"

namespace MegaMoeImpl {

__aicore__ inline Gmm1ExpertLoopState CreateGmm1ExpertLoopState(const DispatchGmm1SwigluContext &context)
{
    Gmm1ProblemShape shape;
    Get<N_VALUE>(shape) = context.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.tokenHiddenDim;
    return Gmm1ExpertLoopState{shape, 0};
}

template <typename ActivationType, bool EnableA8W4>
__aicore__ inline bool UpdateGmm1GroupParams(const DispatchGmm1SwigluContext &context,
                                             const DispatchGmm1SwigluArgs &args,
                                             DispatchGmm1SwigluScratch<ActivationType> &scratch,
                                             Gmm1ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt)
{
    // Generic AIV1 only dispatches and advances startBlockIdx. Keep M for the
    // tile count, but avoid maintaining address offsets that this role never consumes.
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
__aicore__ inline void UpdateGmm1GlobalBuffer(const DispatchGmm1SwigluContext &context,
                                              const DispatchGmm1SwigluArgs &args, BlockEpilogue &epilogueOp,
                                              GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state,
                                              uint32_t expertIdx)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t aElemsPerByte = Std::IsSame<ActivationType, uint8_t>::value ? 2U : 1U;
    constexpr uint32_t bElemsPerByte = Std::IsSame<WeightType, fp4x2_e2m1_t>::value ? 2U : 1U;
    constexpr uint32_t cElemsPerByte = Std::IsSame<SwigluOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = expertIdx;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResPtr + state.rowOffset * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal = args.dispatchRevDataPtr + state.rowOffset * k / aElemsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = args.dispatchRevScalePtr + state.rowOffset * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = args.weight1Ptr + expertOffset * n * k / bElemsPerByte * sizeof(ActivationType);
    gmmAddrInfo.bScaleGlobal = args.weight1ScalePtr + expertOffset * n * scaleK * sizeof(QuantScaleType);
    if constexpr (g_coreType == AIV) {
        bool runsSwiglu = true;
        if constexpr (EnableA8W4) {
            runsSwiglu = GetSubBlockIdx() == 1;
        }
        if (runsSwiglu) {
            int64_t scaleN =
                Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                state.rowOffset * n / SWIGLU_N_HALF / cElemsPerByte,
                state.rowOffset * scaleN,
                expertOffset,
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
}

template <typename QuantOutType, typename WeightType, typename SwigluOutType, typename QuantScaleType, bool EnableA8W4,
          uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void RunGmm1SwigluStageOneCore(const DispatchGmm1SwigluContext &context,
                                                 const DispatchGmm1SwigluArgs &args, const Params &gmmParams,
                                                 BlockEpilogue &epilogueOp, const GMMAddrInfo &gmmAddrInfo,
                                                 const Gmm1ExpertLoopState &state,
                                                 DispatchGmm1SwigluState &runtimeState, uint32_t expertIdx)
{
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType,
                                                Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, false>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, true, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, false>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, false, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                            false>(epilogueOp, gmmParams, state.problemShape, gmmAddrInfo,
                                                   runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
                                                   context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
    }
}

// Every physical core in one MIX_AIC_1_2 block calls this function with the
// same logical blockJob and physical countWorkspace contexts. The function preserves
// dispatch(e + 1) prefetch and GMM1/Swiglu(e) consumption order.
template <typename QuantOutType, typename ActivationType, typename WeightType, typename SwigluOutType,
          typename QuantScaleType, bool EnableA8W4, uint32_t Gmm1TileM, uint32_t EpilogueTileM,
          bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void DispatchGmm1SwigluOneCore(const DispatchGmm1SwigluContext &context,
                                                 const DispatchGmm1SwigluArgs &args, const Params &gmmParams,
                                                 BlockEpilogue &epilogueOp,
                                                 DispatchGmm1SwigluScratch<ActivationType> &scratch,
                                                 DispatchGmm1SwigluState &runtimeState)
{
    if (context.blockJob.totalJobs == 0U || context.blockJob.jobIndex >= context.blockJob.totalJobs ||
        context.countWorkspace.blockNum == 0U || context.countWorkspace.blockIdx >= context.countWorkspace.blockNum ||
        context.expertPerRank == 0U || context.blockNumPerRank == 0U || context.bufferConfig.routeItemsPerBatch <= 0 ||
        context.bufferConfig.routeBatchCount <= 0 || context.bufferConfig.bufferCount <= 0) {
        return;
    }
    InitDispatchGmm1SwigluBuffers<ActivationType, EnableA8W4>(context, args, scratch);
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(context);
    GMMAddrInfo gmm1AddrInfo;
    uint64_t currentSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1) {
            SendCntCalOneCore<ActivationType, EnableA8W4, TopkWeightsPrefetch>(context, args, scratch, 0, nextSendCnt);
            if (nextSendCnt != 0) {
                DispatchTokensOneCore<ActivationType, QuantScaleType, EnableA8W4, Gmm1TileM, TopkWeightsPrefetch>(
                    context, args, scratch, 0);
            }
        }
    }

    for (uint32_t expertIdx = 0; expertIdx < context.expertPerRank; ++expertIdx) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1) {
                currentSendCnt = nextSendCnt;
                if (expertIdx + 1 < context.expertPerRank) {
                    SendCntCalOneCore<ActivationType, EnableA8W4, TopkWeightsPrefetch>(context, args, scratch,
                                                                                       expertIdx + 1, nextSendCnt);
                    if (nextSendCnt != 0) {
                        DispatchTokensOneCore<ActivationType, QuantScaleType, EnableA8W4, Gmm1TileM,
                                              TopkWeightsPrefetch>(context, args, scratch, expertIdx + 1);
                    }
                }
            }
        }

        if (!UpdateGmm1GroupParams<ActivationType, EnableA8W4>(context, args, scratch, gmm1State, expertIdx,
                                                               currentSendCnt)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                               TopkWeightsPrefetch>(context, args, epilogueOp, gmm1AddrInfo, gmm1State, expertIdx);
        RunGmm1SwigluStageOneCore<QuantOutType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4, Gmm1TileM,
                                  EpilogueTileM, TopkWeightsPrefetch>(context, args, gmmParams, epilogueOp,
                                                                      gmm1AddrInfo, gmm1State, runtimeState, expertIdx);
    }
    if constexpr (TopkWeightsPrefetch) {
        int32_t allDoneTag = static_cast<int32_t>(context.expertCapacityPerRank + 1U);
        __gm__ int32_t *allDoneAddr = reinterpret_cast<__gm__ int32_t *>(args.gmm1TileStatusPtr) +
                                      static_cast<uint64_t>(context.expertCapacityPerRank) * context.maxTilesPerExpert;
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

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DISPATCH_GMM1_SWIGLU_H
