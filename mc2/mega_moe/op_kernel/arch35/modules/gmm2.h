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
 * \file gmm2.h
 * \brief Single-core GMM2 module: second grouped matmul + address update + combine dispatch.
 */

#ifndef MEGA_MOE_GMM2_H
#define MEGA_MOE_GMM2_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace MegaMoeModules {
namespace GMM2 {

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
struct GMM2State {
    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    uint32_t startBlockIdx;
    int32_t vecSetSyncCom;
    uint16_t gmm2PingPongIdx;
    uint32_t aicNum;
    int64_t combineUbTensorSize;
    int32_t dispatchFlagSlotsPerExpert;
    Params params;
};

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline bool UpdateGroupParams(
    GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, ExpertLoopState &gmm2State, uint32_t expertIdx,
    GlobalTensor<int32_t> &expertRevNumsGlobalTensor)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(gmm2State.problemShape);
        uint64_t n = Get<N_VALUE>(gmm2State.problemShape);
        uint64_t k = Get<K_VALUE>(gmm2State.problemShape);
        gmm2State.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(gmm2State.baseOffset) += m * k / state.A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(gmm2State.baseOffset) += n * k / state.B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(gmm2State.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(gmm2State.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(gmm2State.baseOffset) += m * n / SWIGLU_N_HALF / state.C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(gmm2State.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(gmm2State.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(gmm2State.baseOffset) += k * n / SWIGLU_N_HALF / state.B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(gmm2State.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(gmm2State.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(gmm2State.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(gmm2State.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(gmm2State.baseOffset) += m * k;
    }

    uint64_t offsetInCnt = expertIdx * 8 * state.aicNum + 8 * tile.tileIdx;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            expertRevNumsGlobalTensor[offsetInCnt]);
    Get<M_VALUE>(gmm2State.problemShape) = expertRevNumsGlobalTensor.GetValue(offsetInCnt);

    if (Get<M_VALUE>(gmm2State.problemShape) == 0) {
        return false;
    }
    return true;
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void UpdateGlobalBuffer(
    GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT) {
        gmmAddrInfo.gmm2OutGlobal =
            state.params.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(gmm2State.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal =
        state.params.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(gmm2State.baseOffset) * sizeof(QuantOutType);
    gmmAddrInfo.aScaleGlobal = state.params.workspaceInfo.swigluQuantScalePtr +
                               Get<IDX_C_SCALE_OFFSET>(gmm2State.baseOffset) * sizeof(QuantScaleOutType);
    gmmAddrInfo.bGlobal = state.params.b2GmAddr + Get<IDX_B2_OFFSET>(gmm2State.baseOffset) * sizeof(QuantOutType);
    gmmAddrInfo.bScaleGlobal =
        state.params.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(gmm2State.baseOffset) * sizeof(QuantScaleOutType);

    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)state.params.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(gmm2State.baseOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)state.params.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(gmm2State.baseOffset) * state.dispatchFlagSlotsPerExpert;
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void InitCombineBuffers(
    GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, uint32_t k)
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        state.combineUbTensorSize = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void GroupMatmulWithCombine(
    GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
    uint32_t expertIdx)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode,
            SwigluQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            state.params, gmm2State.problemShape, gmmAddrInfo, state.startBlockIdx, state.vecSetSyncCom,
            gmm2State.expertBeforeCnt, state.gmm2PingPongIdx, expertIdx);
    } else {
        if (state.params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType, true>(
                state.params, gmm2State.problemShape, gmmAddrInfo, state.startBlockIdx, state.vecSetSyncCom,
                gmm2State.expertBeforeCnt, state.gmm2PingPongIdx, expertIdx);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType>(
                state.params, gmm2State.problemShape, gmmAddrInfo, state.startBlockIdx, state.vecSetSyncCom,
                gmm2State.expertBeforeCnt, state.gmm2PingPongIdx, expertIdx);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        ProcessCombine(state, tile, gmmAddrInfo, gmm2State, expertIdx);
    }
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void ProcessCombine(
    GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
    uint32_t expertIdx)
{
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(state.params.tilingData->h, L1_TILE_N);

    GlobalTensor<int32_t> gmm2CombineSyncCounter;
    gmm2CombineSyncCounter.SetGlobalBuffer((__gm__ int32_t*)state.params.workspaceInfo.gmm2CombineSyncCounterPtr);

    uint32_t nScale = Ops::Base::CeilDiv(state.params.tilingData->h, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(state.params.tilingData->h + nScale, static_cast<uint32_t>(ALIGN_32));

    uint32_t m_expert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(m_expert, L1_TILE_M_256);

    uint32_t coreIdForGrouping = tile.aivTileIdx;
    uint32_t totalCoresForGrouping = tile.aivTotalTiles;
    if constexpr (ENABLE_A8W4) {
        if (tile.subTileIdx != 1) {
            return;
        }
        coreIdForGrouping = tile.aivTileIdx / 2;
        totalCoresForGrouping = tile.aivTotalTiles / 2;
    }

    uint32_t myGroup, myIdxInGrp, myGrpSize;
    MegaMoeImpl::ComputeCoreGrouping(coreIdForGrouping, tokenGroupsThisExpert, totalCoresForGrouping,
        myGroup, myIdxInGrp, myGrpSize);

    if (myGroup >= tokenGroupsThisExpert) {
        return;
    }

    __gm__ int32_t* myCounterAddr = (__gm__ int32_t*)gmm2CombineSyncCounter.GetPhyAddr()
        + expertIdx * tile.aivTotalTiles * INT_CACHELINE
        + tile.aivTileIdx * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(myCounterAddr) != nTilesPerGroup)
    {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) {
        };
    }
    uint32_t tokenStart = myGroup * L1_TILE_M_256;
    uint32_t tokenCount = (L1_TILE_M_256 < m_expert - tokenStart) ? L1_TILE_M_256 : m_expert - tokenStart;
    uint32_t tokensPerCore = Ops::Base::CeilDiv(tokenCount, myGrpSize);
    int32_t myTokenOffset = myIdxInGrp * tokensPerCore;
    int32_t myTokenCount = 0;
    if (myTokenOffset < (int32_t)tokenCount) {
        myTokenCount = (tokensPerCore < tokenCount - myTokenOffset) ? tokensPerCore : tokenCount - myTokenOffset;
    }
    if (myTokenCount > 0) {
        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, myTokenCount * TRIPLE_SIZE);
        offset += myTokenCount * TRIPLE_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> tripleGm;
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(state.params.workspaceInfo.tripleInfoPtr
            + (gmm2State.expertBeforeCnt + tokenStart + myTokenOffset) * TRIPLE_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(tripleTensor, tripleGm, myTokenCount * TRIPLE_SIZE);
        PipeBarrier<PIPE_MTE2>();
        MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t>(
            tokenStart + myTokenOffset, myTokenCount, state.params.tilingData->h, expertIdx,
            state.params.combineCommParams.rankId,
            gmmAddrInfo.gmm2OutGlobal, state.params, tripleTensor, state.combineUbTensorSize,
            offset, quantTokenSizeBytes);
    }
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void ResetGmm2CombineSyncCounters(
    GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, LocalTensor<int32_t> &resetTensor)
{
    if constexpr(g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(
            static_cast<int64_t>(state.params.tilingData->expertPerRank) * tile.aivTotalTiles * INT_CACHELINE);
        int32_t coreLen, coreOffset;
        TilingByTile(totalCounters, coreLen, coreOffset, tile.aivTileIdx, tile.aivTotalTiles);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)state.params.workspaceInfo.gmm2CombineSyncCounterPtr);
        if (coreLen > 0) {
            Duplicate(resetTensor, 0, coreLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(gmm2CombineSyncCounterGm[coreOffset], resetTensor, coreLen);
        }
    }
}

} // namespace GMM2
} // namespace MegaMoeModules

#endif
