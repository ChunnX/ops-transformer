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
 * \brief Single-core GMM2 module extracted from mega_moe.h.
 *        All multi-core identification (GetBlockIdx, GetBlockNum, GetSubBlockIdx,
 *        blockIdx_, aivCoreIdx_, subBlockIdx_, blockAivNum_, blockNum_) replaced by TileInfo.
 */

#ifndef MEGA_MOE_GMM2_H
#define MEGA_MOE_GMM2_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace GMM2 {

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
class Gmm2Module {
public:
    using ActivationType = typename std::conditional<
        Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    __aicore__ inline Gmm2Module() {}

    __aicore__ inline bool UpdateGroupParams(
        MegaMoeModules::ExpertLoopState &state, uint32_t expertIdx,
        const MegaMoeModules::TileInfo &tile, const Params &params,
        GlobalTensor<int32_t> &expertRevNumsGlobalTensor)
    {
        if (expertIdx != 0) {
            uint64_t m = Get<M_VALUE>(state.problemShape);
            uint64_t n = Get<N_VALUE>(state.problemShape);
            uint64_t k = Get<K_VALUE>(state.problemShape);
            state.expertBeforeCnt += m;
            Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
            Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
            auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
            Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
            Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
            Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
                m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
            Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
            Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
                k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
            Get<IDX_M_OFFSET>(state.baseOffset) += m;
            Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
            Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
        }

        uint64_t offsetInCnt = expertIdx * INT32_PER_256B * static_cast<uint64_t>(params.tilingData->aicNum)
            + INT32_PER_256B * tile.tileIdx;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            expertRevNumsGlobalTensor[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor.GetValue(offsetInCnt);

        if (Get<M_VALUE>(state.problemShape) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void UpdateGlobalBuffer(
        GMMAddrInfo &gmmAddrInfo, const MegaMoeModules::ExpertLoopState &state,
        const Params &params, int32_t dispatchFlagSlotsPerExpert)
    {
        if constexpr (ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT) {
            gmmAddrInfo.gmm2OutGlobal =
                params.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal =
            params.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params.workspaceInfo.swigluQuantScalePtr +
            Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params.b2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)params.workspaceInfo.flagSwiGluToGmm2Ptr +
            Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
        gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)params.workspaceInfo.flagDispatchToGmm1Ptr +
            Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert;
    }

    __aicore__ inline void ResetGmm2CombineSyncCounters(
        const MegaMoeModules::TileInfo &tile, const Params &params,
        uint32_t expertPerRank, LocalTensor<int32_t> &resetTensor)
    {
        if constexpr(g_coreType == AIV) {
            int32_t totalCounters = static_cast<int32_t>(
                static_cast<int64_t>(expertPerRank) * tile.aivTotalTiles * INT_CACHELINE);
            int32_t tileLen, tileOffset;
            MegaMoeModules::TilingByTile(totalCounters, tileLen, tileOffset,
                tile.aivTileIdx, tile.aivTotalTiles);
            GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
            gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.gmm2CombineSyncCounterPtr);
            if (tileLen > 0) {
                Duplicate(resetTensor, 0, tileLen);
                SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
                DataCopy(gmm2CombineSyncCounterGm[tileOffset], resetTensor, tileLen);
            }
        }
    }

    __aicore__ inline void InitCombineBuffers(uint32_t k, int64_t &combineUbTensorSize)
    {
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
            uint32_t nAlign32 = Ops::Base::CeilAlign(k, static_cast<uint32_t>(ALIGN_32));
            uint32_t nScale = Ops::Base::CeilDiv(k, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k + nScale, static_cast<uint32_t>(ALIGN_32));
            uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
            combineUbTensorSize = (singleTokenBytes * 2) / sizeof(bfloat16_t);
        }
    }

    __aicore__ inline void ProcessCombine(
        const GMMAddrInfo &gmmAddrInfo, const MegaMoeModules::ExpertLoopState &gmm2State,
        uint32_t expertIdx, const MegaMoeModules::TileInfo &tile,
        const Params &params, uint32_t rankId, uint32_t k, int64_t combineUbTensorSize)
    {
        uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k, MegaMoeImpl::L1_TILE_N);

        GlobalTensor<int32_t> gmm2CombineSyncCounter;
        gmm2CombineSyncCounter.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.gmm2CombineSyncCounterPtr);

        uint32_t nScale = Ops::Base::CeilDiv(k, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k + nScale, static_cast<uint32_t>(ALIGN_32));

        uint32_t m_expert = Get<M_VALUE>(gmm2State.problemShape);
        uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(m_expert, MegaMoeImpl::L1_TILE_M_256);

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
        uint32_t tokenStart = myGroup * MegaMoeImpl::L1_TILE_M_256;
        uint32_t tokenCount = (MegaMoeImpl::L1_TILE_M_256 < m_expert - tokenStart)
            ? MegaMoeImpl::L1_TILE_M_256 : m_expert - tokenStart;
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
            tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr
                + (gmm2State.expertBeforeCnt + tokenStart + myTokenOffset) * TRIPLE_SIZE * sizeof(int32_t)));
            AscendC::DataCopy(tripleTensor, tripleGm, myTokenCount * TRIPLE_SIZE);
            PipeBarrier<PIPE_MTE2>();
            MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t>(
                tokenStart + myTokenOffset, myTokenCount, k, expertIdx, rankId,
                gmmAddrInfo.gmm2OutGlobal, params, tripleTensor, combineUbTensorSize,
                offset, quantTokenSizeBytes);
        }
    }

    __aicore__ inline void GroupMatmulWithCombine(
        const GMMAddrInfo &gmmAddrInfo, const MegaMoeModules::ExpertLoopState &state,
        uint32_t expertIdx, const MegaMoeModules::TileInfo &tile,
        const Params &params, uint32_t &startBlockIdx,
        int32_t &vecSetSyncCom, uint16_t &gmm2PingPongIdx,
        uint32_t rankId, uint32_t k, int64_t combineUbTensorSize)
    {
        if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
            MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode,
                SwigluQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                params, state.problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom,
                state.expertBeforeCnt, gmm2PingPongIdx, expertIdx);
        } else {
            if (params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
                MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                    QuantScaleOutType, QuantScaleOutType, true>(
                    params, state.problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom,
                    state.expertBeforeCnt, gmm2PingPongIdx, expertIdx);
            } else {
                MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                    QuantScaleOutType, QuantScaleOutType>(
                    params, state.problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom,
                    state.expertBeforeCnt, gmm2PingPongIdx, expertIdx);
            }
        }
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
            ProcessCombine(gmmAddrInfo, state, expertIdx, tile, params, rankId, k, combineUbTensorSize);
        }
    }
};

} // namespace GMM2

#endif
