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
 * \file gmm1.h
 * \brief Single-core GMM1 module: UpdateGroupParams, UpdateGlobalBuffer,
 *        GroupMatmulWithSwigluQuant. All multi-core primitives (GetBlockIdx,
 *        GetBlockNum, GetSubBlockIdx, blockIdx_, aivCoreIdx_, subBlockIdx_,
 *        blockAivNum_, blockNum_) are replaced by TileInfo passed explicitly.
 */

#ifndef MEGA_MOE_ARCH35_SINGLE_CORE_GMM1_H
#define MEGA_MOE_ARCH35_SINGLE_CORE_GMM1_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"

namespace MegaMoeModules {
namespace GMM1 {

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
struct Gmm1 {
    using ActivationType = typename std::conditional<
        Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;

    template <AddrUpdateMode Mode>
    __aicore__ inline static bool UpdateGroupParams(
        ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt,
        const TileInfo &tile, uint32_t aicNum, const Params &params,
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

        if constexpr (Mode == AddrUpdateMode::kGmm1) {
            if (tile.subTileIdx == 0) {
                __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)params.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                    static_cast<uint64_t>(expertIdx) * aicNum * INT_CACHELINE +
                    static_cast<uint64_t>(tile.tileIdx) * INT_CACHELINE;
                while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }

                uint64_t offsetInCnt = expertIdx * 8 * aicNum + 8 * tile.tileIdx;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                    expertRevNumsGlobalTensor[offsetInCnt]);
                Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor.GetValue(offsetInCnt);
            } else {
                Get<M_VALUE>(state.problemShape) = sendCnt;
            }
        }

        if (Get<M_VALUE>(state.problemShape) == 0) {
            return false;
        }
        return true;
    }

    template <AddrUpdateMode Mode>
    __aicore__ inline static void UpdateGlobalBuffer(
        GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
        const Params &params, BlockEpilogue &epilogueOp,
        int32_t dispatchFlagSlotsPerExpert)
    {
        if constexpr (Mode == AddrUpdateMode::kGmm1) {
            if constexpr (ENABLE_A8W4) {
                gmmAddrInfo.gmm1OutGlobal =
                    params.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
            }
            gmmAddrInfo.aGlobal = params.workspaceInfo.dispatchRevDataPtr +
                Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
            gmmAddrInfo.aScaleGlobal = params.workspaceInfo.dispatchRevScalePtr +
                Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

            gmmAddrInfo.bGlobal = params.bGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
            gmmAddrInfo.bScaleGlobal = params.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) *
                                        sizeof(QuantScaleOutType);

            if constexpr(g_coreType == AIV) {
                Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                    Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                    Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
                epilogueOp.UpdateGlobalAddr(vecBaseOffset);
            }
        }

        gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)params.workspaceInfo.flagSwiGluToGmm2Ptr +
                                    Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
        gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)params.workspaceInfo.flagDispatchToGmm1Ptr +
                                    Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert;
    }

    __aicore__ inline static void GroupMatmulWithSwigluQuant(
        BlockEpilogue &epilogueOp, const Params &params,
        const TupleShape &problemShape, const GMMAddrInfo &gmmAddrInfo,
        uint32_t &startBlockIdx, int32_t &vecSetSyncCom)
    {
        if constexpr (ENABLE_A8W4) {
            MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
                QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom);
        } else {
            if (params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                MegaMoeImpl::GroupMatmulSwigluQuant<
                    QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>(
                    epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom);
            } else {
                MegaMoeImpl::GroupMatmulSwigluQuant<
                    QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                    epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom);
            }
        }
    }
};

} // namespace GMM1
} // namespace MegaMoeModules

#endif
