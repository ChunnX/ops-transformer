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
 * \brief Single-core GMM1 module: first grouped matmul + address update.
 */

#ifndef MEGA_MOE_GMM1_H
#define MEGA_MOE_GMM1_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"

using namespace AscendC;

namespace MegaMoeModules {
namespace GMM1 {

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
struct GMM1State {
    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp;

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    uint32_t startBlockIdx;
    int32_t vecSetSyncCom;
    uint32_t aicNum;
    Params params;
};

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline bool UpdateGroupParams(
    GMM1State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, ExpertLoopState &gmm1State, uint32_t expertIdx, uint64_t sendCnt,
    GlobalTensor<int32_t> &expertRevNumsGlobalTensor)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(gmm1State.problemShape);
        uint64_t n = Get<N_VALUE>(gmm1State.problemShape);
        uint64_t k = Get<K_VALUE>(gmm1State.problemShape);
        gmm1State.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(gmm1State.baseOffset) += m * k / state.A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(gmm1State.baseOffset) += n * k / state.B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(gmm1State.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(gmm1State.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(gmm1State.baseOffset) += m * n / SWIGLU_N_HALF / state.C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(gmm1State.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(gmm1State.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(gmm1State.baseOffset) += k * n / SWIGLU_N_HALF / state.B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(gmm1State.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(gmm1State.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(gmm1State.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(gmm1State.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(gmm1State.baseOffset) += m * k;
    }

    if (tile.subTileIdx == 0) {
        __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)state.params.workspaceInfo.flagSendCntCalToUpdParamsPtr +
            static_cast<uint64_t>(expertIdx) * state.aicNum * INT_CACHELINE +
            static_cast<uint64_t>(tile.tileIdx) * INT_CACHELINE;
        while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
        uint64_t offsetInCnt = expertIdx * 8 * state.aicNum + 8 * tile.tileIdx;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            expertRevNumsGlobalTensor[offsetInCnt]);
        Get<M_VALUE>(gmm1State.problemShape) = expertRevNumsGlobalTensor.GetValue(offsetInCnt);
    } else {
        Get<M_VALUE>(gmm1State.problemShape) = sendCnt;
    }

    if (Get<M_VALUE>(gmm1State.problemShape) == 0) {
        return false;
    }
    return true;
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void UpdateGlobalBuffer(
    GMM1State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm1State,
    int32_t dispatchFlagSlotsPerExpert)
{
    if constexpr (ENABLE_A8W4) {
        gmmAddrInfo.gmm1OutGlobal =
            state.params.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(gmm1State.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal = state.params.workspaceInfo.dispatchRevDataPtr +
        Get<IDX_A_OFFSET>(gmm1State.baseOffset) * sizeof(QuantOutType);
    gmmAddrInfo.aScaleGlobal = state.params.workspaceInfo.dispatchRevScalePtr +
        Get<IDX_A_SCALE_OFFSET>(gmm1State.baseOffset) * sizeof(QuantScaleOutType);

    gmmAddrInfo.bGlobal = state.params.bGmAddr + Get<IDX_B_OFFSET>(gmm1State.baseOffset) * sizeof(QuantOutType);
    gmmAddrInfo.bScaleGlobal = state.params.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(gmm1State.baseOffset) *
                                sizeof(QuantScaleOutType);

    if constexpr(g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(gmm1State.baseOffset), Get<IDX_C_SCALE_OFFSET>(gmm1State.baseOffset),
            Get<IDX_FLAG_OFFSET>(gmm1State.baseOffset), 0L, 0L, 0L};
        state.epilogueOp.UpdateGlobal_addr(vecBaseOffset);
    }

    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)state.params.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(gmm1State.baseOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)state.params.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(gmm1State.baseOffset) * dispatchFlagSlotsPerExpert;
}

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
__aicore__ inline void GroupMatmulWithSwigluQuant(
    GMM1State<QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType, QuantMode, ENABLE_A8W4, ENABLE_A4W4> &state,
    const TileInfo &tile, const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm1State)
{
    if constexpr (ENABLE_A8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
            QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            state.epilogueOp, state.params, gmm1State.problemShape, gmmAddrInfo,
            state.startBlockIdx, state.vecSetSyncCom);
    } else {
        if (state.params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
            state.params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>(
                state.epilogueOp, state.params, gmm1State.problemShape, gmmAddrInfo,
                state.startBlockIdx, state.vecSetSyncCom);
        } else {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                state.epilogueOp, state.params, gmm1State.problemShape, gmmAddrInfo,
                state.startBlockIdx, state.vecSetSyncCom);
        }
    }
}

} // namespace GMM1
} // namespace MegaMoeModules

#endif
