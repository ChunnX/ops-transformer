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
 * \file gmm1_op.h
 * \brief Single-core GMM1 operator: UpdateGroupParams, UpdateGlobalBuffer,
 *        GroupMatmulWithSwigluQuant. No blockIdx/blockNum/subBlockIdx inside;
 *        all core-identification via TileInfo passed from orchestrator.
 *
 * Interface constraint: each method called from process() has <= 5 parameters.
 * State is bundled into Gmm1Config (shared, set once) and Gmm1Context (per-loop).
 */

#ifndef MEGA_MOE_MODULES_GMM1_OP_H
#define MEGA_MOE_MODULES_GMM1_OP_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"

namespace MegaMoeModules {
namespace GMM1 {

template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
struct Gmm1Op {
    using ActivationType = typename std::conditional<
        Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;

    struct Config {
        Params params;
        uint32_t aicNum;
        int32_t dispatchFlagSlotsPerExpert;
        GlobalTensor<int32_t> expertRevNumsGlobalTensor;
        BlockEpilogue* epilogueOp;
    };

    struct Context {
        ExpertLoopState state;
        GMMAddrInfo addrInfo;
        uint32_t startBlockIdx = 0;
        int32_t vecSetSyncCom = 0;
    };

    __aicore__ inline static Context CreateContext(int64_t hiddenDim, uint32_t k)
    {
        TupleShape initShape;
        Get<N_VALUE>(initShape) = hiddenDim;
        Get<K_VALUE>(initShape) = k;
        BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        return Context{ExpertLoopState{initShape, initOffset, 0}, GMMAddrInfo{}, 0, 0};
    }

    __aicore__ inline static bool UpdateGroupParams(
        Context &ctx, uint32_t expertIdx, uint64_t sendCnt,
        const TileInfo &tile, const Config &cfg)
    {
        if (expertIdx != 0) {
            uint64_t m = Get<M_VALUE>(ctx.state.problemShape);
            uint64_t n = Get<N_VALUE>(ctx.state.problemShape);
            uint64_t k = Get<K_VALUE>(ctx.state.problemShape);
            ctx.state.expertBeforeCnt += m;
            Get<IDX_A_OFFSET>(ctx.state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
            Get<IDX_B_OFFSET>(ctx.state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
            auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            Get<IDX_A_SCALE_OFFSET>(ctx.state.baseOffset) += m * scaleK;
            Get<IDX_B_SCALE_OFFSET>(ctx.state.baseOffset) += n * scaleK;
            Get<IDX_C_OFFSET>(ctx.state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
            Get<IDX_C_SCALE_OFFSET>(ctx.state.baseOffset) +=
                m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            Get<IDX_FLAG_OFFSET>(ctx.state.baseOffset) += 1;
            Get<IDX_B2_OFFSET>(ctx.state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
            Get<IDX_B2_SCALE_OFFSET>(ctx.state.baseOffset) +=
                k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            Get<IDX_Y2_OFFSET>(ctx.state.baseOffset) += m * k;
            Get<IDX_M_OFFSET>(ctx.state.baseOffset) += m;
            Get<IDX_GMM1_OFFSET>(ctx.state.baseOffset) += m * n;
            Get<IDX_GMM2_OFFSET>(ctx.state.baseOffset) += m * k;
        }

        if (tile.subTileIdx == 0) {
            __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)cfg.params.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                static_cast<uint64_t>(expertIdx) * cfg.aicNum * INT_CACHELINE +
                static_cast<uint64_t>(tile.tileIdx) * INT_CACHELINE;
            while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {}
            }

            uint64_t offsetInCnt = expertIdx * INT32_PER_256B * cfg.aicNum + INT32_PER_256B * tile.tileIdx;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                cfg.expertRevNumsGlobalTensor[offsetInCnt]);
            Get<M_VALUE>(ctx.state.problemShape) = cfg.expertRevNumsGlobalTensor.GetValue(offsetInCnt);
        } else {
            Get<M_VALUE>(ctx.state.problemShape) = sendCnt;
        }

        if (Get<M_VALUE>(ctx.state.problemShape) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline static void UpdateGlobalBuffer(
        Context &ctx, const TileInfo &tile, const Config &cfg)
    {
        if constexpr (ENABLE_A8W4) {
            ctx.addrInfo.gmm1OutGlobal =
                cfg.params.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(ctx.state.baseOffset) * sizeof(bfloat16_t);
        }
        ctx.addrInfo.aGlobal = cfg.params.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(ctx.state.baseOffset) * sizeof(ActivationType);
        ctx.addrInfo.aScaleGlobal = cfg.params.workspaceInfo.dispatchRevScalePtr +
            Get<IDX_A_SCALE_OFFSET>(ctx.state.baseOffset) * sizeof(QuantScaleOutType);

        ctx.addrInfo.bGlobal = cfg.params.bGmAddr + Get<IDX_B_OFFSET>(ctx.state.baseOffset) * sizeof(ActivationType);
        ctx.addrInfo.bScaleGlobal = cfg.params.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(ctx.state.baseOffset) *
                                    sizeof(QuantScaleOutType);

        if constexpr(g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(ctx.state.baseOffset), Get<IDX_C_SCALE_OFFSET>(ctx.state.baseOffset),
                Get<IDX_FLAG_OFFSET>(ctx.state.baseOffset), 0L, 0L, 0L};
            cfg.epilogueOp->UpdateGlobalAddr(vecBaseOffset);
        }

        ctx.addrInfo.swigluToGmm2Flag = (__gm__ int32_t*)cfg.params.workspaceInfo.flagSwiGluToGmm2Ptr +
                                    Get<IDX_FLAG_OFFSET>(ctx.state.baseOffset) * INT_CACHELINE;
        ctx.addrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)cfg.params.workspaceInfo.flagDispatchToGmm1Ptr +
                                    Get<IDX_FLAG_OFFSET>(ctx.state.baseOffset) * cfg.dispatchFlagSlotsPerExpert;
    }

    __aicore__ inline static void Compute(Context &ctx, const Config &cfg)
    {
        if constexpr (ENABLE_A8W4) {
            MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
                QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                *cfg.epilogueOp, cfg.params, ctx.state.problemShape, ctx.addrInfo,
                ctx.startBlockIdx, ctx.vecSetSyncCom);
        } else {
            if (cfg.params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                cfg.params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                MegaMoeImpl::GroupMatmulSwigluQuant<
                    QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>(
                    *cfg.epilogueOp, cfg.params, ctx.state.problemShape, ctx.addrInfo,
                    ctx.startBlockIdx, ctx.vecSetSyncCom);
            } else {
                MegaMoeImpl::GroupMatmulSwigluQuant<
                    QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                    *cfg.epilogueOp, cfg.params, ctx.state.problemShape, ctx.addrInfo,
                    ctx.startBlockIdx, ctx.vecSetSyncCom);
            }
        }
    }
};

} // namespace GMM1
} // namespace MegaMoeModules

#endif
