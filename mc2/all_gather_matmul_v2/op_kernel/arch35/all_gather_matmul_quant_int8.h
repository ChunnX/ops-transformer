/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_quant_int8.h
 * \brief INT8 quant matmul with AllGather for arch35 (950PR)
 */

#ifndef __ALL_GATHER_MATMUL_QUANT_INT8_H__
#define __ALL_GATHER_MATMUL_QUANT_INT8_H__

#include "lib/hccl/hccl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../all_gather_matmul_aiv_mode.h"
#include "all_gather_matmul_tiling_arch35.h"

namespace AllGatherMatmulImpl {

using namespace AscendC;
using namespace AllGatherMatmulAIVModeImpl;

template <typename X1Type, typename X2Type, typename BiasType, typename ScaleType, typename YType, bool weightNZ,
          bool TA, bool TB>
class AllGatherMatmulQuantInt8
    : public AllGatherMatmulAIVMode<X1Type, X2Type, BiasType, ScaleType, YType, weightNZ, TA, TB> {
public:
    __aicore__ inline AllGatherMatmulQuantInt8()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleInv1, GM_ADDR scaleInv2, GM_ADDR biasGM,
                                GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR contextGM,
                                Mc2Tiling::AllGatherMatmulTilingDataInt8 *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling, TPipe *tPipe);
};

template <typename X1Type, typename X2Type, typename BiasType, typename ScaleType, typename YType, bool weightNZ,
          bool TA, bool TB>
__aicore__ inline void AllGatherMatmulQuantInt8<X1Type, X2Type, BiasType, ScaleType, YType, weightNZ, TA, TB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleInv1, GM_ADDR scaleInv2, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
    GM_ADDR workspaceGM, GM_ADDR contextGM, Mc2Tiling::AllGatherMatmulTilingDataInt8 *tilingData,
    __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling, TPipe *tPipe)
{
    auto &&info = tilingData->allGatherMatmulInfo;
    auto &&cocTiling = tilingData->cocTiling;

    constexpr bool quantFlag =
        (std::is_same<X1Type, int8_t>::value && std::is_same<X2Type, int8_t>::value) ||
        (std::is_same<X1Type, AscendC::int4b_t>::value && std::is_same<X2Type, AscendC::int4b_t>::value);
    using supportX1Type =
        typename std::conditional<std::is_same<X1Type, AscendC::int4b_t>::value, int8_t, X1Type>::type;
    using supportX2Type =
        typename std::conditional<std::is_same<X2Type, AscendC::int4b_t>::value, int8_t, X2Type>::type;

    this->aGM_ = aGM;
    this->bGM_ = bGM;
    this->cGM_ = cGM;
    this->allgatherGM_ = gatherOut;
    this->x1ScaleGM_ = scaleInv1;
    this->x2ScaleGM_ = scaleInv2;

    this->m = info.M;
    this->k = info.K;
    this->n = info.N;

    this->m0 = cocTiling.m0;
    this->k0 = cocTiling.k0;
    this->n0 = cocTiling.n0;

    this->m_loop = cocTiling.mLoop;
    this->n_loop = cocTiling.nLoop;
    this->k_loop = cocTiling.kLoop;
    this->pValue = cocTiling.pValue;

    this->max_ub_ping_pong_size = cocTiling.ubMoveNum / MAX_BLOCK_COUNT;
    this->comm_npu_split = cocTiling.commNpuSplit;
    this->comm_data_split = cocTiling.commDataSplit;
    this->comm_direct = cocTiling.commDirect;
    this->len_per_loop = cocTiling.lenPerLoop;
    this->swizzlCount = cocTiling.swizzlCount;
    this->swizzlDirect = cocTiling.swizzlDirect;
    this->core_count = this->comm_npu_split * this->comm_data_split;

    this->coreIdx = GetBlockIdx();
    this->coreNum = GetBlockNum();
    this->aivIdx = GetSubBlockIdx();
    this->blockIdx = this->coreIdx / GetTaskRation();

    this->aAlignSize = info.aAlignSize;
    this->bAlignSize = info.bAlignSize;
    this->hasAAlign = info.hasAAlign;
    this->hasBAlign = (info.hasBAlign && !(weightNZ));
    this->gm_a_align = reinterpret_cast<GM_ADDR>(this->hasAAlign ? workspaceGM : 0);
    this->gm_b_align = reinterpret_cast<GM_ADDR>(this->hasBAlign ? workspaceGM + this->aAlignSize : 0);
    this->gm_a_src = reinterpret_cast<__gm__ supportX1Type *>(this->hasAAlign ? this->gm_a_align : this->aGM_);
    this->gm_b_src = reinterpret_cast<__gm__ supportX2Type *>(this->hasBAlign ? this->gm_b_align : this->bGM_);
    this->gm_accum =
        reinterpret_cast<__gm__ int32_t *>(quantFlag ? workspaceGM + this->aAlignSize + this->bAlignSize : 0);

    this->m_align = Block512B<X1Type>::AlignUp(this->m);
    this->k_align = Block512B<X1Type>::AlignUp(this->k);
    this->n_align = Block512B<X1Type>::AlignUp(this->n);
    this->aligned_a = this->hasAAlign;
    this->aligned_b = this->hasBAlign;

    this->isX2ScaleTypeInt64 = info.isX2ScaleTypeInt64;
    this->dequantType = info.dequantType;
    this->needAivDequant = quantFlag && (this->dequantType == PER_TOKEN || std::is_same<YType, bfloat16_t>::value);
    bool needPerChannelA8W8 = quantFlag && !(this->isX2ScaleTypeInt64 && std::is_same<YType, float16_t>::value);
    this->needPerChannel = std::is_same_v<X1Type, AscendC::int4b_t> || needPerChannelA8W8;
    this->needPerToken = quantFlag && this->dequantType == PER_TOKEN;

    __gm__ HcclA5CombineOpParam *winContext_ = (__gm__ HcclA5CombineOpParam *)GetHcclContext<HCCL_GROUP_ID_0>();
    this->rankId = winContext_->rankId;
    this->worldSize = winContext_->rankDim;
    for (int i = 0; i < this->worldSize; i++) {
        this->stateAddrPerRank[i] = (GM_ADDR)winContext_->windowsIn[i];
    }

    constexpr int32_t FLAG_OFFSET_VAL = 180 * 1024 * 1024 / sizeof(int32_t);
    constexpr int32_t FLAG_OFFSET_BYTES = 180 * 1024 * 1024;

    uint64_t gm_scale_workspace_st =
        this->aAlignSize + this->bAlignSize + this->m * this->n * this->worldSize * sizeof(int32_t);
    this->gm_scale_workspace = this->needPerToken ? workspaceGM + gm_scale_workspace_st : 0;

    AllGatherMatmulAIVMode<X1Type, X2Type, BiasType, ScaleType, YType, weightNZ, TA, TB>::AICInit();

    AllGatherMatmulAIVMode<X1Type, X2Type, BiasType, ScaleType, YType, weightNZ, TA, TB>::AIVInit();
}

} // namespace AllGatherMatmulImpl

#endif // __ALL_GATHER_MATMUL_QUANT_INT8_H__