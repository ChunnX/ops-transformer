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
 * \file gqmm_tensor_api_mx_kernel.h
 * \brief
 */

#ifndef GQMM_TENSOR_API_MX_KERNEL_H
#define GQMM_TENSOR_API_MX_KERNEL_H

#include "blaze/block/block_mmad_mx.h"
#include "blaze/kernel/kernel_qgmm_mx.h"
#include "blaze/policy/dispatch_policy.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;
using QuantBasicApiMMTiling = GroupedMatmulTilingData::QuantBasicApiMMTiling;

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmTensorApiMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                            GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                            const GMMQuantParams* gmmBaseParamsIn,
                                            const QuantBasicApiMMTiling* mmTilingDataIn, AscendC::TPipe* que)
{
    (void)workspace;
    (void)que;
    using AType = xType;
    using BType = wType;
    using BiasType = float;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = Blaze::Gemm::MatmulShape;
    using BlockMmadPolicy = Blaze::Gemm::MatmulWithScaleMx<0>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = void;
    using GmmKernel = Blaze::Gemm::Kernel::KernelQGmmMx<ProblemShape, BlockMmad, BlockEpilogue>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmParams{gmmBaseParamsIn->groupNum, mmTilingDataIn->m,          mmTilingDataIn->n,
                        mmTilingDataIn->k,         mmTilingDataIn->baseM,      mmTilingDataIn->baseN,
                        mmTilingDataIn->baseK,     mmTilingDataIn->kAL1,       mmTilingDataIn->kBL1,
                        mmTilingDataIn->scaleKAL1, mmTilingDataIn->scaleKBL1,  mmTilingDataIn->isBias,
                        mmTilingDataIn->dbL0C,     gmmBaseParamsIn->groupType, gmmBaseParamsIn->groupListType};

    Params params = {
        {1, 1, 1, 1},
        {reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AType>(0, x)),
         reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BType>(0, weight)),
         reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y)),
         mmTilingDataIn->isBias == 1 ? reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias))
                                     : nullptr,
         reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AscendC::fp8_e8m0_t>(0, perTokenScale)),
         reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AscendC::fp8_e8m0_t>(0, scale))},
        groupList == nullptr ? nullptr : reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int64_t>(0, groupList)),
        gmmParams};

    GmmKernel gmm;
    gmm(params);
}

#endif
