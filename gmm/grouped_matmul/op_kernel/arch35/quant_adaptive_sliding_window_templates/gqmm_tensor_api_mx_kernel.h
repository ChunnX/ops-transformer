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


#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;
using QuantBasicApiMMTiling = GroupedMatmulTilingData::QuantBasicApiMMTiling;

namespace {
__aicore__ inline bool CanEnableThreeL1Buffer(const QuantBasicApiMMTiling* mmTilingDataIn)
{
    if (mmTilingDataIn->kAL1 == 0U || mmTilingDataIn->kBL1 == 0U || mmTilingDataIn->scaleKAL1 == 0U) {
        return false;
    }
    constexpr uint64_t halfL1Size = 256UL * 1024UL;
    uint64_t aL1Size = static_cast<uint64_t>(mmTilingDataIn->baseM) *
                       Blaze::Gemm::CeilAlign(static_cast<uint64_t>(mmTilingDataIn->kAL1), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    uint64_t bL1Size = static_cast<uint64_t>(mmTilingDataIn->baseN) *
                       Blaze::Gemm::CeilAlign(static_cast<uint64_t>(mmTilingDataIn->kBL1), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    uint64_t scaleKSize = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(mmTilingDataIn->scaleKAL1),
                                               Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    uint64_t scaleL1Size =
        (static_cast<uint64_t>(mmTilingDataIn->baseM) + static_cast<uint64_t>(mmTilingDataIn->baseN)) * scaleKSize;
    uint64_t biasL1Size = mmTilingDataIn->isBias == 1 ? static_cast<uint64_t>(mmTilingDataIn->baseN) * sizeof(float) : 0UL;
    uint64_t commonL1Size = aL1Size + bL1Size + scaleL1Size + biasL1Size;
    return (commonL1Size + aL1Size <= halfL1Size) && (commonL1Size + bL1Size <= halfL1Size);
}
} // namespace

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmTensorApiMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                            GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                            const GMMQuantParams* gmmBaseParamsIn,
                                            const QuantBasicApiMMTiling* mmTilingDataIn, AscendC::TPipe* que)
{
    (void)workspace;
    (void)que;
    if ASCEND_IS_AIV {
        return;
    }
    using AType = xType;
    using BType = wType;
    using BiasType = float;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = Blaze::Gemm::MatmulShape;
    if (CanEnableThreeL1Buffer(mmTilingDataIn)) {
        using BlockMmadPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false, Blaze::Gemm::TRIPLE_BUFFER_COUNT>;
        using QgmmBlockMmad =
            Blaze::Gemm::Block::BlockMmadQGmmMx<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC,
                                                BiasType, LayoutC>;
        using BlockEpilogue = void;
        using QgmmKernel = Blaze::Gemm::Kernel::KernelQGmmMx<ProblemShape, QgmmBlockMmad, BlockEpilogue>;
        using Params = typename QgmmKernel::Params;
        using GMMTiling = typename QgmmKernel::GMMTiling;

        GMMTiling gmmParams{static_cast<uint32_t>(gmmBaseParamsIn->groupNum),
                            static_cast<uint32_t>(mmTilingDataIn->m),
                            static_cast<uint32_t>(mmTilingDataIn->n),
                            static_cast<uint32_t>(mmTilingDataIn->k),
                            static_cast<uint32_t>(mmTilingDataIn->baseM),
                            static_cast<uint32_t>(mmTilingDataIn->baseN),
                            static_cast<uint32_t>(mmTilingDataIn->baseK),
                            static_cast<uint32_t>(mmTilingDataIn->kAL1),
                            static_cast<uint32_t>(mmTilingDataIn->kBL1),
                            static_cast<uint32_t>(mmTilingDataIn->scaleKAL1),
                            static_cast<uint32_t>(mmTilingDataIn->scaleKBL1),
                            static_cast<uint8_t>(mmTilingDataIn->isBias),
                            static_cast<uint8_t>(mmTilingDataIn->dbL0C),
                            static_cast<int8_t>(gmmBaseParamsIn->groupType),
                            static_cast<uint8_t>(gmmBaseParamsIn->groupListType)};

        Params params = {
            {static_cast<int64_t>(gmmParams.m), static_cast<int64_t>(gmmParams.n), static_cast<int64_t>(gmmParams.k),
             static_cast<int64_t>(1)},
            {reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AType>(0, x)),
             reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BType>(0, weight)),
             reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y)),
             mmTilingDataIn->isBias == 1 ? reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias))
                                         : nullptr,
             perTokenScale,
             scale == nullptr ? nullptr
                              : reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AscendC::fp8_e8m0_t>(0, scale))},
            groupList,
            gmmParams};

        QgmmKernel gmm;
        gmm(params);
    } else {
        using BlockMmadPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false, Blaze::Gemm::DOUBLE_BUFFER_COUNT>;
        using QgmmBlockMmad =
            Blaze::Gemm::Block::BlockMmadQGmmMx<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC,
                                                BiasType, LayoutC>;
        using BlockEpilogue = void;
        using QgmmKernel = Blaze::Gemm::Kernel::KernelQGmmMx<ProblemShape, QgmmBlockMmad, BlockEpilogue>;
        using Params = typename QgmmKernel::Params;
        using GMMTiling = typename QgmmKernel::GMMTiling;

        GMMTiling gmmParams{static_cast<uint32_t>(gmmBaseParamsIn->groupNum),
                            static_cast<uint32_t>(mmTilingDataIn->m),
                            static_cast<uint32_t>(mmTilingDataIn->n),
                            static_cast<uint32_t>(mmTilingDataIn->k),
                            static_cast<uint32_t>(mmTilingDataIn->baseM),
                            static_cast<uint32_t>(mmTilingDataIn->baseN),
                            static_cast<uint32_t>(mmTilingDataIn->baseK),
                            static_cast<uint32_t>(mmTilingDataIn->kAL1),
                            static_cast<uint32_t>(mmTilingDataIn->kBL1),
                            static_cast<uint32_t>(mmTilingDataIn->scaleKAL1),
                            static_cast<uint32_t>(mmTilingDataIn->scaleKBL1),
                            static_cast<uint8_t>(mmTilingDataIn->isBias),
                            static_cast<uint8_t>(mmTilingDataIn->dbL0C),
                            static_cast<int8_t>(gmmBaseParamsIn->groupType),
                            static_cast<uint8_t>(gmmBaseParamsIn->groupListType)};

        Params params = {
            {static_cast<int64_t>(gmmParams.m), static_cast<int64_t>(gmmParams.n), static_cast<int64_t>(gmmParams.k),
             static_cast<int64_t>(1)},
            {reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AType>(0, x)),
             reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BType>(0, weight)),
             reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y)),
             mmTilingDataIn->isBias == 1 ? reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias))
                                         : nullptr,
             perTokenScale,
             scale == nullptr ? nullptr
                              : reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AscendC::fp8_e8m0_t>(0, scale))},
            groupList,
            gmmParams};

        QgmmKernel gmm;
        gmm(params);
    }
}

#endif
