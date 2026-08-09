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
 * \file gqmm_tensor_api_cube_kernel.h
 * \brief Grouped quantized matmul cube kernel adapter (A8W8 fixpipe dequant, Tensor API)
 */

#ifndef GQMM_TENSOR_API_CUBE_KERNEL_H
#define GQMM_TENSOR_API_CUBE_KERNEL_H

#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/kernel/kernel_gmm_cube_quant.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmTensorApiCubeKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                              GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                              const GMMQuantParams *gmmBaseParamsIn,
                                              const TCubeTiling *mmTilingDataIn, AscendC::TPipe *que)
{
    (void)workspace;
    (void)que;
    if ASCEND_IS_AIV {
        return;
    }
    using AType = xType;
    // BType must be a tuple<weightType, scaleType>: block_mmad_a8w8_fixpipe_quant.h reads
    // BType = tuple_element<0>, X2ScaleType = tuple_element<1>.
    using BType = AscendC::Std::tuple<wType, scaleType>;
    using BiasType = biasType;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    // ScheduleType_ = KernelGroupedMmadFixpipeQuant routes GemmUniversal to kernel_gmm_cube_quant.h.
    using BlockMmadPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
    using GmmBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC,
                                                       BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, GmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    // l1BufferNum is the L1 slot count (2 or 4), NOT tiling's depthA1 (= stepKa * DB_SIZE).
    // GMM quant tiling always emits depthA1 = 2 * stepKa (see CalStepKs), i.e. exactly 2 slots per K window,
    // and only l1BufferNum == 2 preserves independent kAL1 / kBL1 in block_mmad_a8w8_fixpipe_quant.h.
    constexpr uint32_t GMM_CUBE_L1_BUFFER_NUM = 2U;

    GMMTiling gmmParams{static_cast<uint32_t>(gmmBaseParamsIn->groupNum),
                        static_cast<int64_t>(mmTilingDataIn->M),
                        static_cast<int64_t>(mmTilingDataIn->N),
                        static_cast<int64_t>(mmTilingDataIn->Ka),
                        static_cast<uint32_t>(mmTilingDataIn->baseM),
                        static_cast<uint32_t>(mmTilingDataIn->baseN),
                        static_cast<uint32_t>(mmTilingDataIn->baseK),
                        static_cast<uint32_t>(mmTilingDataIn->stepKa * mmTilingDataIn->baseK),
                        static_cast<uint32_t>(mmTilingDataIn->stepKb * mmTilingDataIn->baseK),
                        GMM_CUBE_L1_BUFFER_NUM,
                        static_cast<uint32_t>(gmmBaseParamsIn->aQuantMode),
                        static_cast<uint32_t>(gmmBaseParamsIn->bQuantMode),
                        static_cast<uint8_t>(mmTilingDataIn->isBias),
                        static_cast<uint8_t>(mmTilingDataIn->dbL0C),
                        static_cast<int8_t>(gmmBaseParamsIn->groupType),
                        static_cast<uint8_t>(gmmBaseParamsIn->groupListType),
                        static_cast<uint8_t>(gmmBaseParamsIn->singleW),
                        static_cast<uint8_t>(gmmBaseParamsIn->singleX),
                        static_cast<uint8_t>(gmmBaseParamsIn->singleY),
                        1U,
                        1U};

    GM_ADDR aDataAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<xType>(0, x));
    GM_ADDR bDataAddr = gmmBaseParamsIn->singleW == 1 ?
                            reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<wType>(0, weight)) :
                            weight;
    GM_ADDR yDataAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    GM_ADDR biasDataAddr = mmTilingDataIn->isBias == 1 ?
                               reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias)) :
                               nullptr;
    GM_ADDR perTokenScaleDataAddr = perTokenScale;
    GM_ADDR scaleDataAddr = scale;
    GM_ADDR groupListDataAddr = groupList;

    Params params = {{gmmParams.m, gmmParams.n, gmmParams.k, static_cast<int64_t>(1)},
                     {aDataAddr, bDataAddr, yDataAddr, biasDataAddr, perTokenScaleDataAddr, scaleDataAddr},
                     {},
                     groupListDataAddr,
                     gmmParams};

    GmmKernel gmm;
    gmm(params);
}

#endif
