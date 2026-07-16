/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PRODUCT OR SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_mxquant_tensor_api.h
 * \brief TensorApi entry point for GroupedMatmulSwigluQuantV2 MXFP8 ND scenario.
 *        Assembles Blaze BlockMmad + Blaze Epilogue + Blaze Scheduler + MIX Kernel.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_TENSOR_API_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_TENSOR_API_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/gemm/block/block_mmad_swiglu_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_gmm_swiglu_mx_tensor_api.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant_te.h"
#include "../grouped_matmul_swiglu_quant_v2_utils.h"

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluAswtTensorApi(
    GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
    GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
    GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)weightAssistanceMatrix;
    (void)smoothScale;
    (void)workspace;

    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, mmTilingData, mmTilingData_, tiling);

    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using C1Type = float;
    using BiasType = float;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using weightscaleType = AscendC::fp8_e8m0_t;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockMmadPolicy = Blaze::Gemm::GroupedMatmulSwiGluMixMx<0>;
    using SwigluBlockMmad = Blaze::Gemm::Block::BlockMmad<
        BlockMmadPolicy, AType, LayoutA, BType, LayoutB, C1Type, LayoutC, BiasType, LayoutC>;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuantTe<
        CType, C1Type, weightscaleType>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

    using SwigluKernel = Blaze::Gemm::Kernel::KernelGmmSwiGluMixMx<
        ProblemShape, SwigluBlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename SwigluKernel::Params;
    using GMMTiling = typename SwigluKernel::GMMTiling;

    const uint8_t singleW = (gmmSwigluQuantParams_.isMxWeightNzMultiTensor == 0) ? 1 : 0;

    uint64_t scaleFactorA = matmulTiling->mxTypePara & MX_SCALE_FACTOR_MASK;
    uint64_t scaleFactorB = (matmulTiling->mxTypePara >> MX_SCALE_FACTOR_BIT) & MX_SCALE_FACTOR_MASK;
    scaleFactorA = scaleFactorA == 0 ? MX_SCALE_FACTOR_DEFAULT : scaleFactorA;
    scaleFactorB = scaleFactorB == 0 ? MX_SCALE_FACTOR_DEFAULT : scaleFactorB;
    uint64_t kAL1 = static_cast<uint64_t>(matmulTiling->stepKa) * static_cast<uint64_t>(matmulTiling->baseK);
    uint64_t kBL1 = static_cast<uint64_t>(matmulTiling->stepKb) * static_cast<uint64_t>(matmulTiling->baseK);
    uint64_t scaleKL1 = Cgmct::Gemm::Min(Cgmct::Gemm::Max(scaleFactorA * kAL1, scaleFactorB * kBL1),
                                         static_cast<uint64_t>(matmulTiling->Ka));

    GMMTiling gmmParams{static_cast<uint32_t>(gmmSwigluQuantParams_.groupNum),
                        static_cast<int64_t>(mmTilingData_.M),                                    // m (per-group from groupList)
                        static_cast<int64_t>(mmTilingData_.N),                      // n
                        static_cast<int64_t>(mmTilingData_.Ka),                     // k
                        static_cast<uint32_t>(mmTilingData_.baseM),
                        static_cast<uint32_t>(mmTilingData_.baseN),
                        static_cast<uint32_t>(mmTilingData_.baseK),
                        static_cast<uint32_t>(mmTilingData_.kAL1),
                        static_cast<uint32_t>(mmTilingData_.kBL1),
                        static_cast<uint32_t>(mmTilingData_.scaleKL1),
                        static_cast<uint32_t>(mmTilingData_.scaleKL1),
                        static_cast<uint8_t>(mmTilingData_.dbL0C),
                        static_cast<int8_t>(0),                                     // groupType
                        static_cast<uint8_t>(gmmSwigluQuantParams_.groupListType),
                        singleW};

    GM_ADDR aDataAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL_SWIGLU_QUANT::GetTensorAddr<AType>(0, x));
    GM_ADDR scaleADataAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL_SWIGLU_QUANT::GetTensorAddr<AscendC::fp8_e8m0_t>(0, xScale));
    GM_ADDR bDataAddr = singleW == 1 ?
        reinterpret_cast<GM_ADDR>(GROUPED_MATMUL_SWIGLU_QUANT::GetTensorAddr<BType>(0, weight)) :
        weight;
    GM_ADDR scaleBDataAddr = singleW == 1 ?
        reinterpret_cast<GM_ADDR>(
            GROUPED_MATMUL_SWIGLU_QUANT::GetTensorAddr<AscendC::fp8_e8m0_t>(0, weightScale)) :
        weightScale;
    GM_ADDR yDataAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL_SWIGLU_QUANT::GetTensorAddr<CType>(0, y));
    GM_ADDR yScaleDataAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL_SWIGLU_QUANT::GetTensorAddr<int8_t>(0, yScale));
    GM_ADDR groupListDataAddr = groupList;

    Params params = {{gmmParams.m, gmmParams.n, gmmParams.k, static_cast<int64_t>(1)},
                     {aDataAddr, bDataAddr, nullptr, nullptr, scaleADataAddr, scaleBDataAddr},
                     {yDataAddr, yScaleDataAddr, gmmParams.baseM, gmmParams.baseN},
                     groupListDataAddr,
                     gmmParams};

    SwigluKernel op;
    op(params);
}

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_TENSOR_API_H
