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
 * \file grouped_matmul_swiglu_quant_v2_mxquant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_H

#include "cgmct/kernel/kernel_gmm_swiglu_mxquant.h"
#include "cgmct/block/block_mx_mm_aic_to_aiv_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;


template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluAswt(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                     GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
                                     GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);       \
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, mmTilingData, mmTilingData_, tiling);                       \
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    // 定义矩阵的类型和布局
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using weightscaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    // 定义MMAD类型
    using C1Type = float;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueSwigluQuant<L0TileShape, CType, C1Type, weightscaleType, weightscaleType,
                                                          true>;
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;
    using BlockMmad = Block::BlockMxMmAicToAivBuilder<AType, LayoutA, BType, LayoutB, BiasType, C1Type, LayoutC, L1TileShape,
                                                 L0TileShape, BlockScheduler, QuantMatmulWithTileMultiBlock<>,
                        Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;
    using QGmmKernel =
    Kernel::KernelGmmSwiGluMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum, gmmSwigluQuantParams_.groupListType, mmTilingData_.baseM,
                        mmTilingData_.baseN, mmTilingData_.baseK};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {// template shape, gmm shape can not get now
                    {1, 1, 1, 1},
                    // mmad args
                    {x, weight, weightScale, xScale, y, groupList},
                    {y, yScale, nullptr, nullptr, nullptr, static_cast<uint32_t>(mmTilingData_.baseM),
                        static_cast<uint32_t>(mmTilingData_.baseN)},
                    // gmm tiling data
                    gmmParams};
    QGmmKernel op;
    op(params);
}

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluMxFp8NdTensorApi(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                                 GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale,
                                                 GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);       \
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, mmTilingData, mmTilingData_, tiling);                       \
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using TensorLayoutA = AscendC::Std::conditional_t<TagToTrans<LayoutA>::value, AscendC::Te::DNExtLayoutPtn,
                                                      AscendC::Te::NDExtLayoutPtn>;
    using TensorLayoutB = AscendC::Std::conditional_t<TagToTrans<LayoutB>::value, AscendC::Te::DNExtLayoutPtn,
                                                      AscendC::Te::NDExtLayoutPtn>;
    using TensorLayoutC = AscendC::Te::NDExtLayoutPtn;
    using WeightScaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using C1Type = float;
    using BlockEpilogue = Block::BlockEpilogueSwigluQuant<L0TileShape, CType, C1Type, WeightScaleType,
                                                          WeightScaleType, true>;
    using ProblemShape = MatmulShape;
    using TensorBlockMmadPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<TensorBlockMmadPolicy, AType, TensorLayoutA, BType,
                                                    TensorLayoutB, C1Type, TensorLayoutC, BiasType, TensorLayoutC>;
    using QGmmKernel =
        Kernel::KernelGmmSwiGluMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum, gmmSwigluQuantParams_.groupListType, mmTilingData_.baseM,
                        mmTilingData_.baseN, mmTilingData_.baseK};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {{1, 1, 1, 1},
                     {x, weight, weightScale, xScale, y, groupList},
                     {y, yScale, nullptr, nullptr, nullptr, static_cast<uint32_t>(mmTilingData_.baseM),
                      static_cast<uint32_t>(mmTilingData_.baseN)},
                     gmmParams};
    QGmmKernel op;
    op(params);
}

#endif
