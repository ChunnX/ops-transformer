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
 * \file grouped_matmul_swiglu_quant_v2_hif4quant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_HIF4QUANT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_HIF4QUANT_H

#include "cgmct/kernel/kernel_gmm_swiglu_hif4quant.h"
#include "cgmct/block/block_mmad_mx.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluHiF4(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                     GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
                                     GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantBasicApiTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluQuantBasicApiTilingDataParams, basicTilingData, basicTilingData_, tiling);
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
    using weightscaleType = AscendC::hif4_scale;
    using BiasType = float;
    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    // 定义MMAD类型
    using C1Type = float;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueSwigluHif4Quant<L0TileShape, CType, C1Type, weightscaleType, weightscaleType,
                                                          true>;
    using BlockMmadPolicy = MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, 0>;
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;
    using BlockMmad = Block::BlockMmadMx<BlockMmadPolicy, L1TileShape, L0TileShape, AType, LayoutA, BType, LayoutB,
                                           C1Type, LayoutC, BiasType, layout::RowMajor, void>;
    using QGmmKernel =
    Kernel::KernelGmmSwiGluHif4Quant<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum,  basicTilingData_.m,          basicTilingData_.n,
                        basicTilingData_.k,              basicTilingData_.baseM,       basicTilingData_.baseN,
                        basicTilingData_.baseK,          basicTilingData_.kAL1,       basicTilingData_.kBL1,
                        basicTilingData_.scaleKAL1,      basicTilingData_.scaleKBL1,  basicTilingData_.isBias,
                        basicTilingData_.dbL0C,          gmmSwigluQuantParams_.groupListType};
    Params params = {// template shape, gmm shape can not get now
                    {1, 1, 1, 1},
                    // mmad args
                    {x, weight, weightScale, xScale, y, nullptr, groupList},
                    {y, yScale, nullptr, nullptr, nullptr, static_cast<uint32_t>(basicTilingData_.baseM),
                        static_cast<uint32_t>(basicTilingData_.baseN)},
                    // gmm tiling data
                    gmmParams};
    QGmmKernel op;
    op(params);
}

#endif
