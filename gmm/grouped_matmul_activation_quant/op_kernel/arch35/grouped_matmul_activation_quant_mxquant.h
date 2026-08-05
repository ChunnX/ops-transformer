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
 * \file grouped_matmul_activation_quant_mxquant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_ACTIVATION_QUANT_MXQUANT_H
#define GROUPED_MATMUL_ACTIVATION_QUANT_MXQUANT_H

#include "cgmct/kernel/kernel_gmm_activation_mxquant.h"
#include "cgmct/block/block_mmad_mx.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "cgmct/policy/dispatch_policy.h"
#include "grouped_matmul_activation_quant_tiling_data.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

namespace {
constexpr int64_t DEFAULT_PROBLEM_SHAPE_VALUE = 1;
} // namespace

namespace GroupedMatmulActivationQuant {
template <typename layoutA, typename layoutB>
__aicore__ inline void GmmActivationMxQuant(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                            GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(GMMActivationQuantTilingDataParams, tilingData, tiling);
    const auto &gmmActivationQuantParams_ = tilingData.gmmActivationQuantParams;
    const auto &mmTilingData_ = tilingData.mmTilingData;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using MmadCType = float;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajor;
    using ScaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using C1Type = MmadCType;
    using BlockEpilogue = Block::BlockEpilogueActivationQuant<L0TileShape, CType, C1Type, ScaleType, ScaleType, true>;
    using ProblemShape = MatmulShape;
    using BlockMmadPolicy = MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, 0>;
    using BlockMmad = Block::BlockMmadMx<BlockMmadPolicy, L1TileShape, L0TileShape, AType, LayoutA, BType, LayoutB,
                                         MmadCType, LayoutC, BiasType, LayoutC, void>;
    using QGmmKernel =
        Kernel::KernelGmmActivationMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmActivationQuantParams_.groupNum,
                        mmTilingData_.m,
                        mmTilingData_.n,
                        mmTilingData_.k,
                        mmTilingData_.baseM,
                        mmTilingData_.baseN,
                        mmTilingData_.baseK,
                        mmTilingData_.kAL1,
                        mmTilingData_.kBL1,
                        mmTilingData_.scaleKAL1,
                        mmTilingData_.scaleKBL1,
                        mmTilingData_.isBias,
                        mmTilingData_.dbL0C,
                        static_cast<int8_t>(GroupedMatmul::GMM_SPLIT_M),
                        gmmActivationQuantParams_.groupListType};
    Params params = {{DEFAULT_PROBLEM_SHAPE_VALUE, DEFAULT_PROBLEM_SHAPE_VALUE, DEFAULT_PROBLEM_SHAPE_VALUE,
                      DEFAULT_PROBLEM_SHAPE_VALUE},
                     {x, weight, weightScale, xScale, nullptr, nullptr, groupList},
                     {y, yScale, nullptr, nullptr, nullptr, static_cast<uint32_t>(mmTilingData_.baseM),
                      static_cast<uint32_t>(mmTilingData_.baseN),
                      static_cast<uint32_t>(gmmActivationQuantParams_.roundMode),
                      static_cast<uint32_t>(gmmActivationQuantParams_.scaleAlg), gmmActivationQuantParams_.dstTypeMax,
                      static_cast<uint32_t>(gmmActivationQuantParams_.activationType)},
                     gmmParams};
    QGmmKernel op;
    op(params);
}
} // namespace GroupedMatmulActivationQuant

#endif
