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
 * \file grouped_matmul_swiglu_quant_weight_nz_v3_mxquant.h
 * \brief Cgmct entry for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V3_MXQUANT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V3_MXQUANT_H

#include "cgmct/kernel/kernel_gmm_swiglu_mxquant.h"
#include "cgmct/block/block_mx_mm_aic_to_aiv_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

namespace Cgmct {
namespace Gemm {
namespace Tile {} // namespace Tile
} // namespace Gemm
} // namespace Cgmct

namespace GroupedMatmulSwigluQuantV3 {

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluAswtV3(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                       GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
                                       GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantV3TilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluQuantV3TilingDataParams, mmTilingData, mmTilingData_, tiling);

    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using WeightScaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using C1Type = float;
    using BlockEpilogue =
        Block::BlockEpilogueSwigluQuant<L0TileShape, CType, C1Type, WeightScaleType, WeightScaleType, true>;
    using ProblemShape = MatmulShape;
    using DefaultTileCopy = Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>;
    using BlockMmad =
        Block::BlockMxMmAicToAivBuilder<AType, LayoutA, BType, LayoutB, BiasType, C1Type, LayoutC, L1TileShape,
                                        L0TileShape, BlockScheduler, QuantMatmulWithTileMultiBlock<>, DefaultTileCopy>;
    using QGmmKernel = Kernel::KernelGmmSwiGluMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    using EpilogueArguments = typename QGmmKernel::BlockEpilogueArguments;

    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum,
                        gmmSwigluQuantParams_.groupListType,
                        mmTilingData_.baseM,
                        mmTilingData_.baseN,
                        mmTilingData_.baseK,
                        gmmSwigluQuantParams_.isMxWeightNzMultiTensor};
    gmmParams.matmulTiling = &mmTilingData_;

    // Keep the aggregate initialization ABI-compatible with the V2 kernel;
    // the appended V3 fields have in-class defaults and are filled below.
    EpilogueArguments epilogueArgs{y,
                                   yScale,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   static_cast<uint32_t>(mmTilingData_.baseM),
                                   static_cast<uint32_t>(mmTilingData_.baseN)};
    epilogueArgs.swigluMode = gmmSwigluQuantParams_.swigluMode;
    epilogueArgs.clampLimit = gmmSwigluQuantParams_.clampLimit;
    epilogueArgs.gluAlpha = gmmSwigluQuantParams_.gluAlpha;
    epilogueArgs.gluBias = gmmSwigluQuantParams_.gluBias;
    // scaleAlg is part of the V3 tiling payload.  Read the named field here
    // so scale_alg=1 reaches the cuBLAS MX quantization path;
    // reserved0 is retained only for compatibility with older payloads.
    epilogueArgs.scaleAlg = gmmSwigluQuantParams_.scaleAlg;
    epilogueArgs.dstTypeMax = gmmSwigluQuantParams_.dstTypeMax;
    Params params = {{1, 1, 1, 1}, {x, weight, weightScale, xScale, y, groupList}, epilogueArgs, gmmParams};
    QGmmKernel op;
    op(params);
}

} // namespace GroupedMatmulSwigluQuantV3

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V3_MXQUANT_H
