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
 * \file grouped_matmul_finalize_routing_pertoken_dequant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H

#include <cstdint>

#include "cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h"
#include "cgmct/block/block_mmad_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

template <typename layoutA, typename layoutB>
__aicore__ inline void grouped_matmul_finalize_routing_pertoken_dequant(
    GM_ADDR x, GM_ADDR w, GM_ADDR w_scale, GM_ADDR bias, GM_ADDR x_scale, GM_ADDR group_list, GM_ADDR share_input,
    GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    auto gmmFinalizeRoutingQuantParams_ = tilingData.gmmFinalizeRoutingDataParams;
    auto matmulTiling_ = tilingData.matmulTiling;
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::printf(
            "zzzlog [GMMFR][arch35 entry] groupNum=%u groupListType=%u batch=%u sharedInputOffset=%u sharedInputLen=%u "
            "residualScale=%f baseM=%u baseN=%u baseK=%u M=%u N=%u K=%u usedCoreNum=%u "
            "x=0x%llx w=0x%llx w_scale=0x%llx x_scale=0x%llx row_index=0x%llx y=0x%llx workspace=0x%llx\n",
            gmmFinalizeRoutingQuantParams_.groupNum, static_cast<uint32_t>(gmmFinalizeRoutingQuantParams_.groupListType),
            gmmFinalizeRoutingQuantParams_.batch, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
            gmmFinalizeRoutingQuantParams_.sharedInputLen, gmmFinalizeRoutingQuantParams_.residualScale,
            matmulTiling_.baseM, matmulTiling_.baseN, matmulTiling_.baseK,
            matmulTiling_.M, matmulTiling_.N, matmulTiling_.Ka, matmulTiling_.usedCoreNum,
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(x)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(w)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(w_scale)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(x_scale)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(row_index)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(y)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(workspaceGM)));
    }

    using L1TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using weightscaleType = DTYPE_SCALE;
    using BiasType = bfloat16_t;
    using LayoutBias = layout::RowMajor;
    using C1Type = std::conditional_t<std::is_same_v<AType, int8_t>, int32_t, float>; // matmul output dtype
    using xscaleType = float;
    using rowIndexType = DTYPE_ROW_INDEX;
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::GlobalTensor<rowIndexType> rowIndexDbg;
        rowIndexDbg.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType *>(row_index));
        const uint32_t rowIndexSample = (gmmFinalizeRoutingQuantParams_.batch < 8U) ? gmmFinalizeRoutingQuantParams_.batch : 8U;
        for (uint32_t i = 0; i < rowIndexSample; ++i) {
            auto idxVal = static_cast<long long>(rowIndexDbg.GetValue(i));
            AscendC::printf("zzzlog [GMMFR][arch35 row_index] i=%u val=%lld batch=%u\n", i, idxVal,
                            gmmFinalizeRoutingQuantParams_.batch);
            if (idxVal < 0 || static_cast<uint64_t>(idxVal) >= gmmFinalizeRoutingQuantParams_.batch) {
                AscendC::printf("zzzlog [GMMFR][arch35 potential_oob] row_index[%u]=%lld out of [0,%u)\n", i, idxVal,
                                gmmFinalizeRoutingQuantParams_.batch);
            }
        }

        AscendC::GlobalTensor<int64_t> groupListDbg;
        groupListDbg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(group_list));
        const uint32_t groupSample = (gmmFinalizeRoutingQuantParams_.groupNum < 8U) ? gmmFinalizeRoutingQuantParams_.groupNum : 8U;
        for (uint32_t i = 0; i < groupSample; ++i) {
            auto gVal = static_cast<long long>(groupListDbg.GetValue(i));
            AscendC::printf("zzzlog [GMMFR][arch35 group_list] i=%u val=%lld\n", i, gVal);
        }
    }

    using ProblemShape = Cgmct::Gemm::MatmulShape;
    using BlockScheduler = Cgmct::Gemm::GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadBuilder =
        Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, C1Type, LayoutC, BiasType, LayoutBias, L1TileShape,
                                L0TileShape, BlockScheduler, MatmulMultiBlock<>,
                                Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;

    using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<CType, BiasType>;

    using BlockEpilogueDequant =
        Cgmct::Gemm::Block::BlockEpilogueDequantFinalizeRouting<CType, C1Type, weightscaleType, xscaleType, BiasType,
                                                                rowIndexType>;

    using GmmKernel =
        Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                     BlockEpilogueDequant, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;
    Params params = {
        {1, 1, 1, 1},                // problem shape
        {x, w, y, bias, group_list}, // BlockMmadParams
        {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
         gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
         gmmFinalizeRoutingQuantParams_.residualScale},                                          // prologue params
        {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN}, // epilogue params
        gmmParams};
    GmmKernel gmm;
    gmm(params);
}
#endif