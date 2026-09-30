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
 * \brief TensorAPI entry point for GroupedMatmulFinalizeRouting per-token dequant
 *        (INT8 x INT8 / FP8 E4M3FN x FP8 E4M3FN / HIFLOAT8 x HIFLOAT8 full quantization).
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H

#include "blaze/epilogue/block/block_epilogue_dequant_finalize_routing.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_gmm_pertoken_dequant_finalize_routing.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_mix_finalize_routing.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

template <typename layoutA, typename layoutB, int scaleType, int rowIndType>
__aicore__ inline void grouped_matmul_finalize_routing_pertoken_dequant(
    GM_ADDR x, GM_ADDR w, GM_ADDR w_scale, GM_ADDR bias, GM_ADDR x_scale, GM_ADDR group_list, GM_ADDR share_input,
    GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)offset;
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const auto &gmmParamsIn = tilingData.gmmFinalizeRoutingDataParams;
    const auto &matmulTiling = tilingData.matmulTiling;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    // Keep the MMAD accumulator in int32 for INT8 inputs and float for FP8/HIFLOAT8.
    using C1Type = AscendC::Std::conditional_t<AscendC::Std::is_same_v<AType, int8_t>, int32_t, float>;
    // 2 represents bf16 dtype for the per-channel weight scale.
    using WeightScaleType = AscendC::Std::conditional_t<scaleType == 2, bfloat16_t, float>;
    using BiasType = bfloat16_t;
    using RowIndexType = AscendC::Std::conditional_t<rowIndType == 1, int32_t, int64_t>;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockMmadPolicy = Blaze::Gemm::MatmulWithScaleMix<0, false, Blaze::Gemm::KernelGmmPertokenDequantMixFinalizeRouting>;
    // The Mix block takes B as a tuple; the per-token path has no fixpipe scale, so the
    // tuple only carries the weight dtype.
    using BTypeTuple = AscendC::Std::tuple<BType>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BTypeTuple, LayoutB, C1Type, LayoutC, BiasType,
                                      LayoutBias>;
    using BlockPrologue = Blaze::Gemm::Kernel::BlockPrologueFinalizeRouting<CType, BiasType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

    const uint32_t kAL1 = static_cast<uint32_t>(matmulTiling.stepKa) * static_cast<uint32_t>(matmulTiling.baseK);
    const uint32_t kBL1 = static_cast<uint32_t>(matmulTiling.stepKb) * static_cast<uint32_t>(matmulTiling.baseK);
    constexpr uint32_t doubleBufferNum = 2;

    typename BlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = x;
    mmadParams.bGmAddr = w;

    typename BlockPrologue::Params prologueParams{share_input,
                                                  y,
                                                  gmmParamsIn.sharedInputOffset,
                                                  gmmParamsIn.sharedInputLen,
                                                  matmulTiling.N,
                                                  gmmParamsIn.batch,
                                                  gmmParamsIn.residualScale};

    if (gmmParamsIn.deterministicFlag == 1) {
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
        using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequantFinalizeRouting<
            CType, C1Type, WeightScaleType, float, BiasType, RowIndexType, float, true>;
        using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad,
                                                             AscendC::Std::tuple<BlockPrologue, BlockEpilogue>,
                                                             BlockScheduler>;
        typename BlockEpilogue::Params epilogueParams{};
        epilogueParams.yGmAddr = y;
        epilogueParams.x2ScaleGmAddr = w_scale;
        epilogueParams.x1ScaleGmAddr = x_scale;
        epilogueParams.biasGmAddr = bias;
        epilogueParams.logitGmAddr = logit;
        epilogueParams.rowIndexGmAddr = row_index;
        epilogueParams.baseM = matmulTiling.baseM;
        epilogueParams.baseN = matmulTiling.baseN;
        epilogueParams.workspaceGmAddr = userWorkspace;
        epilogueParams.deterWorkspaceSize = gmmParamsIn.deterWorkspaceSize;

        typename GmmKernel::GMMTiling gmmTiling{gmmParamsIn.groupNum,
                                                gmmParamsIn.batch,
                                                gmmParamsIn.sharedInputOffset,
                                                gmmParamsIn.sharedInputLen,
                                                gmmParamsIn.residualScale,
                                                static_cast<uint32_t>(matmulTiling.baseM),
                                                static_cast<uint32_t>(matmulTiling.baseN),
                                                static_cast<uint32_t>(matmulTiling.baseK),
                                                kAL1,
                                                kBL1,
                                                doubleBufferNum,
                                                gmmParamsIn.hasBias,
                                                static_cast<uint8_t>(matmulTiling.dbL0C),
                                                gmmParamsIn.groupListType,
                                                gmmParamsIn.deterWorkspaceSize,
                                                static_cast<uint32_t>(matmulTiling.usedCoreNum)};
        typename GmmKernel::Params params{
            ProblemShape{static_cast<int64_t>(matmulTiling.M), static_cast<int64_t>(matmulTiling.N),
                         static_cast<int64_t>(matmulTiling.Ka), static_cast<int64_t>(1)},
            mmadParams, prologueParams, epilogueParams, group_list, gmmTiling};
        GmmKernel gmm;
        gmm(params);
    } else {
        using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequantFinalizeRouting<
            CType, C1Type, WeightScaleType, float, BiasType, RowIndexType, float, false>;
        using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad,
                                                             AscendC::Std::tuple<BlockPrologue, BlockEpilogue>,
                                                             BlockScheduler>;
        typename BlockEpilogue::Params epilogueParams{};
        epilogueParams.yGmAddr = y;
        epilogueParams.x2ScaleGmAddr = w_scale;
        epilogueParams.x1ScaleGmAddr = x_scale;
        epilogueParams.biasGmAddr = bias;
        epilogueParams.logitGmAddr = logit;
        epilogueParams.rowIndexGmAddr = row_index;
        epilogueParams.baseM = matmulTiling.baseM;
        epilogueParams.baseN = matmulTiling.baseN;

        typename GmmKernel::GMMTiling gmmTiling{gmmParamsIn.groupNum,
                                                gmmParamsIn.batch,
                                                gmmParamsIn.sharedInputOffset,
                                                gmmParamsIn.sharedInputLen,
                                                gmmParamsIn.residualScale,
                                                static_cast<uint32_t>(matmulTiling.baseM),
                                                static_cast<uint32_t>(matmulTiling.baseN),
                                                static_cast<uint32_t>(matmulTiling.baseK),
                                                kAL1,
                                                kBL1,
                                                doubleBufferNum,
                                                gmmParamsIn.hasBias,
                                                static_cast<uint8_t>(matmulTiling.dbL0C),
                                                gmmParamsIn.groupListType,
                                                0,
                                                0};
        typename GmmKernel::Params params{
            ProblemShape{static_cast<int64_t>(matmulTiling.M), static_cast<int64_t>(matmulTiling.N),
                         static_cast<int64_t>(matmulTiling.Ka), static_cast<int64_t>(1)},
            mmadParams, prologueParams, epilogueParams, group_list, gmmTiling};
        GmmKernel gmm;
        gmm(params);
    }
    (void)workspaceGM;
}
#endif
