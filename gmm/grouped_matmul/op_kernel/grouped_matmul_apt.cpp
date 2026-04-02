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
 * \file grouped_matmul_apt.cpp
 * \brief
 */

#include "grouped_matmul_utils.h"
#include "arch35/grouped_matmul_tiling_data_apt.h"
using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;
#if defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/policy/dispatch_policy.h"
#include "arch35/weight_quant_basic_block/block/block_mmad.h"
#include "arch35/weight_quant_basic_block/block/grouped_matmul_scheduler_n_resplit.h"
#include "arch35/weight_quant_basic_block/prologue/block_prologue.h"
#include "arch35/weight_quant_basic_block/kernel/kernel.h"
#include "arch35/weight_quant_basic_block/weight_quant_tiling_key.h"

__aicore__ inline void LaunchMxA8W4VectorAntiQuantResplit(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR antiquantScale, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);
    
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using ScaleBType = DTYPE_ANTIQUANT_SCALE;
    using ScaleAType = DTYPE_PER_TOKEN_SCALE;
    using BiasType = DTYPE_BIAS;
    using CType = DTYPE_Y;

    using DispatchPolicy = GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit;
    using L1TileShape = decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL, 0UL));
    using L0TileShape = decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL));
    // (n, k) (k1,n1,n0,k0)
    constexpr static auto _1 = AscendC::Std::Int<1>{};
    constexpr static auto _16 = AscendC::Std::Int<32>{};
    constexpr static auto _32 = AscendC::Std::Int<32>{};
    constexpr static auto _512 = AscendC::Std::Int<512>{};
    using LayoutA = typename AscendC::Te::NDLayoutFormat<AType>;
    using LayoutB = decltype(AscendC::Te::MakeLayout(AscendC::Te::MakeShape(AscendC::Te::MakeShape(1UL, _32), AscendC::Te::MakeShape(1UL, _16))), AscendC::Te::MakeStride(AscendC::Te::MakeStride(_1, 16UL), AscendC::Te::MakeStride(_32, _512)));
    using LayoutC = typename AscendC::Te::NDLayoutFormat<AType>;
    using LayoutScaleA = typename AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using LayoutScaleB = typename AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;
    using ProblemShape = decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler = Block::GroupedMatmulSchedulerNResplit;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AscendC::Std::tuple<AType, ScaleAType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>, AscendC::Std::tuple<BType, ScaleBType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType>;
    using BlockEpilogue = void;
    using BlockPrologue = Prologue::BlockPrologue<DispatchPolicy, AType, BType, BiasType>;
    using KernelImpl =
        Kernel::GroupedMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, BlockPrologue>;

    typename BlockMmad::Params mmadParams{reinterpret_cast<__gm__ AType *>(x), reinterpret_cast<__gm__ float8_e8m0_t *>(perTokenScale), reinterpret_cast<__gm__ float8_e8m0_t *>(antiquantScale), reinterpret_cast<__gm__ CType *>(y), gmmBaseParams_.kSize, gmmBaseParams_.hasBias};
    typename BlockScheduler::Params schedulerParams{gmmBaseParams_.mainBlockCount, gmmBaseParams_.mainBlockSize, gmmBaseParams_.firstTailBlockCount, gmmBaseParams_.firstTailBlockSize, gmmBaseParams_.secondTailBlockCount, gmmBaseParams_.secondTailBlockSize, gmmBaseParams_.coreNum, gmmBaseParams_.cubeNumBlocksN, mmTilingData_.baseM, gmmBaseParams_.nSize};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType *>(weight), reinterpret_cast<__gm__ BiasType *>(bias), gmmBaseParams_.hasBias};
    typename KernelImpl::Params params = {
        {0UL, gmmBaseParams_.kSize, gmmBaseParams_.nSize, gmmBaseParams_.groupNum},
        mmadParams, schedulerParams, prologueParams, gmmBaseParams_.groupNum, gmmBaseParams_.groupType, gmmBaseParams_.groupListType, groupList};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY, int8_t ALGORITHM_CATEGORY>
__aicore__ inline constexpr bool IsMxA8W4VectorAntiQuantResplit()
{
    return W_TYPE == WQGMM_FRACTAL_NZ &&
           OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
           C_QUANT_TYPE == WQGMM_NONE &&
           W_QUANT_TYPE == WQGMM_MX &&
           WQ_B_TRANS == WQGMM_TRANS &&
           WQ_A_TRANS == WQGMM_NO_TRANS &&
           TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4 &&
           ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&
           ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT;
}
#endif

using namespace AscendC;
using namespace GROUPED_MATMUL;


template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    AscendCUtils::SetOverflow(1);
#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_ANTI_QUANT)
    REGISTER_TILING_DEFAULT(GMMWeightQuantTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if ORIG_DTYPE_X == DT_FLOAT8_E4M3FN
        if constexpr (IsMxA8W4VectorAntiQuantResplit<W_TYPE, OFFSET_OR_BIAS_EXIT, C_QUANT_TYPE, W_QUANT_TYPE,
                                                     WQ_B_TRANS, WQ_A_TRANS, TEMPLATE_CUSTOM_SC,
                                                     ALGORITHM_SUB_CATEGORY, ALGORITHM_CATEGORY>()) {
            LaunchMxA8W4VectorAntiQuantResplit(x, weight, bias, antiquantScale, groupList, perTokenScale, y, tiling);
        }
    #endif
#endif
#endif
}
