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
 * \file gmm_weight_quant_tensor_api_mx_kernel.h
 * \brief Adapter from GroupedMatmul operator inputs and tiling data to the MX A8W4 Tensor API kernel.
 */

#ifndef GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
#define GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H

#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL {

struct GmmWeightQuantMxTiling {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint8_t cubeNumBlocksN = 0;
    uint32_t mainBlockSize = 0;
    uint64_t mainBlockCount = 0;
    uint16_t firstTailBlockSize = 0;
    uint16_t secondTailBlockSize = 0;
    uint16_t firstTailBlockCount = 0;
    uint16_t secondTailBlockCount = 0;
    uint16_t baseM = 0;
    uint32_t groupListType = 0;
    uint32_t hasBias = 0;
};

template <class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_, bool IsSingleMultiSingle_>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam *gmmBaseParams, const TCubeTiling *mmTilingData)
{
    using XType = XType_;
    using WeightType = WeightType_;
    using ScaleBType = ScaleBType_;
    using BiasType = BiasType_;
    using YType = YType_;
    constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;
    using AType = XType;
    using BType = WeightType;
    using ScaleAType = ScaleBType;
    using CType = YType;

    static_assert(AscendC::Std::is_same_v<AType, fp8_e4m3fn_t>, "AType must be fp8_e4m3fn_t");
    static_assert(AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t>,
                  "BType must be fp4x2_e2m1_t or fp4x2_e1m2_t");
    static_assert(AscendC::Std::is_same_v<ScaleAType, fp8_e8m0_t>, "Scale type must be fp8_e8m0_t");
    static_assert(AscendC::Std::is_one_of_v<BiasType, half, bfloat16_t>, "BiasType must be half or bfloat16_t");
    static_assert(AscendC::Std::is_one_of_v<CType, half, bfloat16_t>, "CType must be half or bfloat16_t");
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithWeightQuantMx;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::ZNLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;

    using ProblemShape = decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler =
        Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL))>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AscendC::Std::tuple<AType, ScaleAType>,
                                      AscendC::Std::tuple<LayoutA, LayoutScaleA>,
                                      AscendC::Std::tuple<BType, ScaleBType>,
                                      AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = void;
    using BlockPrologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<AType, BType, BiasType>;
    using KernelImpl =
        Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<ProblemShape, BlockMmad, BlockScheduler, BlockEpilogue,
                                                    BlockPrologue, IS_SINGLE_MULTI_SINGLE>;

    GmmWeightQuantMxTiling gmmTiling{};
    gmmTiling.groupNum = gmmBaseParams->groupNum;
    gmmTiling.coreNum = gmmBaseParams->coreNum;
    gmmTiling.kSize = gmmBaseParams->kSize;
    gmmTiling.nSize = gmmBaseParams->nSize;
    gmmTiling.cubeNumBlocksN = gmmBaseParams->cubeNumBlocksN;
    gmmTiling.mainBlockSize = gmmBaseParams->mainBlockSize;
    gmmTiling.mainBlockCount = gmmBaseParams->mainBlockCount;
    gmmTiling.firstTailBlockSize = gmmBaseParams->firstTailBlockSize;
    gmmTiling.secondTailBlockSize = gmmBaseParams->secondTailBlockSize;
    gmmTiling.firstTailBlockCount = gmmBaseParams->firstTailBlockCount;
    gmmTiling.secondTailBlockCount = gmmBaseParams->secondTailBlockCount;
    gmmTiling.baseM = mmTilingData->baseM;
    gmmTiling.groupListType = gmmBaseParams->groupListType;
    gmmTiling.hasBias = gmmBaseParams->hasBias;

    GM_ADDR xAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<XType>(0, x));
    GM_ADDR weightAddr = weight;
    GM_ADDR scaleBAddr = antiquantScale;
    GM_ADDR biasAddr = bias;
    GM_ADDR yAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    // SMS keeps W/ScaleB/Bias as tensor-list descriptors for per-group lookup.
    if constexpr (!IS_SINGLE_MULTI_SINGLE) {
        // The host builds this tiling-key bit as MX && singleX && !singleWeight && singleY. Therefore true is
        // SMS (singleWeight == 0), while a host-valid false key is SSS (singleWeight == 1); MX MMM is rejected by
        // the host NZ weight-dimension checks. Resolve each one-element list once for the SSS path.
        weightAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<WeightType>(0, weight));
        scaleBAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleBType>(0, antiquantScale));
        if (gmmBaseParams->hasBias != 0) {
            biasAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias));
        }
    }

    uint64_t maxNBlock = 0;
    if (gmmTiling.mainBlockCount > 0) {
        maxNBlock = gmmTiling.mainBlockSize;
    }
    if (gmmTiling.firstTailBlockCount > 0) {
        maxNBlock = maxNBlock > gmmTiling.firstTailBlockSize ? maxNBlock : gmmTiling.firstTailBlockSize;
    }
    if (gmmTiling.secondTailBlockCount > 0) {
        maxNBlock = maxNBlock > gmmTiling.secondTailBlockSize ? maxNBlock : gmmTiling.secondTailBlockSize;
    }
    maxNBlock = maxNBlock == 0 ? gmmTiling.nSize : maxNBlock;
    constexpr int64_t INITIAL_K_L1 = 512;
    constexpr int64_t SCALE_K_L1 = 4096;
    constexpr int64_t INITIAL_K_L0 = 256;
    typename BlockMmad::Params mmParams{
        xAddr,
        perTokenScale,
        scaleBAddr,
        yAddr,
        AscendC::Te::MakeShape(static_cast<int64_t>(gmmTiling.baseM), static_cast<int64_t>(maxNBlock), INITIAL_K_L1,
                               SCALE_K_L1),
        AscendC::Te::MakeShape(static_cast<int64_t>(gmmTiling.baseM), static_cast<int64_t>(maxNBlock), INITIAL_K_L0),
        static_cast<uint64_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT),
        gmmTiling.hasBias != 0};
    // baseN/baseK/stepKb are intentionally not forwarded. The scheduler recreates the host's main/first-tail/
    // second-tail N blocks, and BlockMmad derives KL1 from each actual (mL1, nL1) tile.
    typename BlockScheduler::Params schedulerParams{gmmTiling.mainBlockCount,
                                                    gmmTiling.mainBlockSize,
                                                    gmmTiling.firstTailBlockCount,
                                                    gmmTiling.firstTailBlockSize,
                                                    gmmTiling.secondTailBlockCount,
                                                    gmmTiling.secondTailBlockSize,
                                                    gmmTiling.coreNum,
                                                    gmmTiling.cubeNumBlocksN,
                                                    gmmTiling.baseM,
                                                    gmmTiling.nSize};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType *>(weightAddr)};
    typename KernelImpl::Params params{{0UL, gmmTiling.kSize, gmmTiling.nSize, gmmTiling.groupNum},
                                       mmParams,
                                       schedulerParams,
                                       prologueParams,
                                       groupList,
                                       gmmTiling.groupListType,
                                       gmmTiling.hasBias,
                                       biasAddr};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

} // namespace GROUPED_MATMUL

#endif // GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
