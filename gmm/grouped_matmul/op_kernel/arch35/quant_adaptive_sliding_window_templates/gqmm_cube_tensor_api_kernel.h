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
 * \file gqmm_cube_tensor_api_kernel.h
 * \brief V310_GMM_QUANT_CUBE / V310_GMM_QUANT_PERTENSOR_CUBE 的 Blaze（Tensor API）
 */

#ifndef GQMM_CUBE_TENSOR_API_KERNEL_H
#define GQMM_CUBE_TENSOR_API_KERNEL_H

#include "blaze/gemm/kernel/kernel_qgmm_cube.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;

namespace GROUPED_MATMUL {

template <class xType, class wType, class biasType, class scaleType, class yType, class xLayout, class wLayout,
          class yLayout>
__aicore__ inline void GmmCubeTensorApiKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                              GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    if ASCEND_IS_AIV {
        return;
    }
    GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);
    GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);

    using AType = xType;
    using BType = wType;
    using CType = yType;
    using BiasType = biasType;
    using ScaleGmType = scaleType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BTypeTuple = AscendC::Std::tuple<BType, ScaleGmType>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                 Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
    using CubeBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType,
                                                        LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using CubeKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, CubeBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename CubeKernel::Params;
    using GMMTiling = typename CubeKernel::GMMTiling;

    const uint32_t baseK = static_cast<uint32_t>(mmTilingData_.baseK);
    GMMTiling gmmParams{};
    gmmParams.groupNum = gmmQuantParams_.groupNum;
    gmmParams.m = static_cast<int64_t>(mmTilingData_.M);
    gmmParams.n = static_cast<int64_t>(mmTilingData_.N);
    gmmParams.k = static_cast<int64_t>(mmTilingData_.Ka);
    gmmParams.baseM = static_cast<uint32_t>(mmTilingData_.baseM);
    gmmParams.baseN = static_cast<uint32_t>(mmTilingData_.baseN);
    gmmParams.baseK = baseK;
    gmmParams.kAL1 = static_cast<uint32_t>(mmTilingData_.stepKa) * baseK; 
    gmmParams.kBL1 = static_cast<uint32_t>(mmTilingData_.stepKb) * baseK; 
    // nBufferNum 不再由此传入：当前 cube fixpipe-quant BlockMmad 固定双缓冲，
    // kernel 内部直接使用 Blaze::Gemm::DOUBLE_BUFFER_COUNT。
    gmmParams.x1QuantMode = gmmQuantParams_.aQuantMode;
    gmmParams.x2QuantMode = gmmQuantParams_.bQuantMode;
    gmmParams.isBias = static_cast<uint8_t>(gmmQuantParams_.hasBias);
    gmmParams.dbL0C = static_cast<uint8_t>(mmTilingData_.dbL0C);
    gmmParams.groupType = static_cast<int8_t>(gmmQuantParams_.groupType);
    gmmParams.groupListType = gmmQuantParams_.groupListType;
    gmmParams.singleW = gmmQuantParams_.singleW;
    gmmParams.singleX = gmmQuantParams_.singleX;
    gmmParams.singleY = gmmQuantParams_.singleY;

    typename CubeBlockMmad::Params mmadParams{};
    // x/weight/y/bias 为 ListTensorDesc，kernel 内按 single/multi 解析；scale 解析为数据地址
    // （单 tensor [E, n]，每组 n 列连续，与 cgmct GmmASWKernel 的 GetTensorAddr(0, scale) 一致）；
    // perTokenScale 为普通 GM 指针。
    mmadParams.aGmAddr = x;
    mmadParams.bGmAddr = weight;
    mmadParams.cGmAddr = y;
    mmadParams.biasGmAddr = bias;
    if constexpr (AscendC::IsSameType<CType, int32_t>::value) {
        // INT8×INT8 + INT32 输出：L0C 直接写回 y，不需要 scale，
        // host 允许 scale 为空，不能把 scale 当 ListTensorDesc 解引用。
        mmadParams.scaleAGmAddr = nullptr;
        mmadParams.scaleBGmAddr = nullptr;
    } else {
        mmadParams.scaleAGmAddr = perTokenScale;
        mmadParams.scaleBGmAddr =
            reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleGmType>(0, scale));
    }

    Params params{};
    params.problemShape = ProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
    params.mmadParams = mmadParams;
    params.epilogueParams = {};
    params.groupListGmAddr = groupList;
    params.gmmArrayGmAddr = gmmArrayAddr_;
    params.gmmParams = gmmParams;

    CubeKernel gmm;
    gmm(params);
}

} // namespace GROUPED_MATMUL
#endif
