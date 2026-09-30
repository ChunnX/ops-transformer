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

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qgmm_cube.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantCubeBasicApiTilingData = GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData;

namespace GROUPED_MATMUL {

template <class xType, class wType, class biasType, class scaleType, class yType, class xLayout, class wLayout,
          class yLayout>
__aicore__ inline void GmmCubeTensorApiKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                              GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    if ASCEND_IS_AIV {
        return;
    }
    GET_TILING_DATA_MEMBER(GMMQuantCubeBasicApiTilingData, gmmQuantParams, gmmQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMQuantCubeBasicApiTilingData, mmTilingData, mmTilingData_, tiling);
    GET_TILING_DATA_MEMBER_ADDR(GMMQuantCubeBasicApiTilingData, gmmArray, gmmArrayAddr_, tiling);

    using AType = xType;
    using BType = wType;
    using CType = yType;
    // Host admits only native accumulator bias types when bias is present.
    // Absent optional bias may have an unrelated framework default dtype.
    using BiasType = AscendC::Std::conditional_t<AscendC::IsSameType<xType, int8_t>::value, int32_t, float>;
    // INT32 output never reads scale; keep its unused template type valid even for an absent scale.
    using ScaleGmType = AscendC::Std::conditional_t<AscendC::IsSameType<CType, int32_t>::value, uint64_t, scaleType>;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BTypeTuple = AscendC::Std::tuple<BType, ScaleGmType>;
    // Preserve the tiling member's const/GM address space in the Blaze kernel.
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                 Blaze::Gemm::KernelGroupedMmadFixpipeQuant, decltype(gmmArrayAddr_)>;
    using CubeBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType,
                                                        LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using CubeKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, CubeBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename CubeKernel::Params;
    using GMMTiling = typename CubeKernel::GmmParams;

    // nBufferNum 不再由此传入：当前 cube fixpipe-quant BlockMmad 固定双缓冲，
    // kernel 内部直接使用 Blaze::Gemm::DOUBLE_BUFFER_COUNT。
    GMMTiling gmmParams{gmmQuantParams_.groupNum,
                        static_cast<int64_t>(mmTilingData_.m),
                        static_cast<int64_t>(mmTilingData_.n),
                        static_cast<int64_t>(mmTilingData_.k),
                        static_cast<uint32_t>(mmTilingData_.baseM),
                        static_cast<uint32_t>(mmTilingData_.baseN),
                        static_cast<uint32_t>(mmTilingData_.baseK),
                        mmTilingData_.kAL1,
                        mmTilingData_.kBL1,
                        gmmQuantParams_.aQuantMode,
                        gmmQuantParams_.bQuantMode,
                        mmTilingData_.isBias,
                        static_cast<uint8_t>(mmTilingData_.dbL0C),
                        static_cast<int8_t>(gmmQuantParams_.groupType),
                        gmmQuantParams_.groupListType,
                        gmmQuantParams_.singleW,
                        gmmQuantParams_.singleX,
                        gmmQuantParams_.singleY};

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
        mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleGmType>(0, scale));
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
