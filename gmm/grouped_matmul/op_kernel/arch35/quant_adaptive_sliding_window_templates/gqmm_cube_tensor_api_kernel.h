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
 * \brief V310_GMM_QUANT_CUBE / V310_GMM_QUANT_PERTENSOR_CUBE 的 Blaze（Tensor API）薄封装。
 *
 * 复用现有 GMMQuantTilingData（GMMQuantParams + GMMArray + TCubeTiling），host tiling 零改动；
 * apt.cpp 用 IS_BLAZE 分支派发到 blaze plain kernel（kernel_gmm_cube_quant.h）。
 * 多组（每组独立 m/n/k）由 GMMArray（mList/kList/nList）承载，与旧 cgmct GmmASWKernel 语义一致。
 */

#ifndef GQMM_CUBE_TENSOR_API_KERNEL_H
#define GQMM_CUBE_TENSOR_API_KERNEL_H

#include "blaze/gemm/kernel/kernel_gmm_cube_quant.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;

namespace GROUPED_MATMUL {

// DTYPE_SCALE（ge::DataType 枚举宏）→ GM scale 的 C++ 类型：uint64/int64（scale64）或 fp32/bf16（per-tensor）。
#if defined(DT_UINT64) && DTYPE_SCALE == DT_UINT64
using GmmCubeScaleGmType = uint64_t;
#elif defined(DT_INT64) && DTYPE_SCALE == DT_INT64
using GmmCubeScaleGmType = int64_t;
#elif defined(DT_BF16) && DTYPE_SCALE == DT_BF16
using GmmCubeScaleGmType = bfloat16_t;
#else
using GmmCubeScaleGmType = float;
#endif

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
    using ScaleGmType = GmmCubeScaleGmType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    // BTypeTuple = (B 元素类型, GM scale 类型)，block 的 X2ScaleType 取 tuple[1]。
    using BTypeTuple = AscendC::Std::tuple<BType, ScaleGmType>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
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
    gmmParams.kAL1 = static_cast<uint32_t>(mmTilingData_.stepKa) * baseK; // = stepKa * baseK
    gmmParams.kBL1 = static_cast<uint32_t>(mmTilingData_.stepKb) * baseK; // = stepKb * baseK
    gmmParams.nBufferNum = 2U;                                            // 双缓冲
    gmmParams.x1QuantMode = gmmQuantParams_.aQuantMode;
    gmmParams.x2QuantMode = gmmQuantParams_.bQuantMode;
    gmmParams.isBias = static_cast<uint8_t>(gmmQuantParams_.hasBias);
    gmmParams.dbL0C = static_cast<uint8_t>(mmTilingData_.dbL0C);
    gmmParams.groupType = static_cast<int8_t>(gmmQuantParams_.groupType);
    gmmParams.groupListType = gmmQuantParams_.groupListType;
    gmmParams.singleW = gmmQuantParams_.singleW;
    gmmParams.singleX = gmmQuantParams_.singleX;
    gmmParams.singleY = gmmQuantParams_.singleY;
    gmmParams.mTailCnt = 1U; // quant cube 无 no-quant tail 语义，固定 1
    gmmParams.nTailCnt = 1U;

    typename CubeBlockMmad::Params mmadParams{};
    // x/weight/y/bias 为 ListTensorDesc，kernel 内按 single/multi 解析；scale 解析为数据地址
    // （单 tensor [E, n]，每组 n 列连续，与 cgmct GmmASWKernel 的 GetTensorAddr(0, scale) 一致）；
    // perTokenScale 为普通 GM 指针。
    mmadParams.aGmAddr = x;
    mmadParams.bGmAddr = weight;
    mmadParams.cGmAddr = y;
    mmadParams.biasGmAddr = bias;
    mmadParams.scaleAGmAddr = perTokenScale;
    mmadParams.scaleBGmAddr =
        reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleGmType>(0, scale));

    Params params{};
    params.problemShape = ProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
    params.mmadParams = mmadParams;
    params.epilogueParams = {};
    params.groupListGmAddr = groupList;
    params.gmmArrayGmAddr = reinterpret_cast<GM_ADDR>(gmmArrayAddr_);
    params.gmmParams = gmmParams;

    CubeKernel gmm;
    gmm(params);
}

} // namespace GROUPED_MATMUL
#endif
