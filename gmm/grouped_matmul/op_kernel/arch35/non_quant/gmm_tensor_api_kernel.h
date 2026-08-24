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
 * \file gmm_tensor_api_kernel.h
 * \brief no_quant GroupedMatmul Blaze（Tensor API）薄封装。
 *
 * 复用现有 GMMNoQuantTilingData（GMMNoQuantBaseParams + TCubeTiling），host tiling 零改动；
 * 本薄封装把 tiling 数据转换为 blaze plain kernel（kernel_gmm_basic.h）的 GMMTiling：
 *   - GMMNoQuantBaseParams 字段直读（groupNum/singleX/singleW/singleY/groupType/groupListType/
 *     hasBias/mTailCnt/nTailCnt/weightNoL2Cache）；
 *   - TCubeTiling 字段直读（M/N/Ka → m/n/k，baseM/baseN/baseK，dbL0C）；
 *   - kAL1 = stepKa * baseK、kBL1 = stepKb * baseK 派生；
 *   - nBufferNum 固定 2（双缓冲）。
 */

#ifndef NON_QUANT_GMM_TENSOR_API_KERNEL_H
#define NON_QUANT_GMM_TENSOR_API_KERNEL_H

#include "blaze/gemm/kernel/kernel_gmm_basic.h"
#include "blaze/gemm/block/block_mmad_gmm_basic.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;

namespace GROUPED_MATMUL {

template <class xType, class yType, class biasType, class xLayout, class wLayout, class yLayout>
__aicore__ inline void GmmNoQuantTensorApiKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList,
                                                 GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    if ASCEND_IS_AIV {
        return;
    }
    printf("enter blaze no quant!!!!");
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, gmmNoQuantParam, gmmBaseParams, tiling);
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, mmTilingData, mmTilingData, tiling);

    using AType = xType;
    using BType = xType; // no_quant：x 与 weight 同 dtype
    using CType = yType;
    using BiasType = biasType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmadPolicy = Blaze::Gemm::GroupedMatmulBasic<>;
    using GmmBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                       BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, GmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    // GMMTiling 字段顺序与 kernel_gmm_basic.h:84-105 一致
    GMMTiling gmmParams{
        static_cast<uint32_t>(gmmBaseParams.groupNum),
        static_cast<int64_t>(mmTilingData.M),
        static_cast<int64_t>(mmTilingData.N),
        static_cast<int64_t>(mmTilingData.Ka),
        static_cast<uint32_t>(mmTilingData.baseM),
        static_cast<uint32_t>(mmTilingData.baseN),
        static_cast<uint32_t>(mmTilingData.baseK),
        static_cast<uint32_t>(mmTilingData.stepKa * mmTilingData.baseK), // kAL1 = stepKa * baseK
        static_cast<uint32_t>(mmTilingData.stepKb * mmTilingData.baseK), // kBL1 = stepKb * baseK
        static_cast<uint32_t>(2),                                        // nBufferNum：双缓冲
        static_cast<uint8_t>(gmmBaseParams.hasBias),                     // isBias
        static_cast<uint8_t>(mmTilingData.dbL0C),                        // dbL0C（host 填 1 → L0C 乒乓关）
        static_cast<int8_t>(gmmBaseParams.groupType),
        static_cast<uint8_t>(gmmBaseParams.groupListType),
        static_cast<uint8_t>(gmmBaseParams.singleWeight), // singleW
        static_cast<uint8_t>(gmmBaseParams.singleX),
        static_cast<uint8_t>(gmmBaseParams.singleY),
        static_cast<uint8_t>(gmmBaseParams.weightNoL2Cache),
        static_cast<uint32_t>(gmmBaseParams.mTailCnt),
        static_cast<uint32_t>(gmmBaseParams.nTailCnt),
    };

    // x/weight/y/bias 均为 ListTensorDesc（single 场景也走 list index 0，kernel 内 GetTensorAddrFromList 解析）
    Params params = {{gmmParams.m, gmmParams.n, gmmParams.k, 0},
                     {x, weight, y, bias},
                     {},
                     groupList,
                     gmmParams};

    GmmKernel gmm;
    gmm(params);
}

} // namespace GROUPED_MATMUL
#endif
