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
 * \file grouped_matmul_tensor_api_kernel.h
 * \brief Grouped non-quantized matmul kernel adapter (Tensor API / Blaze)
 */

#ifndef GROUPED_MATMUL_TENSOR_API_KERNEL_H
#define GROUPED_MATMUL_TENSOR_API_KERNEL_H

#include "blaze/gemm/block/block_mmad_gmm_basic.h"
#include "blaze/gemm/kernel/kernel_gmm_basic.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMNoQuantBaseParams = GroupedMatmulTilingData::GMMNoQuantBaseParams;

template <class xType, class wType, class biasType, class yType, class xLayout, class wLayout, class yLayout>
__aicore__ inline void GmmTensorApiBasicKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList,
                                               GM_ADDR y, GM_ADDR workspace,
                                               const GMMNoQuantBaseParams *gmmBaseParamsIn,
                                               const TCubeTiling *mmTilingDataIn, AscendC::TPipe *que)
{
    (void)workspace;
    (void)que;
    if ASCEND_IS_AIV {
        return;
    }
    using AType = xType;
    using BType = wType;
    using BiasType = biasType;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    // ScheduleType_ = KernelGroupedMmadBasic routes GemmUniversal to kernel_gmm_basic.h.
    using BlockMmadPolicy = Blaze::Gemm::GroupedMatmulBasic<0, false, Blaze::Gemm::KernelGroupedMmadBasic>;
    using GmmBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC,
                                                       BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, GmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    // l1BufferNum is the L1 slot count (2 or 4), NOT tiling's depthA1 (= stepKa * DB_SIZE).
    // Non-quant tiling always emits depthA1 = 2 * stepKa (see CalStepKs / CalAswtL1Tiling), i.e. exactly
    // 2 slots per K window, and only l1BufferNum == 2 preserves independent kAL1 / kBL1.
    constexpr uint32_t GMM_BASIC_L1_BUFFER_NUM = 2U;

    GMMTiling gmmParams{static_cast<uint32_t>(gmmBaseParamsIn->groupNum),
                        static_cast<int64_t>(mmTilingDataIn->M),
                        static_cast<int64_t>(mmTilingDataIn->N),
                        static_cast<int64_t>(mmTilingDataIn->Ka),
                        static_cast<uint32_t>(mmTilingDataIn->baseM),
                        static_cast<uint32_t>(mmTilingDataIn->baseN),
                        static_cast<uint32_t>(mmTilingDataIn->baseK),
                        static_cast<uint32_t>(mmTilingDataIn->stepKa * mmTilingDataIn->baseK),
                        static_cast<uint32_t>(mmTilingDataIn->stepKb * mmTilingDataIn->baseK),
                        GMM_BASIC_L1_BUFFER_NUM,
                        static_cast<uint8_t>(mmTilingDataIn->isBias),
                        static_cast<uint8_t>(mmTilingDataIn->dbL0C),
                        static_cast<int8_t>(gmmBaseParamsIn->groupType),
                        static_cast<uint8_t>(gmmBaseParamsIn->groupListType),
                        static_cast<uint8_t>(gmmBaseParamsIn->singleWeight),
                        static_cast<uint8_t>(gmmBaseParamsIn->singleX),
                        static_cast<uint8_t>(gmmBaseParamsIn->singleY),
                        static_cast<uint8_t>(gmmBaseParamsIn->weightNoL2Cache),
                        static_cast<uint32_t>(gmmBaseParamsIn->mTailCnt),
                        static_cast<uint32_t>(gmmBaseParamsIn->nTailCnt)};

    // x / weight / y / bias are always ListTensorDesc at the GM_ADDR level. The kernel resolves per-group
    // pointers (multi-tensor) or list index 0 plus accumulated offsets (single tensor), same as CMCT.
    GM_ADDR biasDataAddr = gmmBaseParamsIn->hasBias == 1 ? bias : nullptr;

    Params params = {{gmmParams.m, gmmParams.n, gmmParams.k, static_cast<int64_t>(1)},
                     {x, weight, y, biasDataAddr},
                     {},
                     groupList,
                     gmmParams};

    GmmKernel gmm;
    gmm(params);
}

#endif
