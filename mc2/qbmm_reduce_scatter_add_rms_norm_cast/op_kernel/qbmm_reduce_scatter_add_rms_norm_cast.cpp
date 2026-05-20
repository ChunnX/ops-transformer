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
 * \file qbmm_reduce_scatter_add_rms_norm_cast.cpp
 * \brief
 */
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include "block/block_mmad.h"
#include "block/block_scheduler_policy.h"

// #include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "include/tensor_api/tensor.h"
#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"
#include "kernel_utils/common_utils.h"
#include "block/block_scheduler_policy.h"
#include "block/quant_matmul_mx_block_mmad_swat.h"


using namespace AscendC;
// using namespace QbmmReduceScatterAddRmsNormCastImpl;

__global__ __aicore__ void qbmm_reduce_scatter_add_rms_norm_cast(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, 
                                                                 GM_ADDR dScaleB, GM_ADDR bias,  GM_ADDR dScaleA,
                                                                 GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QbmmReduceScatterAddRmsNormCastTilingData);
    GET_TILING_DATA_WITH_STRUCT(QbmmReduceScatterAddRmsNormCastTilingData, tilingData, tilingGM);
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    using TypeScaleA = fp8_e8m0_t;
    using TypeScaleB = fp8_e8m0_t;
    static constexpr bool TransA = false;
    static constexpr bool TransB = true;
    constexpr int32_t SCALE_C0 = 2;

    using LayoutA = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    using LayoutB = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    using LayoutC = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;

    using layoutA = LayoutA;
    using layoutB = LayoutB;
    using layoutC = LayoutC;

    using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>;
    using ProblemShape = MatmulShape;

    using TypeATuple = AscendC::Std::tuple<TypeA, fp8_e8m0_t>;
    using TypeBTuple = AscendC::Std::tuple<TypeB, fp8_e8m0_t>;
    using LayoutATuple = AscendC::Std::tuple<LayoutA, LayoutScaleA>;
    using LayoutBTuple = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, TypeATuple, LayoutATuple, TypeBTuple, LayoutBTuple, TypeC, LayoutC>;

    // using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, layoutB, CType, layoutC>;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using CommParams = typename QuantMatmulKernelImpl::CommParams;
    using OutputParams = typename QuantMatmulKernelImpl::OutputParams;

    ProblemShape problemShape{tilingData.qbmmTilingdata.m, tilingData.qbmmTilingdata.n, tilingData.qbmmTilingdata.k, 1UL};
    
    // AIC 输出到 workspaceGM，AIV 从中读取做 alltoall 分发
    BlockMmadParams mmadParams{x1, x2, workspaceGM, dScaleA, dScaleB};
    L1Params l1Params{
        static_cast<uint64_t>(tilingData.qbmmTilingdata.stepK) * tilingData.qbmmTilingdata.baseK,
        tilingData.qbmmTilingdata.scaleKL1};
    BlockSchedulerParams schedulerParams{
        tilingData.qbmmTilingdata.baseM,
        tilingData.qbmmTilingdata.baseN,
        tilingData.qbmmTilingdata.mTailTile,
        tilingData.qbmmTilingdata.nTailTile,
        tilingData.qbmmTilingdata.mBaseTailSplitCnt,
        tilingData.qbmmTilingdata.nBaseTailSplitCnt,
        tilingData.qbmmTilingdata.mTailMain,
        tilingData.qbmmTilingdata.nTailMain};
    QBMMTiling qbmmParams{
        tilingData.qbmmTilingdata.baseM,
        tilingData.qbmmTilingdata.baseN,
        tilingData.qbmmTilingdata.baseK,
        tilingData.qbmmTilingdata.dbL0c};

    CommParams commParams{
        problemShape.m / tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize,
        static_cast<int32_t>(tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize)};
    OutputParams outputParams{y1, y2, x, y, gamma};
    

    TPipe pipe;
    TPipe *pipePtr = nullptr;
    if ASCEND_IS_AIV {
        pipePtr = &pipe;
    }

    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, commParams, outputParams};
    QuantMatmulKernelImpl quantMatmulKernelImpl;
    
    
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // PRINTF("enter kernel!");

        quantMatmulKernelImpl(params, pipePtr);

        // QbmmReduceScatterAddRmsNormCastMte<fp8_e4m3fn_t, fp8_e8m0_t> impl;
        // impl.Init(x1, x2, y, gamma, scale,  bias, perTokenScale, y1, 
        //         y2, x, workspaceGM, pipePtr, &tilingData);
        // impl.Process();
        return;
    }
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // PRINTF("enter kernel!");
        // QbmmReduceScatterAddRmsNormCastMte<DTYPE_Y, DTYPE_SCALE, DTYPE_Y1, DTYPE_Y2, DTYPE_X, DTYPE_GAMMA, true> op;
        // const QbmmReduceScatterAddRmsNormCastTilingData *qBmmReduceScatterAddRmsNormCastTilingData = &tilingData;
        // op.Init(x1, x2, y, gamma, scale,  bias, perTokenScale, y1, 
        //         y2, x, workspaceGM, &pipe, &tilingData);
        // op.Process();
    }
}

