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

#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "tensor.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/common_utils.h"

#include "block/block_scheduler_policy.h"
#include "block/quant_matmul_mx_block_mmad_swat.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"

using namespace AscendC;
// using namespace QbmmReduceScatterAddRmsNormCastImpl;

__global__ __aicore__ void qbmm_reduce_scatter_add_rms_norm_cast(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, 
                                                                 GM_ADDR dScaleB, GM_ADDR bias,  GM_ADDR dScaleA,
                                                                 GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QbmmReduceScatterAddRmsNormCastTilingData);
    GET_TILING_DATA_WITH_STRUCT(QbmmReduceScatterAddRmsNormCastTilingData, tilingData, tilingGM);
    static constexpr uint64_t Stages = 2;
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using CType = bfloat16_t;

    using layoutA = layout::RowMajor;
    using layoutB = layout::ColumnMajor;  // B矩阵(K,N) ND格式，必须用ColumnMajor
    using layoutC = layout::RowMajor;

    using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, Stages>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, layoutB, CType, layoutC>;
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
    // BlockMmadParams mmadParams{x1, x2, x, dScaleA, dScaleB};
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
        // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        PRINTF("enter kernel!");

        quantMatmulKernelImpl(params, pipePtr);
        return;
    }
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        PRINTF("enter kernel!");

        quantMatmulKernelImpl(params, pipePtr);
        return;
    }
}

