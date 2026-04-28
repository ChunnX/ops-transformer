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
// #include "qbmm_reduce_scatter_add_rms_norm_cast_mte.h"

using namespace AscendC;
// using namespace QbmmReduceScatterAddRmsNormCastImpl;

__global__ __aicore__ void qbmm_reduce_scatter_add_rms_norm_cast(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, 
                                                                 GM_ADDR scale, GM_ADDR bias,  GM_ADDR perTokenScale,
                                                                 GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QbmmReduceScatterAddRmsNormCastTilingData);
    GET_TILING_DATA_WITH_STRUCT(QbmmReduceScatterAddRmsNormCastTilingData, tilingData, tilingGM);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        PRINTF("enter kernel!");
        // QbmmReduceScatterAddRmsNormCastMte<DTYPE_Y, DTYPE_SCALE, DTYPE_Y1, DTYPE_Y2, DTYPE_X, DTYPE_GAMMA, false> op;
        // const QbmmReduceScatterAddRmsNormCastTilingData *qBmmReduceScatterAddRmsNormCastTilingData = &tilingData;
        // op.Init(x1, x2, y, gamma, scale,  bias, perTokenScale, y1, 
        //         y2, x, workspaceGM, &pipe, &tilingData);
        // op.Process();
    }
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        PRINTF("enter kernel!");
        // QbmmReduceScatterAddRmsNormCastMte<DTYPE_Y, DTYPE_SCALE, DTYPE_Y1, DTYPE_Y2, DTYPE_X, DTYPE_GAMMA, true> op;
        // const QbmmReduceScatterAddRmsNormCastTilingData *qBmmReduceScatterAddRmsNormCastTilingData = &tilingData;
        // op.Init(x1, x2, y, gamma, scale,  bias, perTokenScale, y1, 
        //         y2, x, workspaceGM, &pipe, &tilingData);
        // op.Process();
    }
}
