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
 * \file upsample_nearest2d_grad.cpp
 * \brief
 */

#include "upsample_nearest_exact2d_grad.h"
#include "upsample_nearest_exact2d_grad_transpose.h"

using namespace UpSampleNearestExact2dGrad;

extern "C" __global__ __aicore__ void upsample_nearest2d_grad(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(UpsampleNearestExact2dGradTilingData, tilingData, tiling);
        const UpsampleNearestExact2dGradTilingData* __restrict tiling_data = &tilingData;
        const TCubeTiling* __restrict matmulTilingWTiling = &(tiling_data->matmulTiling_w);
        const TCubeTiling* __restrict matmulTilingHTiling = &(tiling_data->matmulTiling_h);
        UpSampleNearestExact2dGradND<DTYPE_GRAD_OUTPUT> op;
        REGIST_MATMUL_OBJ(
            &op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
        op.Init(input, output, false, userWS, &tilingData);
        op.Process();
    }
#ifndef __CCE_KT_TEST__
    if (TILING_KEY_IS(100)) {
        if ASCEND_IS_AIV {
            TPipe pipe;
            GET_TILING_DATA_WITH_STRUCT(UpsampleNearestExact2dGradTransposeTilingData, tilingData, tiling);
            UpSampleNearestExact2dGradTranspose<DTYPE_GRAD_OUTPUT, false, false> op;
            op.Init(input, output, userWS, &pipe, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(101)) {
        if ASCEND_IS_AIV {
            TPipe pipe;
            GET_TILING_DATA_WITH_STRUCT(UpsampleNearestExact2dGradTransposeTilingData, tilingData, tiling);
            UpSampleNearestExact2dGradTranspose<DTYPE_GRAD_OUTPUT, false, true> op;
            op.Init(input, output, userWS, &pipe, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(110)) {
        if ASCEND_IS_AIV {
            TPipe pipe;
            GET_TILING_DATA_WITH_STRUCT(UpsampleNearestExact2dGradTransposeTilingData, tilingData, tiling);
            UpSampleNearestExact2dGradTranspose<DTYPE_GRAD_OUTPUT, true, false> op;
            op.Init(input, output, userWS, &pipe, &tilingData);
            op.Process();
        }
    }
#endif
}
