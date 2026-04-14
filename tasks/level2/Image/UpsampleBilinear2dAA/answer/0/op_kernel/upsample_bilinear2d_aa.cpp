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
 * \file upsample_bilinear2d_aa.cpp
 * \brief
 */

#include "upsample_bilinear2d_aa.h"

using namespace UpsampleBilinear2dAA;

extern "C" __global__ __aicore__ void upsample_bilinear2d_aa(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    const UpsampleBilinearAATilingData *__restrict tiling_data = &tilingData;
    const TCubeTiling *__restrict matmulTilingWTiling = &(tiling_data->matmulTiling_w);
    const TCubeTiling *__restrict matmulTilingHTiling = &(tiling_data->matmulTiling_h);

    // foreach(vector) not need workspace
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        if (tiling_data->dataType == 1) {
            UpsampleBilinearAAND<half> op;
            REGIST_MATMUL_OBJ(
                &op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
            op.Init(input, output, userWS, &tilingData);
            op.Process();
        }
        if (tiling_data->dataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            UpsampleBilinearAAND<float> op;
            REGIST_MATMUL_OBJ(
                &op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
            op.Init(input, output, userWS, &tilingData);
            op.Process();
#endif
        }
        if (tiling_data->dataType == 3) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            UpsampleBilinearAAND<bfloat16_t> op;
            REGIST_MATMUL_OBJ(
                &op.pipe, GetSysWorkSpacePtr(), op.matmulW, matmulTilingWTiling, op.matmulH, matmulTilingHTiling);
            op.Init(input, output, userWS, &tilingData);
            op.Process();
#endif
        }
    }
}
