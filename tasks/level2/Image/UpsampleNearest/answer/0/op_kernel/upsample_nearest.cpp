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
 * \file upsample_nearest.cpp
 * \brief
 */
#include "upsample_nearest.h"

using namespace UpsampleNearest;

extern "C" __global__ __aicore__ void upsample_nearest(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const UpsampleNearestTilingData* __restrict tilingDataParams = &tilingData;
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#define INIT_PROCESS                             \
    op.Init(input, output, userWS, &tilingData); \
    op.Process()

    if (TILING_KEY_IS(1000)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 0> op;
            INIT_PROCESS;
        }
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 0> op;
            INIT_PROCESS;
        }
    } else if (TILING_KEY_IS(1001)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 1> op;
            INIT_PROCESS;
        }
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 1> op;
            INIT_PROCESS;
        }
    } else if (TILING_KEY_IS(1002)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 2> op;
            INIT_PROCESS;
        }
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 2> op;
            INIT_PROCESS;
        }
    } else if (TILING_KEY_IS(1003)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 3> op;
            INIT_PROCESS;
        }
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 3> op;
            INIT_PROCESS;
        }
    }
}
