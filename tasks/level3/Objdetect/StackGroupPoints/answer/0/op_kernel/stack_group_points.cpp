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
 * \file stack_group_points.cpp
 * \brief
 */

#include "stack_group_points.h"

extern "C" __global__ __aicore__ void stack_group_points(
    GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices, GM_ADDR indices_batch_cnt, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(Tiling_Data, tiling);
    if (TILING_KEY_IS(0))
    {
        StackGroupPoints<half> op;
        op.Init(
            features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, Tiling_Data.m, Tiling_Data.b,
            Tiling_Data.c, Tiling_Data.n, Tiling_Data.nsample, Tiling_Data.res, Tiling_Data.reminder,
            Tiling_Data.featuresSize, Tiling_Data.indicesSize, Tiling_Data.fbcSize, Tiling_Data.ibcSize,
            Tiling_Data.outLength, Tiling_Data.actCore, Tiling_Data.standard);
        op.Process();
    }
    else if (TILING_KEY_IS(1))
    {
        StackGroupPoints<float> op;
        op.Init(
            features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, Tiling_Data.m, Tiling_Data.b,
            Tiling_Data.c, Tiling_Data.n, Tiling_Data.nsample, Tiling_Data.res, Tiling_Data.reminder,
            Tiling_Data.featuresSize, Tiling_Data.indicesSize, Tiling_Data.fbcSize, Tiling_Data.ibcSize,
            Tiling_Data.outLength, Tiling_Data.actCore, Tiling_Data.standard);
        op.Process();
    }
}