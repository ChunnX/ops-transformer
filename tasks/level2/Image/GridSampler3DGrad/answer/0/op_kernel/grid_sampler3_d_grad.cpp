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
 * \file grid_sampler3_d_grad.cpp
 * \brief
 */

#include "grid_sampler3_d_grad.h"

using namespace GridSampler3DGrad;

extern "C" __global__ __aicore__ void grid_sampler3_d_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx, GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[6] = {grad, x, grid, dx, dgrid, workspace};

    if (TILING_KEY_IS(1)) {
        GridSampler3DGradNS<float> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    }
}
