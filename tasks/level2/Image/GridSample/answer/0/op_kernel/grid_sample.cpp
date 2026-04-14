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
 * \file grid_sample.cpp
 * \brief
 */

#include "grid_sampler_2d.h"
#include "grid_sampler_2d_bicubic.h"
#include "grid_sampler_2d_nearest.h"
#include "grid_sampler_2d_slide_window.h"
#include "grid_sampler_2d_fp16_slide_window.h"
#include "grid_sampler_2d_fullLoad.h"
#include "grid_sampler_3d.h"
#include "grid_sampler_3d_nearest.h"
#include "grid_sampler_3d_portrait.h"

using namespace GridSample;

extern "C" __global__ __aicore__ void grid_sample(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1000220)) {
        // 2D Bilinear fp32 normal
        GridSample::GridSampler2D<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1000221) || TILING_KEY_IS(1001221)) {
        // 2D nearest fp32 normal
        GridSample::GridSampler2DNearest<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1000211) || TILING_KEY_IS(1001211)) {
        // 2D nearest fp16 normal
        GridSample::GridSampler2DNearest<half> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1000222) || TILING_KEY_IS(1001222)) {
        // 2D Bicubic fp32 normal
        GridSample::GridSamplerBicubic2D<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1000212) || TILING_KEY_IS(1001212)) {
        // 2D Bicubic fp16 normal
        GridSample::GridSamplerBicubic2D<half> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1001220)) {
        // 2D Bilinear fp32 slide window
        GridSample::GridSampler2DSlideWindow<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1000210) || TILING_KEY_IS(1001210)) {
        // 2D Bilinear fp16 sliceWindow
        GridSample::GridSampler2DFP16SlideWindow<half> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2000220) || TILING_KEY_IS(2001220)) {
        // 2D Bilinear fp32 fullLoad general
        GridSample::GridSampler2DFullLoad<float, 0> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2000210) || TILING_KEY_IS(2001210)) {
        // 2D Bilinear fp16 fullLoad general
        GridSample::GridSampler2DFullLoad<half, 0> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2100220) || TILING_KEY_IS(2101220)) {
        // 2D Bilinear fp32 fullLoad C=1 and small input
        GridSample::GridSampler2DFullLoad<float, 1> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2100210) || TILING_KEY_IS(2101210)) {
        // 2D Bilinear fp16 fullLoad C=1 and small input
        GridSample::GridSampler2DFullLoad<half, 1> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2200220) || TILING_KEY_IS(2201220)) {
        // 2D Bilinear fp32 fullLoad C=32 and large input
        GridSample::GridSampler2DFullLoad<float, 2> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2200210) || TILING_KEY_IS(2201210)) {
        // 2D Bilinear fp16 fullLoad C=32 and large input
        GridSample::GridSampler2DFullLoad<half, 2> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1010320)) {
        // 3D Bilinear fp32 normal
        GridSample::GridSampler3D<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1010310)) {
        // 3D Bilinear fp16 normal
        GridSample::GridSampler3D<half> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1010330) || TILING_KEY_IS(1011330)) {
        // 3D Bilinear bf16 normal
        GridSample::GridSampler3D<bfloat16_t> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1010321) || TILING_KEY_IS(1011321)) {
        // 3D nearest fp32 normal
        GridSample::GridSampler3DNearest<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1010311) || TILING_KEY_IS(1011311)) {
        // 3D nearest fp16 normal
        GridSample::GridSampler3DNearest<half> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1010331) || TILING_KEY_IS(1011331)) {
        // 3D nearest bf16 normal
        GridSample::GridSampler3DNearest<bfloat16_t> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1011320)) {
        GridSample::GridSampler3DPortrait<float> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1011310)) {
        GridSample::GridSampler3DPortrait<half> op;
        op.Init(x, grid, y, userWS, &tilingData, pipe);
        op.Process();
    }
}
