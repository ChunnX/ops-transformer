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
 * \file roi_align_rotated_grad.cpp
 * \brief
 */
#include "roi_align_rotated_grad.h"

extern "C" __global__ __aicore__ void roi_align_rotated_grad(GM_ADDR grad_output, GM_ADDR rois, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    const RoiAlignRotatedGradTilingData *__restrict tilingDevice = &tilingData;
    KernelRoiAlignRotatedGrad op;
    op.Init(grad_output, rois, grad_input, tilingDevice);
    op.Process();
}