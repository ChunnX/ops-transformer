/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tri_inverse.cpp
 * \brief
 */

#ifndef TRI_INVERSE_USE_MIX_INVERSE
#define TRI_INVERSE_USE_MIX_INVERSE 1
#endif

#if TRI_INVERSE_USE_MIX_INVERSE
#include "tri_inverse_mix.h"
#else
#include "tri_inverse.h"
#endif
#include "tri_inverse_tiling_data.h"

using namespace AscendC;
using namespace TriInverse;

extern "C" __global__ __aicore__ void tri_inverse(GM_ADDR ccIn, GM_ADDR ccOut, GM_ADDR workspaceGM,
                                                  GM_ADDR tilingGM)
{
    (void)workspaceGM;
    REGISTER_TILING_DEFAULT(TriInverseTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    TriInverseOp op(&pipe, &tilingData);
    op.Init(ccIn, ccOut);
    op.Process();
}
