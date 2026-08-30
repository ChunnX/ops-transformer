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
 * \file moe_tutel_combine_x.cpp
 * \brief MoeTutelCombineX kernel entry (Ascend950 / arch35 only).
 *        本算子只有这一个入口文件，不使用 *_apt.cpp；def.cpp 的 opFile.value 亦指向本文件名。
 */

#include "kernel_operator.h"
#include "arch35/moe_tutel_combine_x_kernel.h"
#include "arch35/moe_tutel_combine_x_tiling_data.h"

using namespace AscendC;
using namespace MoeTutelCombineXRegbase;

#define MOE_TUTEL_COMBINE_X_TILING_KEY_MAIN 10000

extern "C" __global__ __aicore__ void moe_tutel_combine_x(GM_ADDR y_grad, GM_ADDR gates, GM_ADDR indices,
                                                          GM_ADDR locations, GM_ADDR x_grad, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    REGISTER_TILING_DEFAULT(MoeTutelCombineXTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeTutelCombineXTilingData, tilingData, tiling);
    TPipe pipe;
    // 即使只有一个模板也必须显式声明非零 TilingKey，供编译器生成运行时入口映射。
    if (TILING_KEY_IS(MOE_TUTEL_COMBINE_X_TILING_KEY_MAIN)) {
        MoeTutelCombineXKernel<DTYPE_Y_GRAD> op;
        op.Init(y_grad, gates, indices, locations, x_grad, &tilingData, &pipe);
        op.Process();
    }
}
