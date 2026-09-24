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
 * \file kv_cache_turbo_quant.cpp
 * \brief kernel 入口（MIX_AIC_1_2）：tilingkey 202/203/204 = Cube 混合路径（AIC GEMM + AIV 准备/量化，CrossCore 握手）
 */
#include "kernel_kv_cache_turbo_quant_cube.h"
#include "kv_cache_turbo_quant_tiling_key.h"

using namespace AscendC;
using namespace KvCacheTurboQuant;

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kv_vectors, GM_ADDR rotation_matrix,
                                                           GM_ADDR qjl_matrix, GM_ADDR quant_idx, GM_ADDR quant_qjl,
                                                           GM_ADDR quant_norm, GM_ADDR quant_gamma, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KvCacheTurboQuantTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // MIX_AIC_1_2：AIC 数 = SetBlockDim，AIV 数 = 2×SetBlockDim。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    if (TILING_KEY_IS(KVTQ_TILING_KEY_MSE2)) {
        KernelKvCacheTurboQuantCube<2> op;
        op.Init(kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl, quant_norm, quant_gamma, workspace,
                &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KVTQ_TILING_KEY_MSE3)) {
        KernelKvCacheTurboQuantCube<3> op;
        op.Init(kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl, quant_norm, quant_gamma, workspace,
                &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KVTQ_TILING_KEY_MSE4)) {
        KernelKvCacheTurboQuantCube<4> op;
        op.Init(kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl, quant_norm, quant_gamma, workspace,
                &tilingData, &pipe);
        op.Process();
    }
}
