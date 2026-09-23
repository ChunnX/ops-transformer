/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file kv_cache_turbo_quant.cpp
 */
#include "kernel_kv_cache_turbo_quant.h"

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kv_vectors, GM_ADDR rotation_matrix,
                                                           GM_ADDR qjl_matrix, GM_ADDR quant_idx, GM_ADDR quant_qjl,
                                                           GM_ADDR quant_norm, GM_ADDR quant_gamma, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelKvCacheTurboQuant op;
    op.Init(kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl, quant_norm, quant_gamma, workspace,
            &tilingData);
    op.Process();
}
