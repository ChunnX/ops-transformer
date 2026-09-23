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
 * \brief KvCacheTurboQuant kernel entry: one tiling key per primary coding width.
 */

#include "kernel_operator.h"
#include "kv_cache_turbo_quant_impl.hpp"

using namespace AscendC;

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kvVectors, GM_ADDR rotationMatrix,
                                                            GM_ADDR qjlMatrix, GM_ADDR quantIdx,
                                                            GM_ADDR quantQjl, GM_ADDR quantNorm,
                                                            GM_ADDR quantGamma, GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    // 1 AIC : 2 AIV. The row count of one AIV is half the row count of one AIC, see the probe notes
    // in the kernel implementation. The cube work per AIC is unchanged, only the vector side is
    // spread over twice as many cores.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::KvCacheTurboQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(optiling::KvCacheTurboQuantTilingData, tilingDataIn, tiling);

    if (tilingDataIn.mseBits == 2U) {
        NsKvCacheTurboQuant::KvCacheTurboQuant<2> op;
        op.Init(kvVectors, rotationMatrix, qjlMatrix, quantIdx, quantQjl, quantNorm, quantGamma,
                workspace, &tilingDataIn);
        op.Process();
    } else if (tilingDataIn.mseBits == 4U) {
        NsKvCacheTurboQuant::KvCacheTurboQuant<4> op;
        op.Init(kvVectors, rotationMatrix, qjlMatrix, quantIdx, quantQjl, quantNorm, quantGamma,
                workspace, &tilingDataIn);
        op.Process();
    } else {
        NsKvCacheTurboQuant::KvCacheTurboQuant<3> op;
        op.Init(kvVectors, rotationMatrix, qjlMatrix, quantIdx, quantQjl, quantNorm, quantGamma,
                workspace, &tilingDataIn);
        op.Process();
    }
}
