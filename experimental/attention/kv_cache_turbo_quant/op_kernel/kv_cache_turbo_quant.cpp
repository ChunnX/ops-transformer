/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_kv_cache_turbo_quant.h"

using namespace AscendC;
using namespace KvCacheTurboQuantKernel;

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix,
                                                           GM_ADDR quantIdx, GM_ADDR quantQjl, GM_ADDR quantNorm,
                                                           GM_ADDR quantGamma, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tilingData, tiling);
    TilingView view;
    view.mseBits = tilingData.mseBits;
    view.idxBytes = tilingData.idxBytes;
    view.qjlBytes = tilingData.qjlBytes;
    view.totalRows = tilingData.totalRows;
    view.coreRows = tilingData.coreRows;
    view.vecCoreNum = tilingData.vecCoreNum;

    TPipe pipe;
    using MM = Matmul<AType, BType, CType, BiasType>;
    MM mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTiling);

    KernelKvCacheTurboQuant<MM> op;
    op.Init(kvVectors, rotationMatrix, qjlMatrix, quantIdx, quantQjl, quantNorm, quantGamma, workspace, view, &pipe);

    if ASCEND_IS_AIV {
        op.InitBuffers();
        op.PrefetchUnits();
    }
    SyncAll();
    op.Rotate(mm);
    if ASCEND_IS_AIV {
        op.QuantizeAndResUnit();
    }
    op.Project(mm);
    if ASCEND_IS_AIV {
        op.PackQjlSigns();
    }
}
