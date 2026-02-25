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
 * \file rfft1_d.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/pad/kernel_operator_pad_intf.h"
#include "lib/matmul_intf.h"
#include "rfft1_d.h"

static const uint32_t DFT_BORDER_VALUE = 4096;

extern "C" __global__ __aicore__ void rfft1_d(GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);

    // FastDFT for length <= 4096
    uint32_t* factorsP = const_cast<uint32_t*>(tilingData.factors);

    KernelRfftFastDFT op(
        tilingData.length, tilingData.batchesPerCore, tilingData.leftOverBatches, tilingData.normal,
        tilingData.dftRealOverallSize, factorsP);

    uint32_t modeLength = 2 * (tilingData.length / 2 + 1);

    auto t1 = PrepareTiling((op.batches + op.advancedBatches) / GetBlockNum(), op.modeLength, tilingData.length);
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulObj, (void*)&t1, op.matmulObjNZ, (void*)&t1);
    op.Init(x, dft, y, userWorkspace);
    op.Process();

    return;
}