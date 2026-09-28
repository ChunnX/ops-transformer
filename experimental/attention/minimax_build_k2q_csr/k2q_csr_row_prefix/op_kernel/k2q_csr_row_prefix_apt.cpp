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
 * K2qCsrRowPrefix apt（ascend950 / A5）。
 * 与 A2 共用 Prefix 逻辑。
 */
#include "kernel_operator.h"
#include "common/k2q_csr_tiling.h"
#include "common/k2q_csr_scratch.h"
#include "common/k2q_csr_prefix.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void k2q_csr_row_prefix(GM_ADDR scratch, GM_ADDR row_ptr, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(K2qCsrTilingData);
    GET_TILING_DATA_WITH_STRUCT(K2qCsrTilingData, tilingData, tiling);
    (void)workspace;

    GM_ADDR rowMap = nullptr;
    GM_ADDR tokenBatch = nullptr;
    GM_ADDR histScratch = nullptr;
    K2qCsrScratch::Split(scratch, &tilingData, rowMap, tokenBatch, histScratch);
    (void)rowMap;
    (void)tokenBatch;

    TPipe pipe;
    K2qCsrRowPrefixKernel op;
    op.Init(row_ptr, nullptr, nullptr, histScratch, &tilingData, &pipe);
    op.Process();
}
