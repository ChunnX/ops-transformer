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
 * \file engram_fetch_wait.cpp
 * \brief engram_fetch_wait算子kernel入口
 */


#include "engram_fetch_wait_arch35.h"

using namespace Mc2Kernel;

__global__ __aicore__ void engram_fetch_wait(GM_ADDR commContext, GM_ADDR fetched, GM_ADDR fetchedOut,
                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchWaitTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchWaitTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    EngramFetchWaitArch35 op;
    op.Init(commContext, workspaceGM, &pipe);
    op.Process();
}
