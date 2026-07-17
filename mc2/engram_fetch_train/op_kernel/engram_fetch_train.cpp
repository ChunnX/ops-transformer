/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
* \file engram_fetch_train.cpp
* \brief EngramFetchTrain 算子 kernel 入口（训练态前向，单一模式）
*        5 步融合 kernel:
*        ① 桶排序: indices → perm, sendCounts, sortedLocalEntry
*        ② a2a(sortedLocalEntry) → recvLocalEntry, recvCounts（用 commBuffer[peer] = a2a GM）
*        ③ gather: localStorageAddr[recvLocalEntry] → recvTokens
*        ④ a2a(recvTokens) → gathered（用 commBuffer[peer] = a2a GM）
*        ⑤ 还原: fetched[perm] = gathered
*
* kernel 签名按 op 定义顺序: input0~1, output0~5, workspace, tiling
* input:  commContext(0), indices(1)
* output: fetched(0), permOut(1), sendCountsOut(2), recvCountsOut(3), recvLocalEntryOut(4), numRecvOut(5)
*/

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "engram_fetch_train_tiling_data.h"
#include "engram_fetch_train_tiling_key.h"
#include "../../engram_fetch/op_kernel/engram_fetch.h"

using namespace AscendC;
using namespace Mc2Kernel;

template<uint32_t EngramFetchTrainMode>
__global__ __aicore__ void engram_fetch_train(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
    GM_ADDR permOut, GM_ADDR sendCountsOut,
    GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
    GM_ADDR numRecvOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchTrainTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchTrainTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 待实现:
    // ① 桶排序: indices → perm, sendCounts, sortedLocalEntry
    // ② a2a(sortedLocalEntry) → recvLocalEntry, recvCounts (用 commBuffer[peer] = a2a GM)
    // ③ gather: localStorageAddr[recvLocalEntry] → recvTokens
    // ④ a2a(recvTokens) → gathered (用 commBuffer[peer] = a2a GM)
    // ⑤ 还原: fetched[perm] = gathered
    printf("hello engram_fetch_train\n");
}
