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
* \file engram_fetch.cpp
* \brief engram_fetch 算子 kernel 入口
*        INFER_MODE(0): 推理路径（RDMA read），可选输出全 nullptr
*        WITH_GRAD_MODE(1): 训练路径（all-to-all），可选输出非空
*        训练态 issue kernel 内部: ① 桶排序 ② a2a(indices) ③ gather ④ a2a(embedding)
*        wait kernel 内部: ⑤ fetched[perm] = gathered（还原）
*/

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "engram_fetch_tiling_data.h"
#include "engram_fetch_tiling_key.h"
#include "arch35/engram_fetch_arch35.h"

using namespace AscendC;
using namespace Mc2Kernel;

// kernel 签名按 op 定义顺序: input0, input1, output0~5, workspace, tiling
// input:  commContext(0), indices(1)
// output: fetched(0), permOut(1), sendCountsOut(2), recvCountsOut(3), recvLocalEntryOut(4), numRecvOut(5)
// 推理态: permOut~numRecvOut 全为 nullptr
// 训练态: permOut~numRecvOut 非空（save-for-backward）
template<uint32_t EngramFetchMode>
__global__ __aicore__ void engram_fetch(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                        GM_ADDR permOut, GM_ADDR sendCountsOut,
                                        GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                        GM_ADDR numRecvOut,
                                        GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (EngramFetchMode == ENGRAM_FETCH_INFER_MODE) {
        // 推理路径: RDMA read（现有实现）
        TPipe pipe;
        EngramFetchArch35 op;
        op.Init(commContext, indices, fetched, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if constexpr (EngramFetchMode == ENGRAM_FETCH_WITH_GRAD_MODE) {
        // 训练路径: all-to-all（待实现）
        // ① 桶排序: indices → perm, sendCounts, sortedLocalEntry
        // ② a2a(sortedLocalEntry) → recvLocalEntry, recvCounts
        // ③ gather: commBuffer[rankId][recvLocalEntry] → recvTokens
        // ④ a2a(recvTokens) → gathered (用 epHcclBuffer)
        // ⑤ 还原在 wait kernel 内完成
        printf("hello engram_fetch WITH_GRAD_MODE\n");
    }
}