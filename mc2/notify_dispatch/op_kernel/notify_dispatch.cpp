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
 * \file notify_dispatch.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "notify_dispatch.h"
#include "notify_dispatch_tiling.h"
#include "notify_dispatch_tiling_key.h"
#include "notify_dispatch_a5.h"

using namespace Mc2Tiling;

#define KERNEL_USE_WORKSPACE (1 * 1024 * 1024)

__global__ __aicore__ void notify_dispatch(
    GM_ADDR context, GM_ADDR sendData, GM_ADDR tokenPerExpertData, GM_ADDR sendDataOffset, GM_ADDR recvData, GM_ADDR recvCount,
    GM_ADDR recvOffset, GM_ADDR expertGlobalOffset, GM_ADDR srcrankInExpertOffset, GM_ADDR rInSrcrankOffset,
    GM_ADDR totalRecvTokens, GM_ADDR maxBs, GM_ADDR recvTokensPerExpert, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(NotifyDispatchTilingData, tilingData, tiling);

    int localRank = tilingData.notifyDispatchInfo.localRankId;
    int localRankSize = tilingData.notifyDispatchInfo.localRankSize;
    int rank = tilingData.notifyDispatchInfo.rankId;
    int rankSize = tilingData.notifyDispatchInfo.rankSize;
    int64_t len = tilingData.notifyDispatchInfo.sendCount;
    int numTokens = tilingData.notifyDispatchInfo.numTokens;
    int round = tilingData.notifyDispatchInfo.round;
    int perRoundTokens = tilingData.notifyDispatchInfo.perRoundTokens;
    uint64_t totalWinSize = tilingData.notifyDispatchInfo.totalWinSize;

    GM_ADDR sendDataInput = sendData;
    GM_ADDR tokenPerExpertDataInput = tokenPerExpertData;
    GM_ADDR sendDataOffsetOutput = sendDataOffset;
    GM_ADDR recvDataOutput = recvData;

    // fill in unused args
    uint32_t extraFlag = 0;
    GM_ADDR scale = nullptr;
    int root = 0;
    int op = 0;
    int cycleCount = 0;
    int scaleCount = 0;
    GM_ADDR offset = nullptr;
    int blockNum = GetBlockNum();

    (void)context;
    if (TILING_KEY_IS(TILINGKEY_NOTIFY_DISPATCH_INT) || TILING_KEY_IS(TILINGKEY_NOTIFY_DISPATCH_A2)) {
        NotifyDispatch<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    } else if (TILING_KEY_IS(TILINGKEY_NOTIFY_DISPATCH_A5)) {
        NotifyDispatchA5<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    }
    // NotifyDispatch<int> opKernel(rank, rankSize, extraFlag);
    // opKernel.Init(KERNELS_ARGS_CALL_ALL2ALL());
    // opKernel.Process();
}
