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
 * \brief NotifyDispatch kernel entry.
 */

#include "kernel_operator.h"
#include "notify_dispatch.h"
#include "notify_dispatch_tiling.h"
#include "notify_dispatch_tiling_key.h"
#include "notify_dispatch_a5.h"

using namespace Mc2Tiling;

#define KERNEL_USE_WORKSPACE (1 * 1024 * 1024)

template <typename T>
__aicore__ inline __gm__ T *GetTensorAddrFromTensorList(uint16_t index, GM_ADDR tensorListPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *tensorAddrList = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(tensorAddrList + index));
}

template <uint8_t ArchTag>
__global__ __aicore__ void notify_dispatch(GM_ADDR context, GM_ADDR sendData, GM_ADDR tokenPerExpertData,
                                          GM_ADDR sendDataOffset, GM_ADDR recvData, GM_ADDR recv_offset,
                                          GM_ADDR expert_output_offset, GM_ADDR totalRecvTokens, GM_ADDR maxBs,
                                          GM_ADDR recvTokensPerExpert, GM_ADDR ext_handle_data, GM_ADDR workspace,
                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(NotifyDispatchTilingData, tilingData, tiling);

    (void)ext_handle_data;

    GM_ADDR recvOffset = reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(0, recv_offset));
    GM_ADDR recvCount = reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(1, recv_offset));
    GM_ADDR expertGlobalOffset =
        reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(0, expert_output_offset));
    GM_ADDR srcrankInExpertOffset =
        reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(1, expert_output_offset));
    GM_ADDR rInSrcrankOffset =
        reinterpret_cast<GM_ADDR>(GetTensorAddrFromTensorList<int32_t>(2, expert_output_offset));

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

    uint32_t extraFlag = 0;
    GM_ADDR scale = nullptr;
    int root = 0;
    int op = 0;
    int cycleCount = 0;
    int scaleCount = 0;
    GM_ADDR offset = nullptr;

    if constexpr ((ArchTag == TILINGKEY_TPL_A3) || (ArchTag == TILINGKEY_TPL_A2)) {
        NotifyDispatch<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(context, KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    } else if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        NotifyDispatchA5<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(context, KERNELS_ARGS_CALL_ALL2ALL());
        opKernel.Process();
    }
}
