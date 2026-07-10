/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_context.h
 * \brief Hcomm 通信上下文结构体，Host 侧构造后通过 GM 下发到 kernel
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace Dista {
namespace AivComm {

static constexpr uint32_t HCOMM_MAX_RANK_NUM = 64;
static constexpr uint32_t HCOMM_WORKSPACE_SIZE = 512U;
static constexpr uint32_t HCOMM_CHANNEL_NOTIFY_NUM = 3U;

struct CommChannelContext {
    uint64_t channelHandles[HCOMM_MAX_RANK_NUM];
    uint64_t remoteBufferAddrs[HCOMM_MAX_RANK_NUM];
    uint64_t localBufferAddr;
    uint64_t syncBufferAddr;
    uint64_t remoteSyncAddrs[HCOMM_MAX_RANK_NUM];
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t notifyNum;
};

struct CommContext {
    __gm__ CommChannelContext* commCtx;
    uint32_t rankSize;
    uint32_t myRankId;
};

struct SyncBufferContext {
    uint32_t rankSize;
    uint32_t myRankId;
    uint64_t localSyncBufferAddr;
    uint64_t remoteSyncBufferAddrs[HCOMM_MAX_RANK_NUM];
};

} // namespace AivComm
} // namespace Dista
