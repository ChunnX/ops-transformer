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
 * \file moe_ep_dispatch_tiling.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_TILING_H
#define MOE_EP_DISPATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/moe_ep_exception_dump_defs.h"

struct MoeEpCommonTilingData {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t numExperts;
    uint32_t numLocalExperts;
    uint32_t numTokens;
    uint32_t hidden;
    uint32_t topK;
    uint32_t numMaxTokensPerRank;
    uint32_t expertAlignment;
};

struct MoeEpDispatchWindowLayout {
    uint64_t cntWinStateOffset;
    uint64_t slotWinStateOffset;
    uint64_t payloadWinStateOffset; // payload 发送状态位区偏移（slot 状态区之后）
    uint64_t winDataOffset;
    uint64_t payloadStashWinOffset;
    uint64_t completionFlagSourceWinOffset;
};

struct MoeEpDispatchWorkspaceLayout {
    uint32_t tileTokens;
    uint32_t tileCount;
    uint64_t sendCntOffset;       // Final rank/expert totals; tile count rows start at workspace offset zero.
    uint64_t tilePrefixOffset;    // [tileCount + 1][align512(P * sizeof(int32_t))].
    uint64_t expertPartialOffset; // [aivNum][align512(E * sizeof(int32_t))].
    uint64_t tileReadyOffset;     // One local 32-byte ready block per tile, never a remote completion flag.
    uint64_t srcTokenListBytes;        // 发送到每个对端卡的 token索引表 字节数（512B 对齐，int32）
    uint64_t dstRankInfoOffset;        // dstRank 区偏移（sendCnt 区之后），布局 [BS][K] int16
    uint64_t srcTokenTableOffset;      // srcTokenTable 区偏移（dstRank 区之后），布局 [ep][nmtAlign] int32
};

struct MoeEpDispatchInfo {
    MoeEpCommonTilingData cfg;
    MoeEpDispatchWindowLayout window;
    MoeEpDispatchWorkspaceLayout workspace;
    MoeEpDumpMetadata dumpMetadata;
    uint64_t hostPinnedCounterAddr;
    uint64_t totalWinSizeEp;
    uint64_t totalUbSize;
    uint32_t scalesBytes;
    uint32_t perSlotBytes;
    uint32_t metaSlotBytes; // stash 元数据 slot 字节数（scales+topk+weights+pad，无 hidden）
    uint32_t doCpuSync;
    uint32_t isCached;
    uint32_t isTopkWeights;
    uint32_t isMxQuant;
    uint32_t dispatchNotifyCount;
    uint32_t aivNum;
};

struct MoeEpDispatchTilingData {
    MoeEpDispatchInfo moeEpDispatchInfo;
};

#endif // MOE_EP_DISPATCH_TILING_H
