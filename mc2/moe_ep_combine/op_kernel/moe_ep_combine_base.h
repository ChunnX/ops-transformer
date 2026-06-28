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
 * \file moe_ep_combine_base.h
 * \brief
 */
 
#ifndef MOE_EP_COMBINE_BASE_H
#define MOE_EP_COMBINE_BASE_H
#include <cstdint>

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint64_t COMBINE_WIN_ADDR_ALIGN = 512UL;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE];
};


__aicore__ inline GM_ADDR GetCombineWinAddrByRankId(__gm__ Mc2MoeContext *ctx, uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
}

__aicore__ inline GM_ADDR GetCombineSyncNotifyAddr(__gm__ Mc2MoeContext *ctx, uint32_t rankId,
    uint64_t syncNotifyOffset, uint32_t senderRank)
{
    return (GM_ADDR)ctx->epHcclBuffer[rankId] + syncNotifyOffset +
           static_cast<uint64_t>(senderRank) * COMBINE_WIN_ADDR_ALIGN;
}

__aicore__ inline uint64_t GetCommHandle(__gm__ Mc2MoeContext *ctx, uint32_t localRankId, uint32_t rankId)
{
    uint32_t index = rankId > localRankId ? rankId - 1 : rankId;
    return ctx->hcommHandle[index];
}


#endif // MOE_EP_COMBINE_BASE_H