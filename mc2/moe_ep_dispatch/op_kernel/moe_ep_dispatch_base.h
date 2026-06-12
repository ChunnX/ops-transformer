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
 * \file moe_ep_dispatch_base.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_BASE_H
#define MOE_EP_DISPATCH_BASE_H

using namespace AscendC;

constexpr uint64_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE]; // 支持ROCE或者URMA
};

// __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t align)
// {
//     return (x + align - 1) & ~(align - 1);
// }

__aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2MoeContext *ctx, uint32_t rankId)
{
    return (GM_ADDR)ctx->epHcclBuffer[rankId] + A5_MTE_STATE_WIN_SIZE;
}

__aicore__ inline GM_ADDR GetStateAddrByRankId(__gm__ Mc2MoeContext *ctx, uint32_t rankId, uint32_t dataState)
{
    return (GM_ADDR)ctx->epHcclBuffer[rankId] + dataState * WIN_STATE_OFFSET;
}

__aicore__ inline uint64_t GetCommHandle(__gm__ Mc2MoeContext *ctx, uint32_t localRankId, uint32_t rankId)
{
    uint32_t index = rankId > localRankId ? rankId - 1 : rankId;
    return ctx->hcommHandle_[index];
}

#endif  // MOE_EP_DISPATCH_BASE_H