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

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE]; // 支持ROCE或者URMA
};

__aicore__ inline GM_ADDR GetDispatchStateAddrByRankId(__gm__ Mc2MoeContext *ctx, uint32_t rankId)
{
    return (GM_ADDR)ctx->epHcclBuffer_[rankId];
}

__aicore__ inline GM_ADDR GetDispatchWinAddrByRankId(__gm__ Mc2MoeContext *ctx, uint32_t rankId, uint64_t winOffset)
{
    return (GM_ADDR)ctx->epHcclBuffer_[rankId] + winOffset;
}

__aicore__ inline uint64_t GetCommHandle(__gm__ Mc2MoeContext *ctx, uint32_t localRankId, uint32_t rankId)
{
    uint32_t index = rankId > localRankId ? rankId - 1 : rankId;
    return ctx->hcommHandle_[index];
}

#endif  // MOE_EP_DISPATCH_BASE_H