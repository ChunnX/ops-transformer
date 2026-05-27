/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _HPP
#define _HPP

#include <cstdint>

#ifdef __CCE_AICORE__
#include "kernel_operator.h"
#endif

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;

struct CommContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t hcclContextAddr;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};

#ifdef __CCE_AICORE__
inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}
#endif

#endif // _HPP
