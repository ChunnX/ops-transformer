/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SINGLE_CORE_CONTEXT_H
#define MEGA_MOE_SINGLE_CORE_CONTEXT_H

#include "kernel_operator.h"
#include "mega_moe_base.h"

namespace MegaMoeImpl {

// AIV logical thread information supplied by the caller. threadIdx must be a
// zero-based, contiguous index in [0, threadNum), independent of physical AIV IDs.
struct AivThreadContext {
    uint32_t threadIdx;
    uint32_t threadNum;
};

struct MoeTopologyContext {
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
};

struct TokenShapeContext {
    int32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
};

// Stable information shared by the dispatch preparation stage.
struct DispatchPrepareContext {
    AivThreadContext thread;
    MoeTopologyContext topology;
    TokenShapeContext tokenShape;
};

__aicore__ inline void TilingByThreadContext(int32_t totalLen, int32_t &threadLen, int32_t &threadOffset,
    uint32_t threadIdx, uint32_t threadNum, int32_t align = ALIGN_32)
{
    if (threadNum == 0U || threadIdx >= threadNum) {
        threadLen = 0;
        threadOffset = 0;
        return;
    }
    int32_t lenPerThread = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), threadNum);
    int32_t lenPerThreadAlign =
        Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerThread), static_cast<uint32_t>(align));
    threadLen = lenPerThreadAlign;
    threadOffset = static_cast<int32_t>(threadIdx) * lenPerThreadAlign;
    if (threadOffset + threadLen >= totalLen) {
        threadLen = totalLen - threadOffset;
    }
    if (threadOffset >= totalLen) {
        threadLen = 0;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SINGLE_CORE_CONTEXT_H
