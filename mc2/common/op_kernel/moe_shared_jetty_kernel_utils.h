/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_SHARED_JETTY_KERNEL_UTILS_H
#define MOE_SHARED_JETTY_KERNEL_UTILS_H

#include "moe_shared_jetty_context.h"

namespace Mc2Kernel {
template <typename BatchHandleT>
__aicore__ inline void CheckSharedJettyQueueDepth(const BatchHandleT &batch)
{
    uint32_t sqDepth = batch.handle.sqContext.contextInfo.ubJfs.sqDepth;
    uint32_t cqDepth = batch.handle.cqContext.contextInfo.ubJfc.cqDepth;
    if (sqDepth != Mc2Aclnn::MOE_SHARED_JETTY_SQ_DEPTH || cqDepth != Mc2Aclnn::MOE_SHARED_JETTY_CQ_DEPTH) {
        AscendC::printf("MoE shared Jetty depth mismatch, sq=%u, cq=%u\n", sqDepth, cqDepth);
        AscendC::Trap();
    }
}
} // namespace Mc2Kernel

#endif // MOE_SHARED_JETTY_KERNEL_UTILS_H
