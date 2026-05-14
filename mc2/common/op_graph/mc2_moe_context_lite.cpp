/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_moe_context_lite.cpp
 * \brief Lightweight Moe Context implementation for Fusion Pass
 */
#include "mc2_moe_context_lite.h"

#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include "mc2_moe_context_manager.h"

namespace ops {

ge::graphStatus GetMc2ContextDataLite(const std::string &nodeName, const std::string &groupEp,
                                      Mc2MoeContextLite &context, int64_t &hcclBuffSize)
{
    Mc2MoeContext fullContext{0};
    auto status = Mc2MoeContextManager::GetMc2ContextData(nodeName, groupEp, fullContext, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    context.epRankId = fullContext.epRankId;
    context.rankSizePerServer = fullContext.rankSizePerServer;
    context.kfcContextAddr = fullContext.kfcContextAddr;
    for (uint32_t i = 0; i < HCCL_MAX_RANK_SIZE; ++i) {
        context.epHcclBuffer_[i] = fullContext.epHcclBuffer_[i];
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace ops

#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION