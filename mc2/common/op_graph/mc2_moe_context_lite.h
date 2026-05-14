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
 * \file mc2_moe_context_lite.h
 * \brief Lightweight Moe Context interface for Fusion Pass
 */
#ifndef MC2_MOE_CONTEXT_LITE_H_
#define MC2_MOE_CONTEXT_LITE_H_

#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 90000000

#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include "graph/ge_error_codes.h"
#include <string>
#include <vector>
#include <cstdint>

namespace ops {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct Mc2MoeContextLite {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};

ge::graphStatus GetMc2ContextDataLite(const std::string &nodeName, const std::string &groupEp,
                                      Mc2MoeContextLite &context, int64_t &hcclBuffSize);

} // namespace ops

#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#endif // MC2_MOE_CONTEXT_LITE_H_