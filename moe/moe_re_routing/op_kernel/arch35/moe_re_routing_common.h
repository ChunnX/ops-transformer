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
 * \file moe_re_routing_common.h
 * \brief Common constants shared by host tiling and arch35 kernels
 */
#ifndef MOE_RE_ROUTING_COMMON_H
#define MOE_RE_ROUTING_COMMON_H

#include <cstdint>

namespace MoeReRouting {
constexpr int32_t HAS_TOPK_WEIGHT = 1;
}

// Keep topk-weight constants in the lightweight common header.  Host tiling can
// include this header without pulling in AscendC kernel APIs.
namespace MoeReRoutingTopk {
constexpr int64_t INDEX_UB_SIZE = 256;
constexpr int64_t TOPK_GROUP_EXPERTS = 256;
constexpr int64_t TOPK_OFFSET_UB_SIZE = TOPK_GROUP_EXPERTS * sizeof(int64_t);
} // namespace MoeReRoutingTopk

#endif // MOE_RE_ROUTING_COMMON_H
