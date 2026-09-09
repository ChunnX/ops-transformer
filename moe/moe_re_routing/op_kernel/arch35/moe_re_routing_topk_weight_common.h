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
 * \file moe_re_routing_topk_weight_common.h
 * \brief Shared constants and path predicates for the arch35 topk-weight path.
 * The header intentionally contains no AscendC object so it can be included by both host tiling and kernel code.
 */
#ifndef MOE_RE_ROUTING_TOPK_WEIGHT_COMMON_H
#define MOE_RE_ROUTING_TOPK_WEIGHT_COMMON_H

#include <cstdint>

namespace MoeReRoutingTopk {
#if defined(__NPU_ARCH__)
#define MOE_RE_ROUTING_TOPK_AICORE __aicore__
#else
#define MOE_RE_ROUTING_TOPK_AICORE
#endif

enum class CopyPath : uint8_t {
    GROUP,
    SMALL_QUEUE
};
constexpr int64_t INDEX_UB_SIZE = 256;
constexpr int64_t TOPK_GROUP_EXPERTS = 256;
constexpr int64_t TOPK_OFFSET_UB_SIZE = TOPK_GROUP_EXPERTS * sizeof(int64_t);
constexpr int64_t TOPK_COPY_UB_SIZE = INDEX_UB_SIZE * 2 * sizeof(float);

MOE_RE_ROUTING_TOPK_AICORE inline constexpr bool UseDirectH1Path(const int64_t tokenSize, const int64_t expertNum)
{
    return tokenSize == 1 && expertNum <= TOPK_GROUP_EXPERTS;
}

MOE_RE_ROUTING_TOPK_AICORE inline constexpr bool UseSmallQueue(const int64_t tokenSize, const int64_t rows)
{
    return tokenSize > 1 && rows > 0 && rows <= INDEX_UB_SIZE * 2;
}

MOE_RE_ROUTING_TOPK_AICORE inline constexpr CopyPath SelectPath(const int64_t tokenSize, const int64_t rows)
{
    return UseSmallQueue(tokenSize, rows) ? CopyPath::SMALL_QUEUE : CopyPath::GROUP;
}
} // namespace MoeReRoutingTopk

#undef MOE_RE_ROUTING_TOPK_AICORE

#endif // MOE_RE_ROUTING_TOPK_WEIGHT_COMMON_H
