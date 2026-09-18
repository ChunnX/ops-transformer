/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_FAST_WORKSPACE_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_FAST_WORKSPACE_H

#include <cstdint>

namespace fused_quant_li_manage_fast_workspace {
constexpr uint64_t ROUTES = 4U;
constexpr uint64_t TOPK = 2048U;
constexpr uint64_t UNION_CAPACITY = ROUTES * TOPK;
constexpr uint64_t THRESHOLD_STRIDE = 16U;
constexpr uint64_t TAIL_CAPACITY = 256U;
constexpr uint64_t ATTENTION_CAPACITY = TOPK;

constexpr uint64_t Align32(uint64_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}

constexpr uint64_t RoutePairOffset(uint64_t scoreStride, uint64_t batch)
{
    return scoreStride * batch;
}

constexpr uint64_t RouteThresholdOffset(uint64_t scoreStride, uint64_t batch)
{
    return RoutePairOffset(scoreStride, batch) + batch * UNION_CAPACITY * 2U * sizeof(int32_t);
}

constexpr uint64_t RouteCountOffset(uint64_t scoreStride, uint64_t batch)
{
    return RouteThresholdOffset(scoreStride, batch) + batch * ROUTES * THRESHOLD_STRIDE * sizeof(uint16_t);
}

constexpr uint64_t TotalBytes(uint64_t scoreStride, uint64_t batch)
{
    return Align32(RouteCountOffset(scoreStride, batch) + batch * ROUTES * sizeof(int32_t));
}
} // namespace fused_quant_li_manage_fast_workspace

#endif
