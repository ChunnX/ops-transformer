/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file performance_info_checker.h
 * \brief MOE Distribute Dispatch PerformanceInfo 可选特性校验
 *
 * 设计原则（architecture_v0.md §5.4）：
 *   仅做互斥规则校验 + shape 精确值校验。
 *   dim/dtype/format 基础校验统一在阶段 E 通过 TensorCheckRule 处理。
 */

#ifndef MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H
#define MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H

#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"

namespace performance_info_check {

// ========== 互斥规则校验 ==========
inline ge::graphStatus CheckCompatibility(const char *nodeName, bool isLayered)
{
    // performance 与 layered 互斥
    if (isLayered) {
        OPS_LOG_E(nodeName, "Cannot support isPerformance when comm_alg is hierarchy.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== shape 校验 ==========
// dim0 = epWorldSize
inline ge::graphStatus CheckShape(gert::TilingContext *ctx, uint32_t inputIndex, uint32_t epWorldSize,
                                  const char *nodeName)
{
    const gert::StorageShape *shape = ctx->GetOptionalInputShape(inputIndex);
    if (shape == nullptr) {
        OPS_LOG_E(nodeName, "performanceInfo is null.");
        return ge::GRAPH_FAILED;
    }

    if (shape->GetStorageShape().GetDimNum() != moe_dispatch_defs::ONE_DIM) {
        OPS_LOG_E(nodeName, "performanceInfo dim must be 1, but current dim num is %lu.",
                  shape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (shape->GetStorageShape().GetDim(0) != static_cast<int64_t>(epWorldSize)) {
        OPS_LOG_E(nodeName,
                  "performanceInfo's dim0 not equal to epWorldSize, "
                  "performanceInfo's dim0 is %ld, epWorldSize is %u.",
                  shape->GetStorageShape().GetDim(0), epWorldSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace performance_info_check

#endif // MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H
