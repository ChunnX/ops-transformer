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
 */

#ifndef MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H
#define MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H

#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"

namespace performance_info_check {

// ========== 互斥规则校验 ==========

inline ge::graphStatus CheckCompatibility(const char* nodeName, bool isLayered) {
    // performance 与 layered 互斥
    if (isLayered) {
        OPS_LOG_E(nodeName, "Cannot support isPerformance when comm_alg is hierarchy.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== shape 校验 ==========

inline ge::graphStatus CheckShape(
    gert::TilingContext* ctx, uint32_t inputIndex,
    uint32_t epWorldSize, const char* nodeName) {
    const gert::StorageShape* shape = ctx->GetOptionalInputShape(inputIndex);
    if (shape == nullptr) {
        OPS_LOG_E(nodeName, "performanceInfo is null.");
        return ge::GRAPH_FAILED;
    }

    // dim0 = epWorldSize
    if (shape->GetStorageShape().GetDimNum() != moe_dispatch_defs::ONE_DIM) {
        OPS_LOG_E(nodeName, "performanceInfo dim must be 1, but current dim num is %lu.",
            shape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (shape->GetStorageShape().GetDim(0) != static_cast<int64_t>(epWorldSize)) {
        OPS_LOG_E(nodeName, "performanceInfo's dim0 not equal to epWorldSize, "
            "performanceInfo's dim0 is %ld, epWorldSize is %u.",
            shape->GetStorageShape().GetDim(0), epWorldSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== dtype 校验 ==========

inline ge::graphStatus CheckDtype(
    const ge::TensorDesc* desc, const char* nodeName) {
    if (desc == nullptr) {
        OPS_LOG_E(nodeName, "performanceInfo desc is null.");
        return ge::GRAPH_FAILED;
    }

    // performanceInfo dtype: float 或 int32
    ge::DataType dtype = desc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_INT32) {
        OPS_LOG_E(nodeName, "performanceInfo dtype should be float or int32, but got %s.",
            Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace performance_info_check

#endif // MOE_DISPATCH_PERFORMANCE_INFO_CHECKER_H
