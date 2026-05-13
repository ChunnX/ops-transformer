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
 * \file elastic_info_checker.h
 * \brief MOE Distribute Dispatch ElasticInfo 可选特性校验
 */

#ifndef MOE_DISPATCH_ELASTIC_INFO_CHECKER_H
#define MOE_DISPATCH_ELASTIC_INFO_CHECKER_H

#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"

namespace elastic_info_check {

// ========== 互斥规则校验 ==========

inline ge::graphStatus CheckCompatibility(
    const char* nodeName, bool isLayered, uint32_t tpWorldSize) {
    // elastic 与 layered 互斥
    if (isLayered) {
        OPS_LOG_E(nodeName, "Cannot support elasticInfo when comm_alg is hierarchy.");
        return ge::GRAPH_FAILED;
    }

    // elastic 与 tp>1 互斥
    if (tpWorldSize > moe_dispatch_defs::TP_WORLD_SIZE_ONE) {
        OPS_LOG_E(nodeName, "Cannot support elasticInfo when tpWorldSize = %u > 1.", tpWorldSize);
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
        OPS_LOG_E(nodeName, "elasticInfo is null.");
        return ge::GRAPH_FAILED;
    }

    // dim0 = ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize
    // = 4 + 2 * epWorldSize
    int64_t expectedDim0 = static_cast<int64_t>(
        moe_dispatch_defs::ELASTIC_INFO_OFFSET +
        moe_dispatch_defs::RANK_LIST_NUM * epWorldSize);

    if (shape->GetStorageShape().GetDimNum() != moe_dispatch_defs::ONE_DIM) {
        OPS_LOG_E(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
            shape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (shape->GetStorageShape().GetDim(0) != expectedDim0) {
        OPS_LOG_E(nodeName, "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
            "elasticInfo's dim0 is %ld, epWorldSize is %u.",
            shape->GetStorageShape().GetDim(0), epWorldSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== dtype 校验 ==========

inline ge::graphStatus CheckDtype(
    const ge::TensorDesc* desc, const char* nodeName) {
    if (desc == nullptr) {
        OPS_LOG_E(nodeName, "elasticInfo desc is null.");
        return ge::GRAPH_FAILED;
    }

    // elasticInfo dtype: int32 或 int64
    ge::DataType dtype = desc->GetDataType();
    if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
        OPS_LOG_E(nodeName, "elasticInfo dtype should be int32 or int64, but got %s.",
            Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace elastic_info_check

#endif // MOE_DISPATCH_ELASTIC_INFO_CHECKER_H
