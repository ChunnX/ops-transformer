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
 * \file fused_quant_lightning_indexer_manage_infershape.cpp
 * \brief Shape and data type inference for FusedQuantLightningIndexerManage.
 */

#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t QUERY = 2;
constexpr size_t BLOCK_TABLE = 5;
constexpr size_t CACHE_POOL = 12;
constexpr int64_t TOPK = 2048;
constexpr int64_t MISS_CAPACITY = 32768;
} // namespace

static ge::graphStatus InferFusedQuantLightningIndexerManageShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(QUERY) == nullptr ||
        context->GetInputShape(BLOCK_TABLE) == nullptr || context->GetInputShape(CACHE_POOL) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalQueries = context->GetInputShape(QUERY)->GetDim(0);
    const int64_t batch = context->GetInputShape(BLOCK_TABLE)->GetDim(0);
    for (size_t i = 0; i < 7; ++i) {
        if (context->GetOutputShape(i) == nullptr)
            return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape({totalQueries, 1, TOPK});
    *context->GetOutputShape(1) = gert::Shape({totalQueries, 1, TOPK});
    *context->GetOutputShape(2) = gert::Shape({totalQueries});
    *context->GetOutputShape(3) = gert::Shape({batch, MISS_CAPACITY});
    *context->GetOutputShape(4) = gert::Shape({batch, MISS_CAPACITY});
    *context->GetOutputShape(5) = gert::Shape({batch});
    *context->GetOutputShape(6) = *context->GetInputShape(CACHE_POOL);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferFusedQuantLightningIndexerManageDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr)
        return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 7; ++i)
        context->SetOutputDataType(i, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedQuantLightningIndexerManage)
    .InferShape(InferFusedQuantLightningIndexerManageShape)
    .InferDataType(InferFusedQuantLightningIndexerManageDataType);
} // namespace ops
