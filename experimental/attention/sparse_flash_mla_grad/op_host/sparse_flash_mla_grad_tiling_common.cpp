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
 * \file sparse_flash_mla_grad_tiling_common.cpp
 * \brief
 */

#include "sparse_flash_mla_grad_tiling_common.h"

namespace optiling {
namespace sfmg {

ge::graphStatus CheckAttentionInShape(gert::TilingContext *context)
{
    auto attentionInShape = context->GetInputShape(static_cast<size_t>(InputIndex::OUT));
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::Q));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to query %zu", attentionInShapeDim,
                  queryShapeDim);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < queryShapeDim - 1; i++) {
        if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context, "The dim %zu of attentionIn shape is invalid, got %ld, should be %ld", i,
                      attentionInShape->GetStorageShape().GetDim(i), queryShape->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }

    auto oriKVShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_KV));
    auto cmpKVShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_KV));
    if (oriKVShape != nullptr) {
        auto oriKVShapeDim = oriKVShape->GetStorageShape().GetDimNum();
        if (attentionInShapeDim != oriKVShapeDim) {
            OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to ori_kv %zu", 
                attentionInShapeDim, oriKVShapeDim);
        return ge::GRAPH_FAILED;
        }
        // check head_dim, attention D == value D
        if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
            oriKVShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
            OP_LOGE(context, "The head_dim of attentionIn shape is invalid, got %ld, should be %ld",
                    attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1),
                    oriKVShape->GetStorageShape().GetDim(queryShapeDim - 1));
            return ge::GRAPH_FAILED;
        }
    }
    if (cmpKVShape != nullptr) {
        auto cmpKVShapeDim = cmpKVShape->GetStorageShape().GetDimNum();
        if (attentionInShapeDim != cmpKVShapeDim) {
            OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to cmp_kv %zu", attentionInShapeDim,
                    cmpKVShapeDim);
            return ge::GRAPH_FAILED;
        }
        // check head_dim, attention D == value D
        if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
            cmpKVShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
            OP_LOGE(context, "The head_dim of attentionIn shape is invalid, got %ld, should be %ld",
                    attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1),
                    cmpKVShape->GetStorageShape().GetDim(queryShapeDim - 1));
            return ge::GRAPH_FAILED;
        }
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(gert::TilingContext *context)
{
    auto query = context->GetInputDesc(static_cast<size_t>(InputIndex::Q));
    auto attentionIn = context->GetInputDesc(static_cast<size_t>(InputIndex::OUT));
    OP_CHECK_IF(query == nullptr || attentionIn == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or attentionIn is nullptr."), return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(queryType != attentionInType,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "invalid attentionIn dtype should be same with query's dtype"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d, int64_t d2, int64_t n2)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0 || d2 == 0 || n2 == 0);
    OP_CHECK_IF(
        isShapeInValid,
        OP_LOGE(context, "input shape error, got 0 in tnd(%ld,%ld,%ld) or tnd(%ld,%ld,%ld)", t1, n1, d, t1, n1, d2),
        return ge::GRAPH_FAILED);

    auto g = n1 / n2;
    auto ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeValid(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckAttentionInDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

bool IsSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr), OP_LOGW("aShape or bShape is nullptr."), return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

} // namespace sfmg
} // namespace optiling
