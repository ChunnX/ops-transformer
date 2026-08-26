/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "paged_attention_checker_indexer_v2.h"

#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const IndexerV2CheckerInfo &info) const
{
    const bool paged = info.layoutK == CheckerLayout::PA_BBND;
    if (paged != info.blockTable.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "block_table", "block_table must be provided if and only if layout_k is PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if (paged && !info.sequsedK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "seqused_k",
                                                 "seqused_k must be provided when layout_k is PA_BBND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const IndexerV2CheckerInfo &info) const
{
    if (info.layoutK != CheckerLayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckRank(info, info.blockTable, "block_table", 2) != ge::GRAPH_SUCCESS ||
        CheckNotEmpty(info, info.blockTable, "block_table") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.blockSize < 16 || info.blockSize > 1024 || info.blockSize % 16 != 0) {
        const std::string actual = std::to_string(info.blockSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "block_size", actual.c_str(),
                                               "block_size must be a multiple of 16 in [16, 1024]");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *keyShape = info.key.GetShape();
    if (keyShape->GetDim(0) <= 0) {
        const std::string actual = ShapeToString(info.key);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "k", actual.c_str(), "block_num must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckContiguous(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                        const char *name, uint32_t firstCheckedAxis) const
{
    const gert::Shape *shape = tensor.GetShape();
    if (!tensor.IsPresent() || tensor.stride == nullptr || tensor.stride->GetDimNum() == 0 || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t expected = 1;
    for (int64_t i = static_cast<int64_t>(shape->GetDimNum()) - 1; i >= static_cast<int64_t>(firstCheckedAxis); --i) {
        if (shape->GetDim(i) != 1 && tensor.stride->GetStride(i) != expected) {
            const std::string actual = std::to_string(tensor.stride->GetStride(i));
            const std::string reason =
                (firstCheckedAxis == 1U ? "Only axis 0 may be non-contiguous in PA_BBND; expected stride "
                                        : "The tensor must be contiguous; expected stride ") +
                std::to_string(expected) + " at axis " + std::to_string(i);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        expected *= shape->GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const IndexerV2CheckerInfo &info) const
{
    if (info.layoutK == CheckerLayout::PA_BBND) {
        const gert::Shape *blockShape = info.blockTable.GetShape();
        if (blockShape->GetDim(0) != info.batch || blockShape->GetDim(1) <= 0) {
            const std::string actual = ShapeToString(info.blockTable);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                info.opName, "block_table", actual.c_str(), "block_table must have shape [B, max_num_blocks_per_seq]");
            return ge::GRAPH_FAILED;
        }
    }
    const uint32_t firstAxis = info.layoutK == CheckerLayout::PA_BBND ? 1U : 0U;
    if (CheckContiguous(info, info.query, "q", 0U) != ge::GRAPH_SUCCESS ||
        CheckContiguous(info, info.key, "k", firstAxis) != ge::GRAPH_SUCCESS ||
        CheckContiguous(info, info.weights, "w", 0U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.quantized &&
        (CheckContiguous(info, info.queryDescale, "q_descale", 0U) != ge::GRAPH_SUCCESS ||
         CheckContiguous(info, info.keyDescale, "k_descale", firstAxis) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace indexer_v2_checker
} // namespace optiling
