/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN
 * Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not
 * use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#include "common_checker_indexer_v2.h"

#include <utility>
#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {
namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t HEAD_GROUP_LIMIT = 64;
constexpr int64_t TOPK_2K = 2048;
constexpr int64_t TOPK_8K = 8192;
constexpr int64_t TOPK_STEP = 1024;
constexpr int64_t MASK_MODE_DEFAULT = 0;
constexpr int64_t MASK_MODE_RIGHT_DOWN_CAUSAL = 3;
constexpr int64_t CMP_RATIO_LOWER_BOUND = 1;
constexpr int64_t CMP_RATIO_UPPER_BOUND = 128;
constexpr int64_t METADATA_SIZE = 1024;

bool SameDim(const gert::Shape &lhs, uint32_t lhsIndex, const gert::Shape &rhs, uint32_t rhsIndex)
{
    return lhs.GetDim(lhsIndex) == rhs.GetDim(rhsIndex);
}

ge::graphStatus LogShapeMismatch(const IndexerV2CheckerInfo &info, const char *names, const std::string &actual,
                                 const char *reason)
{
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(info.opName, names, actual.c_str(), reason);
    return ge::GRAPH_FAILED;
}
} // namespace

ge::graphStatus CommonChecker::CheckParaExistence(const IndexerV2CheckerInfo &info) const
{
    const std::pair<const CheckerTensor *, const char *> required[] = {
        {&info.query, "q"}, {&info.key, "k"}, {&info.weights, "w"}, {&info.sparseIndices, "sparse_indices"}};
    for (const auto &item : required) {
        if (!item.first->IsPresent() || item.first->desc == nullptr || item.first->GetShape() == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, item.second,
                                                     "Required tensor, descriptor and shape must be provided");
            return ge::GRAPH_FAILED;
        }
    }
    if (info.arch35 && info.returnValue == 1 &&
        (!info.sparseValues.IsPresent() || info.sparseValues.desc == nullptr ||
         info.sparseValues.GetShape() == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "sparse_values",
                                                 "sparse_values must be provided when return_value is 1");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQText.empty() || info.layoutKText.empty()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "layout_q and layout_k",
                                                 "layout_q and layout_k must be provided");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckTensorBasics(const IndexerV2CheckerInfo &info) const
{
    return CheckDtype(info, info.sparseIndices, "sparse_indices", {ge::DT_INT32});
}

ge::graphStatus CommonChecker::CheckLayoutAttrs(const IndexerV2CheckerInfo &info) const
{
    if (info.layoutQ != CheckerLayout::BSND && info.layoutQ != CheckerLayout::TND) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "layout_q", info.layoutQText.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK != CheckerLayout::BSND && info.layoutK != CheckerLayout::TND &&
        info.layoutK != CheckerLayout::PA_BBND) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "layout_k", info.layoutKText.c_str(), "BSND, TND or PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK != CheckerLayout::PA_BBND && info.layoutQText != info.layoutKText) {
        const std::string actual = info.layoutQText + " and " + info.layoutKText;
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "layout_q and layout_k", actual.c_str(),
                                               "layout_q and layout_k must be equal outside PagedAttention");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckTopkAttr(const IndexerV2CheckerInfo &info) const
{
    if (info.topk <= 0 || info.topk > TOPK_8K) {
        const std::string actual = std::to_string(info.topk);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "topk", actual.c_str(), "topk must be in [1, 8192]");
        return ge::GRAPH_FAILED;
    }
    if (!info.arch35 && info.quantized && info.topk > TOPK_2K) {
        const std::string actual = std::to_string(info.topk);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "topk", actual.c_str(),
                                              "topk must be in [1, 2048] on A2/A3 for QuantLightningIndexerV2");
        return ge::GRAPH_FAILED;
    }
    if (!info.arch35 && !info.quantized && info.topk > TOPK_2K && info.topk % TOPK_STEP != 0) {
        const std::string actual = std::to_string(info.topk);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "topk", actual.c_str(),
                                              "topk above 2048 must be a multiple of 1024 on A2/A3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMaskAndCmpRatioAttrs(const IndexerV2CheckerInfo &info) const
{
    if (info.maskMode != MASK_MODE_DEFAULT && info.maskMode != MASK_MODE_RIGHT_DOWN_CAUSAL) {
        const std::string actual = std::to_string(info.maskMode);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "mask_mode", actual.c_str(), "0 or 3");
        return ge::GRAPH_FAILED;
    }
    if (info.cmpRatio < CMP_RATIO_LOWER_BOUND || info.cmpRatio > CMP_RATIO_UPPER_BOUND) {
        const std::string actual = std::to_string(info.cmpRatio);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "cmp_ratio", actual.c_str(),
                                              "cmp_ratio must be in [1, 128]");
        return ge::GRAPH_FAILED;
    }
    if (!info.arch35 && info.quantized && (info.cmpRatio & (info.cmpRatio - 1)) != 0) {
        const std::string actual = std::to_string(info.cmpRatio);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "cmp_ratio", actual.c_str(),
                                              "cmp_ratio must be a power of two on A2/A3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckOtherAttrs(const IndexerV2CheckerInfo &info) const
{
    if ((info.quantized || info.arch35) && info.maxSeqlenQ < -1) {
        const std::string actual = std::to_string(info.maxSeqlenQ);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "max_seqlen_q", actual.c_str(), ">= -1");
        return ge::GRAPH_FAILED;
    }
    if (info.arch35 && info.returnValue != 0 && info.returnValue != 1) {
        const std::string actual = std::to_string(info.returnValue);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "return_value", actual.c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }
    if (!info.arch35 && info.quantized && info.returnValue != 0) {
        const std::string actual = std::to_string(info.returnValue);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "return_value", actual.c_str(),
                                              "return_value must be 0 on A2/A3 for QuantLightningIndexerV2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAttrs(const IndexerV2CheckerInfo &info) const
{
    if (CheckLayoutAttrs(info) != ge::GRAPH_SUCCESS || CheckTopkAttr(info) != ge::GRAPH_SUCCESS ||
        CheckMaskAndCmpRatioAttrs(info) != ge::GRAPH_SUCCESS || CheckOtherAttrs(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const IndexerV2CheckerInfo &info) const
{
    if (CheckTensorBasics(info) != ge::GRAPH_SUCCESS || CheckAttrs(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t qRank = info.layoutQ == CheckerLayout::BSND ? 4U : 3U;
    const uint32_t kRank = info.layoutK == CheckerLayout::TND ? 3U : 4U;
    if (CheckRank(info, info.query, "q", qRank) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.key, "k", kRank) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.weights, "w", qRank - 1U) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.sparseIndices, "sparse_indices", qRank) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.arch35 && info.returnValue == 1 &&
        CheckRank(info, info.sparseValues, "sparse_values", qRank) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.outputIdxOffset.IsPresent() && (info.quantized || info.arch35) &&
        CheckRank(info, info.outputIdxOffset, "output_idx_offset", qRank - 1U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const IndexerV2CheckerInfo &info) const
{
    if (info.kHeads != 1 || info.headDim != HEAD_DIM) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "k head count must be 1 and head_dim must be 128");
    }
    if (info.arch35 && info.qHeads / info.kHeads > HEAD_GROUP_LIMIT) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "q head count divided by k head count must not exceed 64");
    }
    if (!info.arch35 && info.quantized && info.qHeads / info.kHeads != HEAD_GROUP_LIMIT) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "q head count divided by k head count must equal 64 on A2/A3");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMainShapes(const IndexerV2CheckerInfo &info) const
{
    const gert::Shape &q = *info.query.GetShape();
    const gert::Shape &k = *info.key.GetShape();
    const gert::Shape &w = *info.weights.GetShape();
    const uint32_t qS = info.layoutQ == CheckerLayout::BSND ? 1U : 0U;
    const uint32_t qN = info.layoutQ == CheckerLayout::BSND ? 2U : 1U;
    const uint32_t qD = info.layoutQ == CheckerLayout::BSND ? 3U : 2U;
    const uint32_t kN = info.layoutK == CheckerLayout::TND ? 1U : 2U;
    const uint32_t kD = info.layoutK == CheckerLayout::TND ? 2U : 3U;
    const uint32_t wS = info.layoutQ == CheckerLayout::BSND ? 1U : 0U;
    const uint32_t wN = info.layoutQ == CheckerLayout::BSND ? 2U : 1U;
    if (!SameDim(q, qS, w, wS) || !SameDim(q, qN, w, wN) || !SameDim(q, qD, k, kD) ||
        (info.layoutQ == CheckerLayout::BSND && !SameDim(q, 0, w, 0))) {
        const std::string actual =
            ShapeToString(info.query) + ", " + ShapeToString(info.key) + " and " + ShapeToString(info.weights);
        return LogShapeMismatch(info, "q, k and w", actual,
                                "q and w must share sequence/head axes, and q and k must share head_dim");
    }
    if (!info.quantized && info.layoutQ == CheckerLayout::BSND && info.layoutK == CheckerLayout::BSND &&
        !SameDim(q, 0, k, 0)) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "Batch dimensions of BSND q and k must be equal");
    }
    if (k.GetDim(kN) != 1) {
        const std::string actual = ShapeToString(info.key);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "k", actual.c_str(), "The head count of k must be 1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckOutputShapes(const IndexerV2CheckerInfo &info) const
{
    const gert::Shape &q = *info.query.GetShape();
    const gert::Shape &out = *info.sparseIndices.GetShape();
    const uint32_t outN = info.layoutQ == CheckerLayout::BSND ? 2U : 1U;
    if (out.GetDim(0) != q.GetDim(0) || (info.layoutQ == CheckerLayout::BSND && out.GetDim(1) != q.GetDim(1)) ||
        out.GetDim(outN) != info.kHeads || out.GetDim(outN + 1U) != info.topk) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.sparseIndices);
        return LogShapeMismatch(info, "q and sparse_indices", actual,
                                "sparse_indices must be [B,S1,N2,topk] or [T1,N2,topk]");
    }
    if (info.arch35 && info.returnValue == 1) {
        const gert::Shape &values = *info.sparseValues.GetShape();
        if ((info.layoutQ == CheckerLayout::TND && values.GetDim(0) != q.GetDim(0)) ||
            values.GetDim(outN) != info.kHeads || values.GetDim(outN + 1U) != info.topk) {
            const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.sparseValues);
            return LogShapeMismatch(info, "q and sparse_values", actual,
                                    "sparse_values must match T1 (for TND), N2 and topk");
        }
    }
    if (info.arch35 && info.layoutQ == CheckerLayout::TND && info.outputIdxOffset.IsPresent() &&
        info.outputIdxOffset.GetShape()->GetDim(0) != q.GetDim(0)) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.outputIdxOffset);
        return LogShapeMismatch(info, "q and output_idx_offset", actual,
                                "TND q and output_idx_offset must have the same dim 0");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const IndexerV2CheckerInfo &info) const
{
    if (CheckMainShapes(info) != ge::GRAPH_SUCCESS || CheckOutputShapes(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.metadata.IsPresent() && info.metadata.tensor->GetShapeSize() != METADATA_SIZE) {
        const std::string actual = ShapeToString(info.metadata);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "metadata", actual.c_str(), "metadata size must be 1024");
        return ge::GRAPH_FAILED;
    }
    if ((info.quantized || info.arch35) && !info.metadata.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "metadata", "metadata must be provided");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace indexer_v2_checker
} // namespace optiling
