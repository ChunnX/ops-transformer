/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "seq_len_checker_indexer_v2.h"

#include <limits>
#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {

ge::graphStatus SeqLenChecker::CheckOneDimTensor(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                  const char *name, int64_t expectedSize) const
{
    if (!tensor.IsPresent()) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckRank(info, tensor, name, 1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *shape = tensor.GetShape();
    if (shape->GetShapeSize() != expectedSize) {
        const std::string actual = ShapeToString(tensor);
        const std::string reason = "The tensor must contain " + std::to_string(expectedSize) + " elements";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const IndexerV2CheckerInfo &info) const
{
    if (CheckOneDimTensor(info, info.cuSeqlensQ, "cu_seqlens_q", info.batch + 1) != ge::GRAPH_SUCCESS ||
        CheckOneDimTensor(info, info.cuSeqlensK, "cu_seqlens_k", info.batch + 1) != ge::GRAPH_SUCCESS ||
        CheckOneDimTensor(info, info.sequsedQ, "seqused_q", info.batch) != ge::GRAPH_SUCCESS ||
        CheckOneDimTensor(info, info.sequsedK, "seqused_k", info.batch) != ge::GRAPH_SUCCESS ||
        CheckOneDimTensor(info, info.cmpResidualK, "cmp_residual_k", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.outputIdxOffset.IsPresent()) {
        const uint32_t expectedRank = info.layoutQ == CheckerLayout::BSND ? 3U : 2U;
        if (CheckRank(info, info.outputIdxOffset, "output_idx_offset", expectedRank) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckParaExistence(const IndexerV2CheckerInfo &info) const
{
    if ((info.layoutQ == CheckerLayout::TND) != info.cuSeqlensQ.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "cu_seqlens_q", "cu_seqlens_q must be provided only when layout_q is TND");
        return ge::GRAPH_FAILED;
    }
    if ((info.layoutK == CheckerLayout::TND) != info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "cu_seqlens_k", "cu_seqlens_k must be provided only when layout_k is TND");
        return ge::GRAPH_FAILED;
    }
    const bool needResidual = info.cmpRatio > 1 && info.maskMode == 3;
    if (needResidual != info.cmpResidualK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "cmp_residual_k",
            "cmp_residual_k must be provided if and only if cmp_ratio is greater than 1 and mask_mode is 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckCuSeqlensValues(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                     const char *name, int64_t total) const
{
    if (tensor.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t *data = tensor.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t count = tensor.tensor->GetShapeSize();
    if (count <= 0 || data[0] != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, "invalid",
                                               "The first cumulative sequence length must be 0");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 1; i < count; ++i) {
        if (data[i] < data[i - 1]) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, "not nondecreasing",
                                                   "Cumulative sequence lengths must be nonnegative and nondecreasing");
            return ge::GRAPH_FAILED;
        }
    }
    if (data[count - 1] != total) {
        const std::string actual = std::to_string(data[count - 1]);
        const std::string reason = "The final cumulative sequence length must equal " + std::to_string(total);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSeqUsedValues(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                   const CheckerTensor *cuSeqlens, const char *name,
                                                   int64_t maximum) const
{
    if (tensor.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t *data = tensor.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t *cuData =
        cuSeqlens == nullptr || cuSeqlens->tensor == nullptr ? nullptr : cuSeqlens->tensor->GetData<int32_t>();
    for (int64_t i = 0; i < tensor.tensor->GetShapeSize(); ++i) {
        const int64_t currentMaximum =
            cuData == nullptr ? maximum : static_cast<int64_t>(cuData[i + 1]) - static_cast<int64_t>(cuData[i]);
        if (data[i] < 0 || data[i] > currentMaximum) {
            const std::string actual = std::to_string(data[i]);
            const std::string reason = "Each value must be in [0, " + std::to_string(currentMaximum) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckMultiPara(const IndexerV2CheckerInfo &info) const
{
    if (CheckCuSeqlensValues(info, info.cuSeqlensQ, "cu_seqlens_q", info.qSeq) != ge::GRAPH_SUCCESS ||
        CheckCuSeqlensValues(info, info.cuSeqlensK, "cu_seqlens_k", info.kSeq) != ge::GRAPH_SUCCESS ||
        CheckSeqUsedValues(info, info.sequsedQ,
                           info.layoutQ == CheckerLayout::TND ? &info.cuSeqlensQ : nullptr, "seqused_q",
                           info.qSeq) != ge::GRAPH_SUCCESS ||
        CheckSeqUsedValues(info, info.sequsedK,
                           info.layoutK == CheckerLayout::TND ? &info.cuSeqlensK : nullptr, "seqused_k",
                           info.kSeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.cmpResidualK.tensor != nullptr) {
        const int32_t *data = info.cmpResidualK.tensor->GetData<int32_t>();
        if (data != nullptr) {
            for (int64_t i = 0; i < info.cmpResidualK.tensor->GetShapeSize(); ++i) {
                if (data[i] < 0 || data[i] >= info.cmpRatio) {
                    const std::string actual = std::to_string(data[i]);
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, "cmp_residual_k", actual.c_str(), "Each value must satisfy 0 <= value < cmp_ratio");
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    if (info.outputIdxOffset.tensor != nullptr) {
        const int32_t *data = info.outputIdxOffset.tensor->GetData<int32_t>();
        if (data != nullptr) {
            const int64_t count = info.outputIdxOffset.tensor->GetShapeSize();
            const int64_t maxIndex = info.kSeq * info.cmpRatio - 1;
            for (int64_t i = 0; i < count; ++i) {
                if (data[i] <= 0 || static_cast<int64_t>(data[i]) + maxIndex > std::numeric_limits<int32_t>::max()) {
                    const std::string actual = std::to_string(data[i]);
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        info.opName, "output_idx_offset", actual.c_str(),
                        "Each offset must be positive and the offset index must remain within INT32_MAX");
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace indexer_v2_checker
} // namespace optiling
