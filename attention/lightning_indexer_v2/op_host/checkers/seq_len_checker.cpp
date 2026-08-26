/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "seq_len_checker_indexer_v2.h"

#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {
namespace {
bool CheckOptionalDtypes(const IndexerV2CheckerInfo &info)
{
    return info.quantized || info.arch35;
}
} // namespace

ge::graphStatus SeqLenChecker::CheckOneDimTensor(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                  const char *name, int64_t expectedSize) const
{
    if (!tensor.IsPresent()) {
        return ge::GRAPH_SUCCESS;
    }
    if (tensor.tensor->GetShapeSize() != expectedSize) {
        const std::string actual = ShapeToString(tensor);
        const std::string reason = "The tensor must contain " + std::to_string(expectedSize) + " elements";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const IndexerV2CheckerInfo &info) const
{
    const bool checkAllOptionalDtypes = CheckOptionalDtypes(info);
    if (info.layoutK == CheckerLayout::TND && (info.quantized || info.arch35) &&
        CheckDtype(info, info.cuSeqlensK, "cu_seqlens_k", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.sequsedK.IsPresent() &&
        (info.quantized || info.arch35 || info.layoutK == CheckerLayout::BSND) &&
        CheckDtype(info, info.sequsedK, "seqused_k", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQ == CheckerLayout::TND && checkAllOptionalDtypes &&
        CheckDtype(info, info.cuSeqlensQ, "cu_seqlens_q", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.sequsedQ.IsPresent() && checkAllOptionalDtypes &&
        CheckDtype(info, info.sequsedQ, "seqused_q", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.cmpResidualK.IsPresent() && checkAllOptionalDtypes &&
        CheckDtype(info, info.cmpResidualK, "cmp_residual_k", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.outputIdxOffset.IsPresent() && info.arch35 &&
        CheckDtype(info, info.outputIdxOffset, "output_idx_offset", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckParaExistence(const IndexerV2CheckerInfo &info) const
{
    if (info.layoutK == CheckerLayout::TND && !info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                 "cu_seqlens_k must be provided when layout_k is TND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::BSND && info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                 "cu_seqlens_k must not be provided when layout_k is BSND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::PA_BBND && (info.quantized || info.arch35) &&
        info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                 "cu_seqlens_k must not be provided when layout_k is PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQ == CheckerLayout::TND && !info.cuSeqlensQ.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                 "cu_seqlens_q must be provided when layout_q is TND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQ == CheckerLayout::BSND && (info.quantized || info.arch35) &&
        info.cuSeqlensQ.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                 "cu_seqlens_q must not be provided when layout_q is BSND");
        return ge::GRAPH_FAILED;
    }
    if (info.quantized || info.arch35) {
        const bool needResidual = info.cmpRatio != 1 && info.maskMode != 0;
        if (needResidual != info.cmpResidualK.IsPresent()) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                info.opName, "cmp_residual_k",
                "cmp_residual_k must be provided if and only if cmp_ratio is not 1 and mask_mode is not 0");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckMultiPara(const IndexerV2CheckerInfo &info) const
{
    if (info.layoutQ == CheckerLayout::TND) {
        const int64_t cuQSize = info.cuSeqlensQ.tensor->GetShapeSize();
        if (cuQSize <= 1) {
            const std::string actual = std::to_string(cuQSize);
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(info.opName, "cu_seqlens_q", actual.c_str(),
                                                      "The shape size of cu_seqlens_q must be greater than 1");
            return ge::GRAPH_FAILED;
        }
        if (info.layoutK == CheckerLayout::PA_BBND &&
            cuQSize != info.sequsedK.tensor->GetShapeSize() + 1) {
            const std::string actual = ShapeToString(info.cuSeqlensQ) + " and " + ShapeToString(info.sequsedK);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                info.opName, "cu_seqlens_q and seqused_k", actual.c_str(),
                "The size of cu_seqlens_q must equal the size of seqused_k plus 1");
            return ge::GRAPH_FAILED;
        }
        if (info.layoutK == CheckerLayout::TND &&
            cuQSize != info.cuSeqlensK.tensor->GetShapeSize()) {
            const std::string actual = ShapeToString(info.cuSeqlensQ) + " and " + ShapeToString(info.cuSeqlensK);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(info.opName, "cu_seqlens_q and cu_seqlens_k", actual.c_str(),
                                                   "cu_seqlens_q and cu_seqlens_k must have the same size");
            return ge::GRAPH_FAILED;
        }
    }
    if (info.quantized && info.layoutQ == CheckerLayout::BSND &&
        CheckOneDimTensor(info, info.sequsedQ, "seqused_q", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::PA_BBND &&
        CheckOneDimTensor(info, info.sequsedK, "seqused_k", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.quantized && info.layoutK == CheckerLayout::BSND &&
        CheckOneDimTensor(info, info.sequsedK, "seqused_k", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.cmpResidualK.IsPresent() && CheckOptionalDtypes(info) &&
        CheckOneDimTensor(info, info.cmpResidualK, "cmp_residual_k", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace indexer_v2_checker
} // namespace optiling
