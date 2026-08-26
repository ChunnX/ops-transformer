/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "quant_checker_qliv2.h"

#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {
namespace {
constexpr int64_t MODE_FP8 = 1;
constexpr int64_t MODE_INT8 = 2;
constexpr int64_t MODE_MXFP8 = 3;
constexpr int64_t MODE_HIFLOAT8 = 4;
constexpr int64_t MODE_MXFP4 = 5;
constexpr int64_t MX_GROUP = 64;
constexpr int64_t MX_PACK = 2;
}

ge::graphStatus QuantChecker::CheckParaExistence(const IndexerV2CheckerInfo &info) const
{
    if (!info.queryDescale.IsPresent() || info.queryDescale.desc == nullptr ||
        info.queryDescale.GetShape() == nullptr || !info.keyDescale.IsPresent() ||
        info.keyDescale.desc == nullptr || info.keyDescale.GetShape() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "q_descale and k_descale",
                                                 "q_descale and k_descale, including descriptors and shapes, are required");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSinglePara(const IndexerV2CheckerInfo &info) const
{
    if (!info.arch35 && info.layoutK != CheckerLayout::PA_BBND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_k", info.layoutKText.c_str(),
                                               "layout_k must be PA_BBND on A2/A3");
        return ge::GRAPH_FAILED;
    }
    if ((!info.arch35 && info.quantMode != MODE_INT8) ||
        (info.arch35 && (info.quantMode < MODE_FP8 || info.quantMode > MODE_MXFP4))) {
        const std::string actual = std::to_string(info.quantMode);
        const char *expected = info.arch35 ? "1, 2, 3, 4 or 5" : "2";
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "quant_mode", actual.c_str(), expected);
        return ge::GRAPH_FAILED;
    }

    ge::DataType inputType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType scaleType = ge::DT_FLOAT;
    ge::DataType weightType = ge::DT_FLOAT;
    if (info.quantMode == MODE_INT8) {
        inputType = ge::DT_INT8;
        scaleType = ge::DT_FLOAT16;
        weightType = ge::DT_FLOAT16;
    } else if (info.quantMode == MODE_MXFP8) {
        scaleType = ge::DT_FLOAT8_E8M0;
    } else if (info.quantMode == MODE_HIFLOAT8) {
        inputType = ge::DT_HIFLOAT8;
    } else if (info.quantMode == MODE_MXFP4) {
        inputType = ge::DT_FLOAT4_E2M1;
        scaleType = ge::DT_FLOAT8_E8M0;
    }
    if (CheckDtype(info, info.query, "q", {inputType}) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.key, "k", {inputType}) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.queryDescale, "q_descale", {scaleType}) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.keyDescale, "k_descale", {scaleType}) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.weights, "w", {weightType}) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.sparseValues, "sparse_values", {ge::DT_BF16}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckScaleShape(const IndexerV2CheckerInfo &info, const CheckerTensor &input,
                                               const CheckerTensor &scale, const char *inputName,
                                               const char *scaleName) const
{
    const gert::Shape &inputShape = *input.GetShape();
    const gert::Shape &scaleShape = *scale.GetShape();
    const uint32_t inputRank = inputShape.GetDimNum();
    if (info.quantMode == MODE_HIFLOAT8) {
        if (scaleShape.GetDimNum() != 1 || scaleShape.GetDim(0) != 1) {
            const std::string actual = ShapeToString(scale);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, scaleName, actual.c_str(),
                                                  "The scale shape must be [1] when quant_mode is 4");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    const bool mx = info.quantMode == MODE_MXFP8 || info.quantMode == MODE_MXFP4;
    const uint32_t expectedRank = mx ? inputRank + 1U : inputRank - 1U;
    if (scaleShape.GetDimNum() != expectedRank) {
        const std::string actual = std::to_string(scaleShape.GetDimNum()) + "D";
        const std::string expected = std::to_string(expectedRank) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(info.opName, scaleName, actual.c_str(), expected.c_str());
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < inputRank - 1U; ++i) {
        if (scaleShape.GetDim(i) != inputShape.GetDim(i)) {
            const std::string actual = ShapeToString(input) + " and " + ShapeToString(scale);
            const std::string reason = std::string(inputName) + " and " + scaleName +
                                       " must have equal dimensions before the head_dim axis";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(info.opName, scaleName, actual.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (mx) {
        const int64_t scaleD = inputShape.GetDim(inputRank - 1U) / MX_GROUP;
        if (inputShape.GetDim(inputRank - 1U) % MX_GROUP != 0 ||
            scaleShape.GetDim(inputRank - 1U) != scaleD || scaleShape.GetDim(inputRank) != MX_PACK) {
            const std::string actual = ShapeToString(scale);
            const std::string reason = "The last two scale dimensions must be [head_dim/64, 2] in MX mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, scaleName, actual.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const IndexerV2CheckerInfo &info) const
{
    if (CheckScaleShape(info, info.query, info.queryDescale, "q", "q_descale") != ge::GRAPH_SUCCESS ||
        CheckScaleShape(info, info.key, info.keyDescale, "k", "k_descale") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::PA_BBND && info.key.stride != nullptr && info.key.stride->GetDimNum() > 0) {
        const int64_t stride0 = info.key.stride->GetStride(0);
        if (stride0 <= 0 || (info.quantMode == MODE_MXFP4 && stride0 % MX_PACK != 0)) {
            const std::string actual = std::to_string(stride0);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                info.opName, "k stride0", actual.c_str(), "PA stride0 must be positive and MXFP4 stride0 must be even");
            return ge::GRAPH_FAILED;
        }
    }
    if ((info.quantMode == MODE_MXFP8 || info.quantMode == MODE_MXFP4) &&
        info.layoutK == CheckerLayout::PA_BBND && info.keyDescale.stride != nullptr &&
        info.keyDescale.stride->GetDimNum() > 0) {
        const int64_t stride0 = info.keyDescale.stride->GetStride(0);
        if (stride0 <= 0 || stride0 % MX_PACK != 0) {
            const std::string actual = std::to_string(stride0);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                info.opName, "k_descale stride0", actual.c_str(), "PA MX scale stride0 must be positive and even");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace indexer_v2_checker
} // namespace optiling
