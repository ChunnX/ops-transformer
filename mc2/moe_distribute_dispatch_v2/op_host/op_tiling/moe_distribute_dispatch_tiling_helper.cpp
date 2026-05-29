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
 * \file moe_distribute_dispatch_tiling_helper.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_helper.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

using namespace ge;

namespace optiling {
inline bool MoeDistributeDispatchTilingHelper::CheckInputTensorDim(const gert::TilingContext *context,
    const char *nodeName, const bool isScales, const uint32_t quantMode)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShape"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "x", "invalid",
                        std::string("xShape dims must be 2, but current dim num is ") +
                            std::to_string(xStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdShape"),
                    return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "x", "invalid",
                        std::string("expertIdShape dims must be 2, but current dim num is ") +
                            std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));
    // 如果scales不为空进行shape维度检查
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesShape"),
                        return false);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) {
            // the cond is compatible with A2/A3 because static quant is only supported on A5
            OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                nodeName, "quantMode", "invalid",
                                std::string("scales dims must be 2 when quantMode=") +
                                    std::to_string(quantMode) +
                                    ", but current dim num is " +
                                    std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()) +
                                    "."
                            ),
                            return false);
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
        } else {
            OP_TILING_CHECK((scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM) &&
                                (scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS),
                                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                nodeName, "quantMode", "invalid",
                                std::string("scalesShape dims must be 1 or 2 when quantMode is 1, but current dim"
                                                "num is ") +
                                    std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()) +
                                    "."
                            ),
                            return false);
            // additional check for hif8 quant
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
            OP_TILING_CHECK(
                (expandXDesc->GetDataType() == ge::DT_HIFLOAT8) &&
                    (scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM),
                                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName, "scales", "invalid",
                    std::string("scalesShape dims must be 1 when x dtype is hif8 in static quant, but current dim"
                                    "num is ") +
                        std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()) +
                        "."
                ),
                return false);
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            if (scalesStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
                OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
            }
        }
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckDynamicScalesDim(const gert::TilingContext *context,
    const char *nodeName, const uint32_t quantMode)
{
     const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr,
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesShape"), return false);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT))) {
        // quantMode 2: 1dim, the same in A2/A3/A5
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_ONE_DIM_NUM,
                                                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName, "quantMode", "invalid",
                            std::string("dynamicScalesShape dims must be ") +
                                std::to_string(DYNAMIC_SCALE_TWO_DIM_NUM) +
                                " when quantMode=" +
                                std::to_string(quantMode) +
                                ", but current dim num is " +
                                std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()) +
                                "."
                        ),
                        return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    } else {
        // MX/PERTILE
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_TWO_DIM_NUM,
                                                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName, "quantMode", "invalid",
                            std::string("dynamicScalesShape dims must be ") +
                                std::to_string(DYNAMIC_SCALE_ONE_DIM_NUM) +
                                " when quantMode=" +
                                std::to_string(quantMode) +
                                ", but current dim num is " +
                                std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()) +
                                "."
                        ),
                        return false);
        OP_LOGD(nodeName, "dynamicScales dim0=%ld, dim1=%ld",
            dynamicScalesStorageShape->GetStorageShape().GetDim(0),
            dynamicScalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckOutputTensorDim(gert::TilingContext *context,
    const char *nodeName, const uint32_t quantMode)
{
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXShape"), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "x", "invalid",
                        std::string("expandXShape dims must be 2, but current dim num is ") +
                            std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    // Skip checking dynamicScales when quantMode is 0 or 1, the same in A2/A3/A5
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT))
        && (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        OP_TILING_CHECK(!CheckDynamicScalesDim(context, nodeName, quantMode),
            OP_LOGE(nodeName, "CheckDynamicScalesDim failed."), return false);
    }

    const gert::StorageShape *expandIdxStorageShape = context->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxShape"),
                    return false);
    OP_TILING_CHECK(expandIdxStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "x", "invalid",
                        std::string("expandIdxShape dims must be 1, but current dim num is ") +
                            std::to_string(expandIdxStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "expandIdx dim0 = %ld", expandIdxStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsShape"), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "x", "invalid",
                        std::string("expertTokenNumsShape dims must be 1, but current dim num is ") +
                            std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckEpTpRecvTensorDim(
    const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountShape"),
                    return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "H", "invalid",
                        std::string("epRecvCountShape dims must be 1, but current dim num is ") +
                            std::to_string(epRecvCountStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRecvCountShape"), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "H", "invalid",
                        std::string("tpRecvCountShape dims must be 1, but current dim num is ") +
                            std::to_string(tpRecvCountStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorDim(gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const uint32_t opVersion)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, isScales, quantMode),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "Input param", "invalid",
                                                          "The shape of parameter Input param is invalid"),
                    return false);

    // A3/A5 v1接口的x_active_mask不支持传入
    if (opVersion == OP_VERSION_1) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(
            xActiveMaskStorageShape != nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "x", "invalid", "x_active_mask only support input None."),
            return false);
    }

    OP_TILING_CHECK((!CheckOutputTensorDim(context, nodeName, quantMode)) ||
                        (!CheckEpTpRecvTensorDim(context, nodeName)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "Output param", "invalid",
                                                          "The shape of parameter Output param is invalid"),
                    return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckCommonOutputTensorDataType(
    const gert::TilingContext *context, const char *nodeName)
{
    auto expandIdxDesc = context->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc"), return false);
    OP_TILING_CHECK(expandIdxDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandIdx",
                                              Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str(), "int32"),
                    return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsDesc"),
        return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertTokenNums",
                                              Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(), "int64"),
                    return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountsDesc"),
                    return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "epRecvCounts",
                                              Ops::Base::ToString(epRecvCountsDesc->GetDataType()).c_str(), "int32"),
                    return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRecvCountsDesc"),
                    return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "tpRecvCounts",
                                              Ops::Base::ToString(tpRecvCountsDesc->GetDataType()).c_str(), "int32"),
                    return false);

    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckInputTensorDataType(const gert::TilingContext *context,
    const char *nodeName, const bool isScales)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16 or float16"),
        return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc"), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertId",
                                              Ops::Base::ToString(expertIdDesc->GetDataType()).c_str(), "int32"),
                    return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
    }
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorDataType(gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(!CheckInputTensorDataType(context, nodeName, isScales),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "Input param", "invalid",
                                                          "The dtype of parameter Input param is invalid"),
                    return false);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
    if (quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX",
                                                  Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), "int8"),
                        return false);
    } else {
        OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX",
                                                  Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                                  Ops::Base::ToString(xDesc->GetDataType()).c_str()),
                        return false);
    }

    if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                                  Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(),
                                                  "float"),
                        return false);
    }

    OP_TILING_CHECK(!CheckCommonOutputTensorDataType(context, nodeName),
        OP_LOGE(nodeName, "CheckCommonOutputTensorDataType failed."), return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeNoScales(const gert::TilingContext *context,
    const char *nodeName, const bool isScales)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
    OP_TILING_CHECK((NON_QUANT_DTYPE.find(static_cast<ge::DataType>(xDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "one of bf16/fp16/e5m2/e4m3fn/hif8"),
                    return false);
    // ExpandX: the same as X
    OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX",
                                              Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                              Ops::Base::ToString(xDesc->GetDataType()).c_str()),
                    return false);
    // Scales: bf16/fp16: nullptr; hif8: fp32; e5m2/e4m3fn: float/e8m0
    // Dynamic scales: the same as scales, and no validations for bf16/fp16
    // If X is bf16/fp16, the scales must be nullptr, which is validated in CheckQuantModeAndScales
    // Hence the datatype of X must be e5m2/e4m3fn/hif8 when isScales is true
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
            return false);
        OP_TILING_CHECK((xDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
        OP_TILING_CHECK(
            (scalesDesc->GetDataType() != ge::DT_FLOAT) && (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                                      "float or e8m0"),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != scalesDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                                  Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(),
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str()),
                        return false);
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeStaticOrDynamic(
    const gert::TilingContext *context, const char *nodeName, bool isScales)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
        return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16 or float16"),
        return false);
    // Scales: fp32, optional for dynamic/pertoken/pertile, required for static/hif8
    // isScales has been checked in CheckQuantModeAndScales
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
    }
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "float"),
                    return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeMxfp8(
    const gert::TilingContext *context, const char *nodeName)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
        return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16 or float16"),
        return false);
    // No Scales input
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "e8m0"),
                    return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckDistinctTensorDataType(gert::TilingContext *context,
                                                                           const char *nodeName, const bool isScales,
                                                                           const uint32_t quantMode)
{
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeNoScales(context, nodeName, isScales),
            OP_LOGE(nodeName, "CheckTensorDataType for nonquant mode failed."), return false);
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeMxfp8(context, nodeName),
            OP_LOGE(nodeName, "CheckTensorDataType for mx quant mode failed."), return false);
    } else {
        // static/dynamic/pertolen/pertile/hif8
        OP_TILING_CHECK(!CheckTensorDataTypeStaticOrDynamic(context, nodeName, isScales),
                                                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "quantMode", "invalid",
                            std::string("CheckTensorDataType for quantMode ") +
                                std::to_string(quantMode) +
                                " failed."
                        ),
                        return false);
    }
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeA5(gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc"), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertId",
                                              Ops::Base::ToString(expertIdDesc->GetDataType()).c_str(), "int32"),
                    return false);

    OP_TILING_CHECK(!CheckDistinctTensorDataType(context, nodeName, isScales, quantMode),
        OP_LOGE(nodeName, "CheckDistinctTensorDataType failed."), return false);

    OP_TILING_CHECK(!CheckCommonOutputTensorDataType(context, nodeName),
        OP_LOGE(nodeName, "CheckCommonOutputTensorDataType failed."), return false);
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorFormat(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x", Ops::Base::ToString(xDesc->GetStorageFormat()).c_str(),
                                               "not FORMAT_FRACTAL_NZ"),
                    return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertId", Ops::Base::ToString(expertIdDesc->GetStorageFormat()).c_str(),
                                   "not FORMAT_FRACTAL_NZ"),
        return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetStorageFormat()).c_str(),
                                       "not FORMAT_FRACTAL_NZ"),
            return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetStorageFormat()).c_str(),
                                   "not FORMAT_FRACTAL_NZ"),
        return false);

    // quantMode 2, compatible with A2/A3
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
            return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "dynamicScales",
                                                   Ops::Base::ToString(dynamicScalesDesc->GetStorageFormat()).c_str(),
                                                   "not FORMAT_FRACTAL_NZ"),
                        return false);
    }

    auto expandIdxDesc = context->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandIdxDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandIdx",
                                               Ops::Base::ToString(expandIdxDesc->GetStorageFormat()).c_str(),
                                               "not FORMAT_FRACTAL_NZ"),
                    return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsDesc"),
        return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertTokenNums",
                                               Ops::Base::ToString(expertTokenNumsDesc->GetStorageFormat()).c_str(),
                                               "not FORMAT_FRACTAL_NZ"),
                    return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountsDesc"),
                    return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "epRecvCounts",
                                               Ops::Base::ToString(epRecvCountsDesc->GetStorageFormat()).c_str(),
                                               "not FORMAT_FRACTAL_NZ"),
                    return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRecvCountsDesc"),
                    return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "tpRecvCounts",
                                               Ops::Base::ToString(tpRecvCountsDesc->GetStorageFormat()).c_str(),
                                               "not FORMAT_FRACTAL_NZ"),
                    return false);

    return true;
}

ge::graphStatus MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatch(gert::TilingContext *context,
                                                                                    const char *nodeName,
                                                                                    const bool isScales,
                                                                                    const uint32_t quantMode)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, OP_VERSION_1),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "params", "invalid",
                                                          "The shape of parameter params is invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "params", "invalid",
                                                          "The dtype of parameter params is invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode),
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "params", "invalid", "valid format"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatchA5(gert::TilingContext *context,
    const bool isScales, const uint32_t quantMode, const bool isTokenMask)
{
    // nodeName已在调用处判空
    const char *nodeName = context->GetNodeName();
    auto opVersion = OpVersionManager::GetInstance().GetVersion();
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, opVersion),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "params", "invalid",
                                                          "The shape of parameter params is invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataTypeA5(context, nodeName, isScales, quantMode),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "params", "invalid",
                                                          "The dtype of parameter params is invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode),
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "params", "invalid", "valid format"), return ge::GRAPH_FAILED);
    if ((opVersion != OP_VERSION_1) && isTokenMask) {
        OP_TILING_CHECK(!CheckTokenMask(context, nodeName),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "x", "invalid", "xActiveMask is invalid."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchTilingHelper::CheckTokenMask(const gert::TilingContext *context, const char *nodeName)
{
    // Check Dim/DType/Format
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskStorageShape"),
                    return false);
    OP_TILING_CHECK(xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "x", "invalid",
                        std::string("xActiveMask must be 1-dim, but current dim num is ") +
                            std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDimNum()) +
                            "."
                    ),
                    return false);
    auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc"), return false);
    OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "xActiveMask",
                                              Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str(), "bool"),
                    return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "xActiveMask",
                                               Ops::Base::ToString(xActiveMaskDesc->GetStorageFormat()).c_str(),
                                               "not FORMAT_FRACTAL_NZ"),
                    return false);

    return true;
}
}