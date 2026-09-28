/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_swiglu_quant_v2_weight_quant_checker.h"

namespace optiling {
namespace GMMSQWeightQuantChecker {

using namespace GroupedMatmulSwigluQuantV2Tiling;

bool CheckCoreNum(NpuArch npuArch, uint32_t aicNum, uint32_t aivNum)
{
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(OP_NAME, "Invalid aicNum[%u] or aivNum[%u], expect greater than 0.", aicNum, aivNum),
                return false);

    uint32_t expectedAicAivCoreRatio = npuArch == NpuArch::DAV_3510 ? AIC_AIV_CORE_RATIO_2 : AIC_AIV_CORE_RATIO_1;
    OP_CHECK_IF(aicNum * expectedAicAivCoreRatio != aivNum,
                OP_LOGE(OP_NAME,
                        "Invalid cube/vector core ratio. Expected cube:vector = 1:%u, but got cube=%u, vector=%u.",
                        expectedAicAivCoreRatio, aicNum, aivNum),
                return false);
    return true;
}

bool CheckDynamicInputShapeNotEmpty(const gert::TilingContext &context, uint32_t inputIndex, const char *inputName)
{
    const auto *shapePtr = context.GetDynamicInputShape(inputIndex, 0);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(OP_NAME, "Dynamic input shape of %s is nullptr.", inputName),
                return false);

    const auto &storageShape = shapePtr->GetStorageShape();
    if (storageShape.GetShapeSize() == 0) {
        std::string reason = std::string(inputName) + " can not be empty";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, inputName, "empty", reason.c_str());
        return false;
    }
    return true;
}

bool IsDynamicInputEmpty(const gert::TilingContext &context, uint32_t index)
{
    if (context.GetInputDesc(index) == nullptr) {
        return true;
    }

    const auto *shapePtr = context.GetDynamicInputShape(index, 0);
    if (shapePtr == nullptr) {
        return true;
    }
    const auto &storageShape = shapePtr->GetStorageShape();
    return storageShape.GetShapeSize() == 0 || storageShape.GetShapeSize() == 1;
}

bool CheckTilingDataCapacity(gert::TilingContext &context, uint64_t tilingDataSize)
{
    const auto *rawTiling = context.GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(OP_NAME, "GetRawTilingData can not be empty."), return false);
    OP_CHECK_IF(rawTiling->GetData() == nullptr, OP_LOGE(OP_NAME, "RawTilingData cannot be a null pointer."),
                return false);
    OP_CHECK_IF(rawTiling->GetCapacity() < tilingDataSize,
                OP_LOGE(OP_NAME, "context tiling data capacity %zu < actual tiling data size %zu.",
                        rawTiling->GetCapacity(), tilingDataSize),
                return false);
    return true;
}

bool CheckContext(gert::TilingContext &context, uint64_t tilingDataSize)
{
    const auto *attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(OP_NAME, "Function context.GetAttrs() failed!"), return false);

    OP_CHECK_IF(context.GetInputDesc(X_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc x is nullptr."), return false);
    OP_CHECK_IF(context.GetInputDesc(X_SCALE_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc xScale is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputDesc(GROUPLIST_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc groupList is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc weight is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_SCALE_INDEX) == nullptr,
                OP_LOGE(OP_NAME, "InputDesc weightScale is nullptr."), return false);
    OP_CHECK_IF(context.GetOutputDesc(Y_INDEX) == nullptr, OP_LOGE(OP_NAME, "OutputDesc y is nullptr."), return false);
    OP_CHECK_IF(context.GetOutputDesc(Y_SCALE_INDEX) == nullptr, OP_LOGE(OP_NAME, "OutputDesc yScale is nullptr."),
                return false);

    if (unlikely(!IsDynamicInputEmpty(context, WEIGHT_ASSIST_MATRIX_INDEX))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "weightAssistMatrix", "not nullptr",
            "weightAssistMatrix can not be provided, pass a null pointer or an empty list");
        return false;
    }
    if (unlikely(context.GetOptionalInputDesc(BIAS_INDEX) != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "bias", "not nullptr",
                                              "bias can not be provided, pass a null pointer");
        return false;
    }
    if (unlikely(context.GetOptionalInputDesc(SMOOTH_SCALE_INDEX) != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "smoothScale", "not nullptr",
                                              "smoothScale can not be provided, pass a null pointer");
        return false;
    }

    OP_CHECK_IF(context.GetInputShape(X_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputShape x is nullptr."), return false);
    OP_CHECK_IF(context.GetInputShape(X_SCALE_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputShape xScale is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputShape(GROUPLIST_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputShape groupList is nullptr."),
                return false);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, "weight"),
                OP_LOGE(OP_NAME, "Check dynamic input shape of weight failed."), return false);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_SCALE_INDEX, "weightScale"),
                OP_LOGE(OP_NAME, "Check dynamic input shape of weightScale failed."), return false);

    return CheckTilingDataCapacity(context, tilingDataSize);
}

bool GetAttrs(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context)
{
    const auto *attrs = context.GetAttrs();

    params.wTrans = GetOrDefault(attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_WEIGHT), true);
    params.groupListType = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE), 0L);
    params.dequantMode = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE), MX_QUANT_MODE);
    params.dequantDtype = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE),
                                       static_cast<int64_t>(ge::DataType::DT_FLOAT));
    params.quantMode = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE), MX_QUANT_MODE);
    params.quantDtype = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE),
                                     static_cast<int64_t>(ge::DataType::DT_FLOAT8_E4M3FN));

    OP_LOGD(OP_NAME,
            "GetAttrs result: wTrans=%s, groupListType=%ld, dequantMode=%ld, dequantDtype=%ld, "
            "quantMode=%ld, quantDtype=%ld",
            params.wTrans ? "true" : "false", params.groupListType, params.dequantMode, params.dequantDtype,
            params.quantMode, params.quantDtype);
    return true;
}

bool CheckAttrs(const GMMSQWeightQuantInputParams &params)
{
    if (unlikely(params.dequantMode != MX_QUANT_MODE)) {
        std::string dequantModeVal = std::to_string(params.dequantMode);
        std::string dequantModeReason = "dequantMode must be " + std::to_string(MX_QUANT_MODE) + " (MX quant)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dequantMode", dequantModeVal.c_str(),
                                              dequantModeReason.c_str());
        return false;
    }

    if (unlikely(params.dequantDtype != ge::DataType::DT_FLOAT)) {
        std::string dequantDtypeVal = std::to_string(params.dequantDtype);
        std::string dequantDtypeReason =
            "dequantDtype must be " + std::to_string(ge::DataType::DT_FLOAT) + " (DT_FLOAT)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dequantDtype", dequantDtypeVal.c_str(),
                                              dequantDtypeReason.c_str());
        return false;
    }

    if (unlikely(params.quantMode != MX_QUANT_MODE)) {
        std::string quantModeVal = std::to_string(params.quantMode);
        std::string quantModeReason = "quantMode must be " + std::to_string(MX_QUANT_MODE) + " (MX quant)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantMode", quantModeVal.c_str(), quantModeReason.c_str());
        return false;
    }

    if (unlikely(params.quantDtype != ge::DataType::DT_FLOAT8_E4M3FN)) {
        std::string quantDtypeVal = std::to_string(params.quantDtype);
        std::string quantDtypeReason =
            "quantDtype must be " + std::to_string(ge::DataType::DT_FLOAT8_E4M3FN) + " (DT_FLOAT8_E4M3FN)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantDtype", quantDtypeVal.c_str(), quantDtypeReason.c_str());
        return false;
    }

    if (unlikely(!params.wTrans)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transposeWeight", "false",
                                              "the value of transposeWeight must be true");
        return false;
    }

    if (unlikely(params.groupListType != 0 && params.groupListType != 1)) {
        std::string glTypeVal = std::to_string(params.groupListType);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupListType", glTypeVal.c_str(),
                                              "attr groupListType must be 0 or 1");
        return false;
    }

    return true;
}

bool CheckDtypes(const gert::TilingContext &context, NpuArch npuArch)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);

    if (unlikely(xDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)) {
        std::string xDtypeStr = ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "x", xDtypeStr.c_str(), "the dtype of x must be FLOAT8_E4M3FN");
        return false;
    }

    std::vector<ge::DataType> supportedWeightDtype = {ge::DT_FLOAT4_E2M1};
    if (npuArch == NpuArch::DAV_9201) {
        supportedWeightDtype.push_back(ge::DT_HIFLOAT4);
    }
    if (unlikely(std::find(supportedWeightDtype.begin(), supportedWeightDtype.end(), wDesc->GetDataType()) ==
                 supportedWeightDtype.end())) {
        std::string wDtypeStr = ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType());
        std::string wDtypeReason = "the dtype of weight must be in " + ToVectorString(supportedWeightDtype);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "weight", wDtypeStr.c_str(), wDtypeReason.c_str());
        return false;
    }

    if (unlikely(xScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0)) {
        std::string xsDtypeStr = ge::TypeUtils::DataTypeToSerialString(xScaleDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "xScale", xsDtypeStr.c_str(),
                                              "the dtype of xScale must be FLOAT8_E8M0");
        return false;
    }

    if (unlikely(wDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && wScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0)) {
        std::string wsDtypeStr = ge::TypeUtils::DataTypeToSerialString(wScaleDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "weightScale", wsDtypeStr.c_str(),
            "when the dtype of weight is FLOAT4_E2M1, the dtype of weightScale must be FLOAT8_E8M0");
        return false;
    }

    if (unlikely(wDesc->GetDataType() == ge::DT_HIFLOAT4 && wScaleDesc->GetDataType() != ge::DT_HIFLOAT4_SCALE)) {
        std::string wsDtypeStr = ge::TypeUtils::DataTypeToSerialString(wScaleDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "weightScale", wsDtypeStr.c_str(),
            "when the dtype of weight is HIFLOAT4, the dtype of weightScale must be HIFLOAT4_SCALE");
        return false;
    }

    if (unlikely(groupListDesc->GetDataType() != ge::DT_INT64)) {
        std::string glDtypeStr = ge::TypeUtils::DataTypeToSerialString(groupListDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "groupList", glDtypeStr.c_str(),
                                              "the dtype of groupList must be INT64");
        return false;
    }

    return true;
}

bool GetInputs(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context)
{
    const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
    params.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetFormat().GetStorageFormat()));

    const auto *xShapePtr = context.GetInputShape(X_INDEX);
    const auto *wShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);

    const gert::Shape &xShape = xShapePtr->GetOriginShape();
    const gert::Shape &wShape = wShapePtr->GetOriginShape();
    auto xShapeLen = xShape.GetDimNum();
    auto wShapeLen = wShape.GetDimNum();
    OP_CHECK_IF(xShape.GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", "empty", "x can not be empty"), return false);
    if (unlikely(xShapeLen != DIM_NUM_X)) {
        std::string xDimVal = std::to_string(xShapeLen);
        std::string xDimReason = "the shape dim of x must be " + std::to_string(DIM_NUM_X);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "x", xDimVal.c_str(), xDimReason.c_str());
        return false;
    }
    if (unlikely(wShapeLen != DIM_NUM_WEIGHT_ND)) {
        std::string wDimVal = std::to_string(wShapeLen);
        std::string wDimReason = "the shape dim of weight must be " + std::to_string(DIM_NUM_WEIGHT_ND);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight", wDimVal.c_str(), wDimReason.c_str());
        return false;
    }

    auto xInner = xShape.GetDim(DIM_NUM_X - 1);
    auto wInner = wShape.GetDim(DIM_NUM_WEIGHT_ND - 1);
    if (unlikely(xInner != wInner)) {
        std::string incorrectShapes = "x K=" + std::to_string(xInner) + ", weight K=" + std::to_string(wInner);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, "x, weight", incorrectShapes.c_str(),
                                               "K of x must be equal to K of weight");
        return false;
    }
    params.mSize = xShape.GetDim(DIM_0);
    params.kSize = xInner;
    params.nSize = wShape.GetDim(DIM_1);
    params.groupNum = wShape.GetDim(DIM_0);

    OP_LOGD(OP_NAME, "GetInputs result: mSize=%ld, kSize=%ld, nSize=%ld, groupNum=%ld, wFormat=%s", params.mSize,
            params.kSize, params.nSize, params.groupNum, ge::TypeUtils::FormatToSerialString(params.wFormat).c_str());
    return true;
}

bool CheckInputFormats(const gert::TilingContext &context, const GMMSQWeightQuantInputParams &params, NpuArch npuArch)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    auto xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    if (unlikely(xFormat != ge::FORMAT_ND)) {
        std::string xFmtStr = ge::TypeUtils::FormatToSerialString(xFormat);
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "x", xFmtStr.c_str(), "ND");
        return false;
    }

    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    auto xScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()));
    if (unlikely(xScaleFormat != ge::FORMAT_ND && xScaleFormat != ge::FORMAT_NCL)) {
        std::string xsFmtStr = ge::TypeUtils::FormatToSerialString(xScaleFormat);
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "xScale", xsFmtStr.c_str(), "ND or NCL");
        return false;
    }

    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);
    auto groupListFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(groupListDesc->GetFormat().GetStorageFormat()));
    if (unlikely(groupListFormat != ge::FORMAT_ND)) {
        std::string glFmtStr = ge::TypeUtils::FormatToSerialString(groupListFormat);
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "groupList", glFmtStr.c_str(), "ND");
        return false;
    }

    std::vector<ge::Format> supportedWeightFormat = {ge::FORMAT_FRACTAL_NZ};
    if (npuArch == NpuArch::DAV_3510) {
        supportedWeightFormat.push_back(ge::FORMAT_FRACTAL_NZ_C0_32);
    }
    if (unlikely(std::find(supportedWeightFormat.begin(), supportedWeightFormat.end(), params.wFormat) ==
                 supportedWeightFormat.end())) {
        std::string wFmtStr = ge::TypeUtils::FormatToSerialString(params.wFormat);
        std::string supportedFmtStr = ToVectorString(supportedWeightFormat);
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight", wFmtStr.c_str(), supportedFmtStr.c_str());
        return false;
    }

    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    auto wScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wScaleDesc->GetFormat().GetStorageFormat()));
    if (unlikely(wScaleFormat != ge::FORMAT_ND && wScaleFormat != ge::FORMAT_NCHW)) {
        std::string wsFmtStr = ge::TypeUtils::FormatToSerialString(wScaleFormat);
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weightScale", wsFmtStr.c_str(), "ND or NCHW");
        return false;
    }

    return true;
}

std::string ToShapeString(const std::vector<int64_t> &shape)
{
    std::string shapeStr("[");
    const char *sep = "";
    for (auto x : shape) {
        shapeStr.append(sep);
        shapeStr.append(std::to_string(x));
        sep = ", ";
    }
    shapeStr.push_back(']');
    return shapeStr;
}

bool CheckInputShape(const char *variableName, const gert::Shape &shape, const std::vector<int64_t> &expectedShape)
{
    auto shapeLen = shape.GetDimNum();
    if (unlikely(shapeLen != expectedShape.size())) {
        std::string dimVal = std::to_string(shapeLen);
        std::string dimReason =
            "the shape dim of " + std::string(variableName) + " must be " + std::to_string(expectedShape.size());
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, variableName, dimVal.c_str(), dimReason.c_str());
        return false;
    }
    size_t i = 0;
    for (auto dim : expectedShape) {
        if (unlikely(dim != shape.GetDim(i++))) {
            std::string shapeStr = Ops::Base::ToString(shape);
            std::string expectedStr = "shape does not match the expected value " + ToShapeString(expectedShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, variableName, shapeStr.c_str(), expectedStr.c_str());
            return false;
        }
    }
    return true;
}

bool IsInputShapeValid(const GMMSQWeightQuantInputParams &params)
{
    if (unlikely(params.kSize < MIN_K_SIZE)) {
        std::string shapeVal = "K=" + std::to_string(params.kSize);
        std::string reason = "K of x must be >= " + std::to_string(MIN_K_SIZE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", shapeVal.c_str(), reason.c_str());
        return false;
    }
    if (unlikely(params.nSize < MIN_N_SIZE)) {
        std::string shapeVal = "N=" + std::to_string(params.nSize);
        std::string reason = "N of weight must be >= " + std::to_string(MIN_N_SIZE);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", shapeVal.c_str(), reason.c_str());
        return false;
    }
    if (unlikely(params.kSize % ALIGN_SIZE_K != 0)) {
        std::string shapeVal = "K=" + std::to_string(params.kSize);
        std::string reason = "K of x must be aligned to " + std::to_string(ALIGN_SIZE_K);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", shapeVal.c_str(), reason.c_str());
        return false;
    }
    if (unlikely(params.nSize % ALIGN_SIZE_N != 0)) {
        std::string shapeVal = "N=" + std::to_string(params.nSize);
        std::string reason = "N of weight must be aligned to " + std::to_string(ALIGN_SIZE_N);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", shapeVal.c_str(), reason.c_str());
        return false;
    }
    if (unlikely(params.groupNum < 1 || params.groupNum > MAX_GROUP_NUM)) {
        std::string shapeVal = "groupNum=" + std::to_string(params.groupNum);
        std::string reason = "groupNum of weight must be in range [1, " + std::to_string(MAX_GROUP_NUM) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", shapeVal.c_str(), reason.c_str());
        return false;
    }
    return true;
}

int64_t GetC0Size(ge::Format format, ge::DataType dType)
{
    if (format == ge::FORMAT_FRACTAL_NZ && (dType == ge::DT_FLOAT4_E2M1 || dType == ge::DT_HIFLOAT4)) {
        return C0_SIZE << 1;
    }
    return C0_SIZE;
}

bool CheckInputs(const gert::TilingContext &context, const GMMSQWeightQuantInputParams &params, NpuArch npuArch)
{
    if (!IsInputShapeValid(params)) {
        return false;
    }

    if (!CheckInputFormats(context, params, npuArch)) {
        return false;
    }

    auto *xScaleShapePtr = context.GetInputShape(X_SCALE_INDEX);
    auto *groupListShapePtr = context.GetInputShape(GROUPLIST_INDEX);
    auto *weightShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto *weightScaleShapePtr = context.GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);

    auto &xScaleShape = xScaleShapePtr->GetOriginShape();
    auto &groupListShape = groupListShapePtr->GetOriginShape();
    auto &weightScaleShape = weightScaleShapePtr->GetOriginShape();

    if (params.wFormat == ge::FORMAT_FRACTAL_NZ || params.wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        const gert::Shape &wStorageShape = weightShapePtr->GetStorageShape();
        const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
        int64_t c0Size = GetC0Size(params.wFormat, wDesc->GetDataType());
        const std::vector<int64_t> expectedWeightShape = {
            static_cast<int64_t>(params.groupNum), GroupedMatmul::CeilDiv<int64_t>(params.kSize, c0Size),
            GroupedMatmul::CeilDiv<int64_t>(params.nSize, CUBE_BLOCK), CUBE_BLOCK, c0Size};
        if (!CheckInputShape("weight", wStorageShape, expectedWeightShape)) {
            return false;
        }
    }

    int64_t scaleKSize = GroupedMatmul::CeilDiv<int64_t>(params.kSize, MX_SCALE_BLOCK_SIZE);
    const std::vector<int64_t> expectedXScaleShape = {static_cast<int64_t>(params.mSize), scaleKSize, MX_LAST_DIM_2};
    if (!CheckInputShape("xScale", xScaleShape, expectedXScaleShape)) {
        return false;
    }
    if (!CheckInputShape("groupList", groupListShape, {params.groupNum})) {
        return false;
    }

    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    int64_t wScaleLastDim = wScaleDesc->GetDataType() == ge::DT_HIFLOAT4_SCALE ? MX_LAST_DIM_1 : MX_LAST_DIM_2;
    const std::vector<int64_t> expectedWeightScaleShape = {params.groupNum, params.nSize, scaleKSize, wScaleLastDim};
    if (!CheckInputShape("weightScale", weightScaleShape, expectedWeightScaleShape)) {
        return false;
    }
    return true;
}

ge::graphStatus GetShapeAttrsInfo(gert::TilingContext &context, GMMSQWeightQuantInputParams &params, NpuArch npuArch)
{
    OP_LOGD(OP_NAME, "Enter function GetShapeAttrsInfo.");
    if (!CheckContext(context, sizeof(GMMSQArch35Tiling::GMMSQWeightQuantAswTilingData))) {
        return ge::GRAPH_FAILED;
    }
    if (!GetAttrs(params, context)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckAttrs(params)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckDtypes(context, npuArch)) {
        return ge::GRAPH_FAILED;
    }
    if (!GetInputs(params, context)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckInputs(context, params, npuArch)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace GMMSQWeightQuantChecker
} // namespace optiling