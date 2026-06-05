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
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp
 * \brief GMMSQ MxA8W4 weight quant tiling implementation
 */
#include "grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "common/include/err/ops_err.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"
using namespace Ops::Transformer::OpTiling;

namespace {
static constexpr const char* OP_NAME = "GroupedMatmulSwigluQuantV2";
}  // namespace

namespace optiling {
using namespace GroupedMatmulSwigluQuantV2Tiling;

constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t MX_SCALE_BLOCK_SIZE = 64;
constexpr uint32_t MX_INNER_DIM = 2;
constexpr uint32_t ALIGN_SIZE_K = 32;
constexpr uint32_t ALIGN_SIZE_N = 128;
constexpr uint32_t AIC_AIV_CORE_RATIO = 2;

constexpr uint64_t DIM_NUM_X = 2;
constexpr uint64_t DIM_NUM_XSCALE = 3;
constexpr uint64_t DIM_NUM_GROUPLIST = 1;
constexpr uint64_t DIM_NUM_WEIGHT_NZ = 5;
constexpr uint64_t DIM_NUM_WEIGHT_ND = 3;
constexpr uint64_t DIM_NUM_WSCALE = 4;

constexpr int64_t C0_SIZE = 32;
constexpr int64_t CUBE_BLOCK = 16;

constexpr int64_t MIN_M_SIZE = 1;
constexpr int64_t MIN_K_SIZE = 32;
constexpr int64_t MIN_N_SIZE = 128;
constexpr int64_t MAX_GROUP_NUM = 1024;

constexpr int64_t MX_QUANT_MODE = 2;

template <typename T>
constexpr T GetOrDefault(const T* ptr, T defaultValue)
{
    return ptr == nullptr ? defaultValue : *ptr;
}

bool CheckCoreNum(gert::TilingContext *context, uint32_t aicNum, uint32_t aivNum)
{
    OP_CHECK_IF(aicNum <= 0 || aivNum <= 0,
        OP_LOGE(OP_NAME, "Invalid aicNum[%u] or aivNum[%u], expect greater than 0",
            aicNum, aivNum),
        return false);

    OP_CHECK_IF(aicNum * AIC_AIV_CORE_RATIO != aivNum,
        OP_LOGE(OP_NAME,
            "Invalid cube/vector core ratio. Expected cube:vector = 1:%u, "
            "but got cube=%u, vector=%u",
            AIC_AIV_CORE_RATIO, aicNum, aivNum),
        return false);
    return true;
}

bool GetContext(GMMSQWeightQuantInputParams& params, const gert::TilingContext& context)
{
    params.opName = context.GetNodeName();
    return true;
}

bool CheckDynamicInputShapeNotEmpty(const gert::TilingContext& context, uint32_t inputIndex, const char* inputName)
{
    const auto *shapePtr = context.GetDynamicInputShape(inputIndex, 0);
    OP_CHECK_IF(shapePtr == nullptr,
        OP_LOGE(OP_NAME, "Dynamic input shape of %s is nullptr.", inputName),
        return false);

    const auto &storageShape = shapePtr->GetStorageShape();
    OP_CHECK_IF(storageShape.GetShapeSize() == 0,
        OP_LOGE(OP_NAME, "Not yet support empty tensor for %s.", inputName),
        return false);
    return true;
}

bool IsDynamicInputEmpty(const gert::TilingContext& context, uint32_t index)
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

ge::graphStatus CheckTilingDataCapacity(gert::TilingContext& context, uint64_t tilingDataSize)
{
    const auto *rawTiling = context.GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr,
        OP_LOGE(OP_NAME, "GetRawTilingData can not be empty."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTiling->GetData() == nullptr,
        OP_LOGE(OP_NAME, "RawTilingData cannot be a null pointer."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTiling->GetCapacity() < tilingDataSize,
        OP_LOGE(OP_NAME,
            "context tiling data capacity %zu < actual tiling data size %zu.",
            rawTiling->GetCapacity(), tilingDataSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckContext(gert::TilingContext& context, uint64_t tilingDataSize)
{
    const auto *attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
        OP_LOGE(OP_NAME, "Function context.GetAttrs() failed!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.GetInputDesc(X_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "InputDesc x is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(X_SCALE_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "InputDesc xScale is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(GROUPLIST_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "InputDesc groupList is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "InputDesc weight is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "InputDesc weightScale is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetOutputDesc(Y_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "OutputDesc y is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetOutputDesc(Y_SCALE_INDEX) == nullptr,
        OP_LOGE(OP_NAME, "OutputDesc yScale is nullptr."),
        return ge::GRAPH_FAILED);

    if (unlikely(!IsDynamicInputEmpty(context, WEIGHT_ASSIST_MATRIX_INDEX))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "weightAssistMatrix",
            "not nullptr", "weightAssistMatrix is not supported, please pass a null pointer or an empty list");
        return ge::GRAPH_FAILED;
    }
    if (unlikely(context.GetOptionalInputDesc(BIAS_INDEX) != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "bias",
            "not nullptr", "bias is not supported, please pass a null pointer");
        return ge::GRAPH_FAILED;
    }
    if (unlikely(context.GetOptionalInputDesc(SMOOTH_SCALE_INDEX) != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "smoothScale",
            "not nullptr", "smoothScale is not supported, please pass a null pointer");
        return ge::GRAPH_FAILED;
    }

    return CheckTilingDataCapacity(context, tilingDataSize);
}

bool GetAttrs(GMMSQWeightQuantInputParams& params, const gert::TilingContext& context)
{
    params.wTrans = true;
    params.groupListType = 0;
    params.dequantMode = MX_QUANT_MODE;
    params.dequantDtype = static_cast<int64_t>(ge::DataType::DT_FLOAT);
    params.quantMode = MX_QUANT_MODE;
    params.quantDtype = static_cast<int64_t>(ge::DataType::DT_FLOAT8_E4M3FN);

    const auto *attrs = context.GetAttrs();
    if (attrs == nullptr) {
        return true;
    }

    params.wTrans = GetOrDefault(attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_WEIGHT), true);
    params.groupListType = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE), 0L);
    params.dequantMode = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE), MX_QUANT_MODE);
    params.dequantDtype = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE),
        static_cast<int64_t>(ge::DataType::DT_FLOAT));
    params.quantMode = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE), MX_QUANT_MODE);
    params.quantDtype = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE),
        static_cast<int64_t>(ge::DataType::DT_FLOAT8_E4M3FN));
    return true;
}

bool CheckAttrs([[maybe_unused]] const gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    if (unlikely(params.dequantMode != MX_QUANT_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dequantMode",
            std::to_string(params.dequantMode),
            "dequantMode must be " + std::to_string(MX_QUANT_MODE) + " (MX quant)");
        return false;
    }

    if (unlikely(params.dequantDtype != ge::DataType::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dequantDtype",
            std::to_string(params.dequantDtype),
            "dequantDtype must be " + std::to_string(ge::DataType::DT_FLOAT) + " (DT_FLOAT)");
        return false;
    }

    if (unlikely(params.quantMode != MX_QUANT_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantMode",
            std::to_string(params.quantMode),
            "quantMode must be " + std::to_string(MX_QUANT_MODE) + " (MX quant)");
        return false;
    }

    if (unlikely(params.quantDtype != ge::DataType::DT_FLOAT8_E4M3FN)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantDtype",
            std::to_string(params.quantDtype),
            "quantDtype must be " + std::to_string(ge::DataType::DT_FLOAT8_E4M3FN) + " (DT_FLOAT8_E4M3FN)");
        return false;
    }

    if (unlikely(!params.wTrans)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transposeWeight",
            "false",
            "The value of transposeWeight must be true");
        return false;
    }

    if (unlikely(params.groupListType != 0 && params.groupListType != 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupListType",
            std::to_string(params.groupListType),
            "Attr groupListType must be 0 or 1");
        return false;
    }

    return true;
}

bool CheckDtypes(const gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);

    if (unlikely(xDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "x",
            ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()),
            "The dtype of x must be FLOAT8_E4M3FN");
        return false;
    }

    if (unlikely(wDesc->GetDataType() != ge::DT_FLOAT4_E2M1)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "weight",
            ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType()),
            "The dtype of weight must be FLOAT4_E2M1");
        return false;
    }

    if (unlikely(xScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "xScale",
            ge::TypeUtils::DataTypeToSerialString(xScaleDesc->GetDataType()),
            "The dtype of xScale must be FLOAT8_E8M0");
        return false;
    }

    if (unlikely(wScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "weightScale",
            ge::TypeUtils::DataTypeToSerialString(wScaleDesc->GetDataType()),
            "The dtype of weightScale must be FLOAT8_E8M0");
        return false;
    }

    if (unlikely(groupListDesc->GetDataType() != ge::DT_INT64)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "groupList",
            ge::TypeUtils::DataTypeToSerialString(groupListDesc->GetDataType()),
            "The dtype of groupList must be INT64");
        return false;
    }

    return true;
}

bool GetInputs(GMMSQWeightQuantInputParams& params, const gert::TilingContext& context)
{
    const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
    params.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetFormat().GetStorageFormat()));
    if (unlikely(params.wFormat != ge::FORMAT_FRACTAL_NZ && params.wFormat != ge::FORMAT_FRACTAL_NZ_C0_32)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight",
            ge::TypeUtils::FormatToAscendString(params.wFormat).GetString(),
            "FRACTAL_NZ or FRACTAL_NZ_C0_32");
        return false;
    }

    const auto *xShapePtr = context.GetInputShape(X_INDEX);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, "weight"),
        OP_LOGE(OP_NAME, "Check dynamic input shape of weight failed."), return false);
    const auto *wShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(xShapePtr == nullptr || wShapePtr == nullptr,
                OP_LOGE(OP_NAME, "Required input shape is nullptr."), return false);

    const gert::Shape &xShape = xShapePtr->GetOriginShape();
    const gert::Shape &wShape = wShapePtr->GetOriginShape();
    auto xShapeLen = xShape.GetDimNum();
    auto wShapeLen = wShape.GetDimNum();
    OP_CHECK_IF(xShape.GetShapeSize() == 0,
        OP_LOGE(OP_NAME, "Not yet support empty tensor for x"), return false);
    if (unlikely(xShapeLen != DIM_NUM_X)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "x",
            std::to_string(xShapeLen),
            "The shape dim of x must be " + std::to_string(DIM_NUM_X));
        return false;
    }
    if (unlikely(wShapeLen != DIM_NUM_WEIGHT_ND)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight",
            std::to_string(wShapeLen),
            "The shape dim of weight must be " + std::to_string(DIM_NUM_WEIGHT_ND));
        return false;
    }

    // - 1 为内轴
    auto xInner = xShape.GetDim(DIM_NUM_X - 1);
    auto wInner = wShape.GetDim(DIM_NUM_WEIGHT_ND - 1);
    if (unlikely(xInner != wInner)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, "x, weight",
            "x K=" + std::to_string(xInner) + ", weight K=" + std::to_string(wInner),
            "K of x must be equal to K of weight");
        return false;
    }
    params.mSize = xShape.GetDim(DIM_0);
    params.kSize = xInner;
    params.nSize = wShape.GetDim(DIM_1);
    params.groupNum = wShape.GetDim(DIM_0);
    return true;
}

bool CheckInputFormats(const gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    auto xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    if (unlikely(xFormat != ge::FORMAT_ND)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "x",
            ge::TypeUtils::FormatToSerialString(xFormat), "ND");
        return false;
    }

    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    auto xScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()));
    if (unlikely(xScaleFormat != ge::FORMAT_ND && xScaleFormat != ge::FORMAT_NCL)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "xScale",
            ge::TypeUtils::FormatToSerialString(xScaleFormat), "ND or NCL");
        return false;
    }

    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);
    auto groupListFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(groupListDesc->GetFormat().GetStorageFormat()));
    if (unlikely(groupListFormat != ge::FORMAT_ND)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "groupList",
            ge::TypeUtils::FormatToSerialString(groupListFormat), "ND");
        return false;
    }

    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    auto wScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wScaleDesc->GetFormat().GetStorageFormat()));
    if (unlikely(wScaleFormat != ge::FORMAT_ND && wScaleFormat != ge::FORMAT_NCHW)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weightScale",
            ge::TypeUtils::FormatToSerialString(wScaleFormat), "ND or NCHW");
        return false;
    }

    return true;
}

std::string ToShapeString(const std::vector<int64_t>& shape)
{
    std::string shapeStr("[");
    const char* sep = "";
    for (auto x : shape) {
        shapeStr.append(sep);
        shapeStr.append(std::to_string(x));
        sep = ", ";
    }
    shapeStr.push_back(']');
    return shapeStr;
}

bool CheckInputShape(
    gert::TilingContext* context, const char* variableName, const gert::Shape& shape,
    const std::vector<int64_t>& expectedShape)
{
    auto shapeLen = shape.GetDimNum();
    if (unlikely(shapeLen != expectedShape.size())) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, variableName,
            std::to_string(shapeLen),
            "The shape dim of " + std::string(variableName) + " must be " + std::to_string(expectedShape.size()));
        return false;
    }
    size_t i = 0;
    for (auto dim : expectedShape) {
        if (unlikely(dim != shape.GetDim(i++))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, variableName,
                Ops::Base::ToString(shape),
                "Shape does not match the expected value " + ToShapeString(expectedShape));
            return false;
        }
    }
    return true;
}

bool IsInputShapeValid([[maybe_unused]] gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    if (unlikely(params.mSize < MIN_M_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x",
            "M=" + std::to_string(params.mSize),
            "M of x must be >= " + std::to_string(MIN_M_SIZE));
        return false;
    }
    if (unlikely(params.kSize < MIN_K_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x",
            "K=" + std::to_string(params.kSize),
            "K of x must be >= " + std::to_string(MIN_K_SIZE));
        return false;
    }
    if (unlikely(params.nSize < MIN_N_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight",
            "N=" + std::to_string(params.nSize),
            "N of weight must be >= " + std::to_string(MIN_N_SIZE));
        return false;
    }
    if (unlikely(params.kSize % ALIGN_SIZE_K != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x",
            "K=" + std::to_string(params.kSize),
            "K of x must be aligned to " + std::to_string(ALIGN_SIZE_K));
        return false;
    }
    if (unlikely(params.nSize % ALIGN_SIZE_N != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight",
            "N=" + std::to_string(params.nSize),
            "N of weight must be aligned to " + std::to_string(ALIGN_SIZE_N));
        return false;
    }
    if (unlikely(params.groupNum < 1 || params.groupNum > MAX_GROUP_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight",
            "groupNum=" + std::to_string(params.groupNum),
            "groupNum of weight must be in range [1, " + std::to_string(MAX_GROUP_NUM) + "]");
        return false;
    }
    return true;
}

bool CheckInputs(gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    if (!IsInputShapeValid(context, params)) {
        return false;
    }

    if (!CheckInputFormats(context, params)) {
        return false;
    }

    auto* xScaleShapePtr = context.GetInputShape(X_SCALE_INDEX);
    auto* groupListShapePtr = context.GetInputShape(GROUPLIST_INDEX);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, "weight"),
        OP_LOGE(OP_NAME, "Check dynamic input shape of weight failed."), return false);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_SCALE_INDEX, "weightScale"),
        OP_LOGE(OP_NAME, "Check dynamic input shape of weightScale failed."), return false);
    auto* weightShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto* weightScaleShapePtr = context.GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);

    OP_CHECK_IF(xScaleShapePtr == nullptr,
        OP_LOGE(OP_NAME, "xScaleShape cannot be null"), return false);
    OP_CHECK_IF(groupListShapePtr == nullptr,
        OP_LOGE(OP_NAME, "groupListShape cannot be null"), return false);

    auto& xScaleShape = xScaleShapePtr->GetOriginShape();
    auto& groupListShape = groupListShapePtr->GetOriginShape();
    auto& weightScaleShape = weightScaleShapePtr->GetOriginShape();

    if (params.wFormat == ge::FORMAT_FRACTAL_NZ || params.wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        const gert::Shape &wStorageShape = weightShapePtr->GetStorageShape();
        const std::vector<int64_t> expectedWeightShape = {
            static_cast<int64_t>(params.groupNum),
            GroupedMatmul::CeilDiv<int64_t>(params.kSize, C0_SIZE),
            GroupedMatmul::CeilDiv<int64_t>(params.nSize, CUBE_BLOCK),
            CUBE_BLOCK,
            C0_SIZE
        };
        if (!CheckInputShape(&context, "weight", wStorageShape, expectedWeightShape)) {
            return false;
        }
    }

    int64_t scaleKSize = GroupedMatmul::CeilDiv<int64_t>(params.kSize, MX_SCALE_BLOCK_SIZE);
    const std::vector<int64_t> expectedXScaleShape = {
        static_cast<int64_t>(params.mSize),
        scaleKSize,
        MX_INNER_DIM
    };
    if (!CheckInputShape(&context, "xScale", xScaleShape, expectedXScaleShape)) {
        return false;
    }
    if (!CheckInputShape(&context, "groupList", groupListShape, {params.groupNum})) {
        return false;
    }
    const std::vector<int64_t> expectedWeightScaleShape = {
        params.groupNum,
        params.nSize,
        scaleKSize,
        MX_INNER_DIM
    };
    if (!CheckInputShape(&context, "weightScale", weightScaleShape, expectedWeightScaleShape)) {
        return false;
    }
    return true;
}

void PrintInputParams(gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    OP_LOGD(context.GetNodeName(),
            "Input params: mSize=%ld, kSize=%ld, nSize=%ld, groupNum=%ld, "
            "wTrans=%s, groupListType=%ld",
            params.mSize, params.kSize, params.nSize, params.groupNum,
            params.wTrans ? "true" : "false", params.groupListType);
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetPlatformInfo()
{
    const auto *compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE(OP_NAME, "CompileInfo is null"),
                return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;
    OP_LOGI(context_,
        "Compile info: aicNum(%u) aivNum(%lu) ubSize(%lu) baseM(%lu) baseN(%lu) supportL12BtBf16(%lu).",
        compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_, compileInfoPtr_->ubSize_,
        compileInfoPtr_->baseM_, compileInfoPtr_->baseN_, compileInfoPtr_->supportL12BtBf16);

    OP_CHECK_IF(!CheckCoreNum(context_, compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_),
        OP_LOGE(OP_NAME, "Invalid core number ratio"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(!GetContext(inputParams_, *context_),
        OP_LOGE(OP_NAME, "Failed to get context."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckContext(*context_, sizeof(GMMSQArch35Tiling::GMMSQWeightQuantTilingData)) != ge::GRAPH_SUCCESS,
        OP_LOGE(OP_NAME, "Invalid context."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!GetAttrs(inputParams_, *context_),
        OP_LOGE(OP_NAME, "Failed to GetAttrs."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckAttrs(*context_, inputParams_),
        OP_LOGE(OP_NAME, "Failed to check attrs."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckDtypes(*context_, inputParams_),
        OP_LOGE(OP_NAME, "Failed to check dtypes."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!GetInputs(inputParams_, *context_),
        OP_LOGE(OP_NAME, "Failed to GetInputs."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputs(*context_, inputParams_),
        OP_LOGE(OP_NAME, "Failed to check inputs."),
        return ge::GRAPH_FAILED);

    PrintInputParams(*context_, inputParams_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start setting tiling data.");

    tilingData_.groupListType = inputParams_.groupListType;
    tilingData_.coreNum = compileInfoPtr_->aicNum_;
    tilingData_.groupNum = inputParams_.groupNum;
    tilingData_.groupSize = MX_BLOCK_SIZE;
    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;

    OP_LOGD(context_->GetNodeName(),
            "Tiling data set: coreNum=%u, groupNum=%ld, "
            "kSize=%ld, nSize=%ld, groupListType=%u.",
            tilingData_.coreNum, tilingData_.groupNum, tilingData_.kSize, tilingData_.nSize,
            tilingData_.groupListType);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2WeightQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<int64_t>(inputParams_.wTrans), 0L);
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(OP_NAME, "workspaces is nullptr."),
                return ge::GRAPH_FAILED);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start posting tiling data, blockDim=%u, tilingSize=%zu.",
            tilingData_.coreNum, sizeof(tilingData_));

    context_->SetBlockDim(tilingData_.coreNum);
    const auto *rawTiling = context_->GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(OP_NAME, "RawTilingData is nullptr."),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(const_cast<void *>(rawTiling->GetData()), rawTiling->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(OP_NAME, "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));

    OP_LOGD(context_->GetNodeName(), "Tiling data posted successfully.");
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulSwigluQuantV2WeightQuantTiling::Reset()
{
    tilingData_ = GMMSQArch35Tiling::GMMSQWeightQuantTilingData();
    const auto *rawTiling = context_->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        return;
    }
    OP_CHECK_IF(memset_s(const_cast<void *>(rawTiling->GetData()), rawTiling->GetCapacity(), 0,
                         rawTiling->GetCapacity()) != EOK,
                OP_LOGE(OP_NAME, "Fail to clear tiling data"), return);
}
} // namespace optiling
