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
        OP_LOGE(context->GetNodeName(), "Invalid aicNum[%u] or aivNum[%u], expect greater than 0",
            aicNum, aivNum),
        return false);

    OP_CHECK_IF(aicNum * AIC_AIV_CORE_RATIO != aivNum,
        OP_LOGE(context->GetNodeName(),
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
        OP_LOGE(&context, "Dynamic input shape of %s is nullptr.", inputName),
        return false);

    const auto &storageShape = shapePtr->GetStorageShape();
    OP_CHECK_IF(storageShape.GetShapeSize() == 0,
        OP_LOGE(&context, "Not yet support empty tensor for %s.", inputName),
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
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "GetRawTilingData can not be empty."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTiling->GetData() == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "RawTilingData should not be a null pointer."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTiling->GetCapacity() < tilingDataSize,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(),
            "context tiling data capacity %zu < actual tiling data size %zu.",
            rawTiling->GetCapacity(), tilingDataSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckContext(gert::TilingContext& context, uint64_t tilingDataSize)
{
    const auto *attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "Function context.GetAttrs() failed!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.GetInputDesc(X_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "InputDesc x is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(X_SCALE_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "InputDesc xScale is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(GROUPLIST_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "InputDesc groupList is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "InputDesc weight is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_SCALE_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "InputDesc weightScale is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetOutputDesc(Y_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "OutputDesc y is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetOutputDesc(Y_SCALE_INDEX) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "OutputDesc yScale is nullptr."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!IsDynamicInputEmpty(context, WEIGHT_ASSIST_MATRIX_INDEX),
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(),
        "weightAssistMatrix is not supported, please pass a null pointer or an empty list."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetOptionalInputDesc(BIAS_INDEX) != nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(), "bias is not supported, please pass a null pointer."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.GetOptionalInputDesc(SMOOTH_SCALE_INDEX) != nullptr,
        OPS_REPORT_CUBE_INNER_ERR(context.GetNodeName(),
        "smoothScale is not supported, please pass a null pointer."),
        return ge::GRAPH_FAILED);

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
    OP_CHECK_IF(params.dequantMode != MX_QUANT_MODE,
        OP_LOGE(params.opName, "dequantMode must be %ld (MX quant), got %ld.", MX_QUANT_MODE, params.dequantMode),
        return false);

    OP_CHECK_IF(params.dequantDtype != ge::DataType::DT_FLOAT,
        OP_LOGE(params.opName, "dequantDtype must be %ld (DT_FLOAT), got %ld.",
            ge::DataType::DT_FLOAT, params.dequantDtype),
        return false);

    OP_CHECK_IF(params.quantMode != MX_QUANT_MODE,
        OP_LOGE(params.opName, "quantMode must be %ld (MX quant), got %ld.", MX_QUANT_MODE, params.quantMode),
        return false);

    OP_CHECK_IF(params.quantDtype != ge::DataType::DT_FLOAT8_E4M3FN,
        OP_LOGE(params.opName, "quantDtype must be %ld (DT_FLOAT8_E4M3FN), got %ld.", ge::DataType::DT_FLOAT8_E4M3FN,
            params.quantDtype),
        return false);

    OP_CHECK_IF(!params.wTrans,
        OP_LOGE(params.opName, "transposeWeight must be true for MxA8W4."),
        return false);

    OP_CHECK_IF(params.groupListType != 0 && params.groupListType != 1,
        OP_LOGE(params.opName, "groupListType must be 0 (cumsum) or 1 (count), got %ld.", params.groupListType),
        return false);

    return true;
}

bool CheckDtypes(const gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);

    OP_CHECK_IF(xDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN,
        OP_LOGE(params.opName, "x dtype must be FLOAT8_E4M3FN, got %s.",
            ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()).c_str()),
        return false);

    OP_CHECK_IF(wDesc->GetDataType() != ge::DT_FLOAT4_E2M1,
        OP_LOGE(params.opName, "weight dtype must be FLOAT4_E2M1, got %s.",
            ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType()).c_str()),
        return false);

    OP_CHECK_IF(xScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(params.opName, "xScale dtype must be FLOAT8_E8M0, got %s.",
            ge::TypeUtils::DataTypeToSerialString(xScaleDesc->GetDataType()).c_str()),
        return false);

    OP_CHECK_IF(wScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(params.opName, "weightScale dtype must be FLOAT8_E8M0, got %s.",
            ge::TypeUtils::DataTypeToSerialString(wScaleDesc->GetDataType()).c_str()),
        return false);

    OP_CHECK_IF(groupListDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(params.opName, "groupList dtype must be INT64, got %s.",
            ge::TypeUtils::DataTypeToSerialString(groupListDesc->GetDataType()).c_str()),
        return false);

    return true;
}

bool GetInputs(GMMSQWeightQuantInputParams& params, const gert::TilingContext& context)
{
    const auto *wDesc = context.GetInputDesc(WEIGHT_INDEX);
    params.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(params.wFormat != ge::FORMAT_FRACTAL_NZ && params.wFormat != ge::FORMAT_FRACTAL_NZ_C0_32,
        OP_LOGE(&context, "Weight format is only support to be FRACTAL_NZ or FRACTAL_NZ_C0_32, but got %s(%d)",
            ge::TypeUtils::FormatToAscendString(params.wFormat).GetString(), params.wFormat),
        return false);

    const auto *xShapePtr = context.GetInputShape(X_INDEX);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, "weight"),
        OP_LOGE(&context, "Check dynamic input shape of weight failed."), return false);
    const auto *wShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(xShapePtr == nullptr || wShapePtr == nullptr,
                OP_LOGE(&context, "Required input shape is nullptr."), return false);

    const gert::Shape &xShape = xShapePtr->GetOriginShape();
    const gert::Shape &wShape = wShapePtr->GetOriginShape();
    auto xShapeLen = xShape.GetDimNum();
    auto wShapeLen = wShape.GetDimNum();
    OP_CHECK_IF(xShape.GetShapeSize() == 0,
        OP_LOGE(&context, "Not yet support empty tensor for x"), return false);
    OP_CHECK_IF(xShapeLen != DIM_NUM_X,
        OP_LOGE(&context, "input dimension of x should be %zu, but got %zu", DIM_NUM_X, xShapeLen),
        return false);
    OP_CHECK_IF(wShapeLen != DIM_NUM_WEIGHT_ND,
        OP_LOGE(&context, "input dimension of weight should be %zu, but got %zu", DIM_NUM_WEIGHT_ND, wShapeLen),
        return false);

    // - 1 为内轴
    auto xInner = xShape.GetDim(DIM_NUM_X - 1);
    auto wInner = wShape.GetDim(DIM_NUM_WEIGHT_ND - 1);
    OP_CHECK_IF(xInner != wInner,
        OP_LOGE(&context, "Inputs dimension is not match, x kSize=%ld, weight kSize=%ld", xInner, wInner),
        return false);
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
    OP_CHECK_IF(xFormat != ge::FORMAT_ND,
        OP_LOGE(params.opName, "x format must be ND, got %s.",
            ge::TypeUtils::FormatToSerialString(xFormat).c_str()),
        return false);

    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    auto xScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(xScaleFormat != ge::FORMAT_ND && xScaleFormat != ge::FORMAT_NCL,
        OP_LOGE(params.opName, "xScale format must be ND or NCL, but got %s.",
            ge::TypeUtils::FormatToSerialString(xScaleFormat).c_str()),
        return false);

    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);
    auto groupListFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(groupListDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(groupListFormat != ge::FORMAT_ND,
        OP_LOGE(params.opName, "groupList format must be ND, but got %s.",
            ge::TypeUtils::FormatToSerialString(groupListFormat).c_str()),
        return false);

    const auto *wScaleDesc = context.GetInputDesc(WEIGHT_SCALE_INDEX);
    auto wScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wScaleDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(wScaleFormat != ge::FORMAT_ND && wScaleFormat != ge::FORMAT_NCHW,
        OP_LOGE(params.opName, "weightScale format must be ND or NCHW, but got %s.",
            ge::TypeUtils::FormatToSerialString(wScaleFormat).c_str()),
        return false);

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
    OP_CHECK_IF(
        shapeLen != expectedShape.size(),
        OP_LOGE(
            context, "input %s dimension should be %zu, but got %zu", variableName, expectedShape.size(), shapeLen),
        return false);
    size_t i = 0;
    for (auto dim : expectedShape) {
        OP_CHECK_IF(
            dim != shape.GetDim(i++),
            OP_LOGE(context, "Check input %s shape failed, expected %s, but got %s", variableName,
                ToShapeString(expectedShape).c_str(), Ops::Base::ToString(shape).c_str()),
            return false);
    }
    return true;
}

bool IsInputShapeValid([[maybe_unused]] gert::TilingContext& context, const GMMSQWeightQuantInputParams& params)
{
    OP_CHECK_IF(params.mSize < MIN_M_SIZE,
        OP_LOGE(params.opName, "M size %ld < min %ld.", params.mSize, MIN_M_SIZE),
        return false);
    OP_CHECK_IF(params.kSize < MIN_K_SIZE,
        OP_LOGE(params.opName, "K size %ld < min %ld.", params.kSize, MIN_K_SIZE),
        return false);
    OP_CHECK_IF(params.nSize < MIN_N_SIZE,
        OP_LOGE(params.opName, "N size %ld < min %ld.", params.nSize, MIN_N_SIZE),
        return false);
    OP_CHECK_IF(params.kSize % ALIGN_SIZE_K != 0,
        OP_LOGE(params.opName, "K size %ld not %u aligned.", params.kSize, ALIGN_SIZE_K),
        return false);
    OP_CHECK_IF(params.nSize % ALIGN_SIZE_N != 0,
        OP_LOGE(params.opName, "N size %ld not %u aligned.", params.nSize, ALIGN_SIZE_N),
        return false);
    OP_CHECK_IF(params.groupNum < 1 || params.groupNum > MAX_GROUP_NUM,
        OP_LOGE(params.opName, "The input range for the group num is [1, %ld], but got %ld",
            MAX_GROUP_NUM, params.groupNum),
        return false);
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
        OP_LOGE(&context, "Check dynamic input shape of weight failed."), return false);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_SCALE_INDEX, "weightScale"),
        OP_LOGE(&context, "Check dynamic input shape of weightScale failed."), return false);
    auto* weightShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto* weightScaleShapePtr = context.GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);

    OP_CHECK_IF(xScaleShapePtr == nullptr,
        OP_LOGE(params.opName, "xScaleShape should not be null"), return false);
    OP_CHECK_IF(groupListShapePtr == nullptr,
        OP_LOGE(params.opName, "groupListShape should not be null"), return false);

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
                OPS_REPORT_CUBE_INNER_ERR("GroupedMatmulSwigluQuantV2", "CompileInfo is null"),
                return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;
    OP_LOGI(context_,
        "Compile info: aicNum(%u) aivNum(%lu) ubSize(%lu) baseM(%lu) baseN(%lu) supportL12BtBf16(%lu).",
        compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_, compileInfoPtr_->ubSize_,
        compileInfoPtr_->baseM_, compileInfoPtr_->baseN_, compileInfoPtr_->supportL12BtBf16);

    OP_CHECK_IF(!CheckCoreNum(context_, compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_),
        OP_LOGE(context_->GetNodeName(), "Invalid core number ratio"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(!GetContext(inputParams_, *context_),
        OP_LOGE(context_->GetNodeName(), "Failed to get context."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckContext(*context_, sizeof(GMMSQArch35Tiling::GMMSQWeightQuantTilingData)) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Invalid context."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!GetAttrs(inputParams_, *context_),
        OP_LOGE(context_->GetNodeName(), "Failed to GetAttrs."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckAttrs(*context_, inputParams_),
        OP_LOGE(context_->GetNodeName(), "Failed to check attrs."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckDtypes(*context_, inputParams_),
        OP_LOGE(context_->GetNodeName(), "Failed to check dtypes."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!GetInputs(inputParams_, *context_),
        OP_LOGE(context_->GetNodeName(), "Failed to GetInputs."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputs(*context_, inputParams_),
        OP_LOGE(context_->GetNodeName(), "Failed to check inputs."),
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
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is nullptr."),
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
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(context_->GetNodeName(), "RawTilingData is nullptr."),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(const_cast<void *>(rawTiling->GetData()), rawTiling->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
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
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
}
} // namespace optiling
