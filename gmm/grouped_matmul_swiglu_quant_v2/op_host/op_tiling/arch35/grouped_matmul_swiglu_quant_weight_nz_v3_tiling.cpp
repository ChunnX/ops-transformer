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
 * \file grouped_matmul_swiglu_quant_weight_nz_v3_tiling.cpp
 * \brief Ascend 950 MXFP8 weight-NZ tiling for GroupedMatmulSwigluQuantV3.
 */

#include <algorithm>
#include <cmath>

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "../grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#include "grouped_matmul_swiglu_quant_tensor_api_tiling_common.h"

using namespace ge;
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GmmConstant;

namespace optiling {
namespace GroupedMatmulSwigluQuantV3Tiling {
namespace {

size_t GetDynamicInputCount(const gert::TilingContext *context, size_t inputIndex)
{
    size_t count = 0;
    while (context->GetDynamicInputShape(inputIndex, count) != nullptr) {
        ++count;
    }
    return count;
}

bool IsFinitePositive(float value)
{
    return std::isfinite(value) && value > 0.0F;
}

bool IsDenseFormat(ge::Format format)
{
    return format == ge::FORMAT_ND || format == ge::FORMAT_NCL || format == ge::FORMAT_NCHW;
}

} // namespace

void GroupedMatmulSwigluQuantV3Tiling950::Reset()
{
    auto *rawTilingData = context_ == nullptr ? nullptr : context_->GetRawTilingData();
    tilingData_.SetDataPtr(rawTilingData == nullptr ? nullptr : rawTilingData->GetData());
    transposeWeight_ = false;
    swigluMode_ = SWIGLU_MODE;
    clampLimit_ = 7.0F;
    gluAlpha_ = 1.702F;
    gluBias_ = 1.0F;
    scaleAlg_ = SCALE_ALG_OCP;
    dstTypeMax_ = DEFAULT_DST_TYPE_MAX;
    roundMode_ = 0;
    aivNum_ = 0;
    useTensorApi_ = false;
}

ge::graphStatus GroupedMatmulSwigluQuantV3Tiling950::GetPlatformInfo()
{
    if (context_->GetPlatformInfo() != nullptr) {
        auto status = GroupedQmmTiling::GetPlatformInfo();
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
        auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        aivNum_ = platform.GetCoreNumAiv();
        return ge::GRAPH_SUCCESS;
    }

    const auto *compileInfo = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compileInfo is nullptr"),
                return ge::GRAPH_FAILED);
    aicoreParams_.aicNum = compileInfo->aicNum_;
    aicoreParams_.ubSize = compileInfo->ubSize_;
    aicoreParams_.l1Size = compileInfo->l1Size_;
    aicoreParams_.l0aSize = compileInfo->l0ASize_;
    aicoreParams_.l0bSize = compileInfo->l0BSize_;
    aicoreParams_.l0cSize = compileInfo->l0CSize_;
    aivNum_ = compileInfo->aivNum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV3Tiling950::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulSwigluQuantV3Tiling950::GetAttrInt(const gert::RuntimeAttrs *attrs, uint32_t index,
                                                     int64_t &value) const
{
    const int64_t *ptr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(index);
    if (ptr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "attribute[%u] is nullptr", index);
        return false;
    }
    value = *ptr;
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::GetAttrBool(const gert::RuntimeAttrs *attrs, uint32_t index,
                                                      bool &value) const
{
    const bool *ptr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(index);
    if (ptr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "attribute[%u] is nullptr", index);
        return false;
    }
    value = *ptr;
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::GetAttrFloat(const gert::RuntimeAttrs *attrs, uint32_t index,
                                                       float &value) const
{
    const float *ptr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(index);
    if (ptr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "attribute[%u] is nullptr", index);
        return false;
    }
    value = *ptr;
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::GetAttrString(const gert::RuntimeAttrs *attrs, uint32_t index,
                                                        std::string &value) const
{
    const char *ptr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<char>(index);
    if (ptr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "attribute[%u] is nullptr", index);
        return false;
    }
    value = ptr;
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::AnalyzeAttrs()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr"), return false);

    int64_t dequantMode = 0;
    int64_t dequantDtype = 0;
    int64_t quantMode = 0;
    int64_t quantDtype = 0;
    bool transposeWeight = false;
    int64_t groupListType = 0;
    int64_t swigluMode = SWIGLU_MODE;
    int64_t scaleAlg = SCALE_ALG_OCP;
    float dstTypeMax = DEFAULT_DST_TYPE_MAX;
    std::string roundMode;
    OP_CHECK_IF(!GetAttrInt(attrs, ATTR_INDEX_DEQUANT_MODE, dequantMode) ||
                    !GetAttrInt(attrs, ATTR_INDEX_DEQUANT_DTYPE, dequantDtype) ||
                    !GetAttrInt(attrs, ATTR_INDEX_QUANT_MODE, quantMode) ||
                    !GetAttrInt(attrs, ATTR_INDEX_QUANT_DTYPE, quantDtype) ||
                    !GetAttrBool(attrs, ATTR_INDEX_TRANSPOSE_WEIGHT, transposeWeight) ||
                    !GetAttrInt(attrs, ATTR_INDEX_GROUPLIST_TYPE, groupListType) ||
                    !GetAttrInt(attrs, ATTR_INDEX_SWIGLU_MODE, swigluMode) ||
                    !GetAttrFloat(attrs, ATTR_INDEX_CLAMP_LIMIT, clampLimit_) ||
                    !GetAttrFloat(attrs, ATTR_INDEX_GLU_ALPHA, gluAlpha_) ||
                    !GetAttrFloat(attrs, ATTR_INDEX_GLU_BIAS, gluBias_) ||
                    !GetAttrString(attrs, ATTR_INDEX_ROUND_MODE, roundMode) ||
                    !GetAttrInt(attrs, ATTR_INDEX_SCALE_ALG, scaleAlg) ||
                    !GetAttrFloat(attrs, ATTR_INDEX_DST_TYPE_MAX, dstTypeMax),
                OP_LOGE(context_->GetNodeName(), "failed to read V3 attributes"), return false);

    OP_CHECK_IF(dequantMode != MX_QUANT_MODE || quantMode != MX_QUANT_MODE || dequantDtype != FLOAT_DEQUANT_DTYPE,
                OP_LOGE(context_->GetNodeName(), "V3 only supports dequant_mode=2, dequant_dtype=0 and quant_mode=2"),
                return false);
    OP_CHECK_IF(groupListType != 0 && groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "group_list_type must be 0 or 1, but got %ld", groupListType),
                return false);
    OP_CHECK_IF(quantDtype != static_cast<int64_t>(inputParams_.outDataDtype),
                OP_LOGE(context_->GetNodeName(), "quant_dtype must match the FP8 output dtype"), return false);
    OP_CHECK_IF(swigluMode != SWIGLU_MODE,
                OP_LOGE(context_->GetNodeName(), "swiglu_mode currently only supports 2, but got %ld", swigluMode),
                return false);
    OP_CHECK_IF(!std::isfinite(gluAlpha_) || !std::isfinite(gluBias_),
                OP_LOGE(context_->GetNodeName(), "glu_alpha and glu_bias must be finite"), return false);
    OP_CHECK_IF(!IsFinitePositive(clampLimit_),
                OP_LOGE(context_->GetNodeName(), "clamp_limit must be finite and positive for swiglu_mode 2"),
                return false);
    OP_CHECK_IF(roundMode != DEFAULT_ROUND_MODE,
                OP_LOGE(context_->GetNodeName(), "round_mode must be rint, but got %s", roundMode.c_str()),
                return false);
    OP_CHECK_IF(
        scaleAlg != SCALE_ALG_OCP && scaleAlg != SCALE_ALG_CUBLAS,
        OP_LOGE(context_->GetNodeName(), "scale_alg must be 0 or 1 for the MXFP8 output path, but got %ld", scaleAlg),
        return false);
    OP_CHECK_IF(!std::isfinite(dstTypeMax) || dstTypeMax != DEFAULT_DST_TYPE_MAX,
                OP_LOGE(context_->GetNodeName(), "dst_type_max must be 0.0 for the MXFP8 output path"), return false);

    swigluMode_ = swigluMode;
    // roundMode is validated above and the only supported value is rint.
    // Encode it explicitly so the V3 tiling payload carries the effective
    // quantization attributes even though the current kernel has one mode.
    roundMode_ = 0;
    scaleAlg_ = static_cast<uint8_t>(scaleAlg);
    dstTypeMax_ = dstTypeMax;
    inputParams_.transA = false;
    transposeWeight_ = transposeWeight;
    inputParams_.transB = transposeWeight_;
    inputParams_.groupListType = static_cast<int8_t>(groupListType);
    inputParams_.groupType = GroupedMatmul::SPLIT_M;
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto xScaleDesc = context_->GetInputDesc(X_SCALE_INDEX);
    auto groupListDesc = context_->GetInputDesc(GROUPLIST_INDEX);
    auto weightDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto weightScaleDesc = context_->GetDynamicInputDesc(WEIGHT_SCALE_INDEX, 0);
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    auto yScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || xScaleDesc == nullptr || groupListDesc == nullptr || weightDesc == nullptr ||
                    weightScaleDesc == nullptr || yDesc == nullptr || yScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "required descriptor is nullptr"), return false);

    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = weightDesc->GetDataType();
    inputParams_.scaleDtype = weightScaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = xScaleDesc->GetDataType();
    inputParams_.outDataDtype = yDesc->GetDataType();
    inputParams_.outScaleDtype = yScaleDesc->GetDataType();
    inputParams_.cDtype = ge::DT_FLOAT;
    inputParams_.biasDtype = ge::DT_FLOAT;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    inputParams_.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetFormat().GetStorageFormat()));
    inputParams_.scaleFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScaleDesc->GetFormat().GetStorageFormat()));
    inputParams_.cFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetFormat().GetStorageFormat()));
    inputParams_.aQuantMode = QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = QuantMode::MX_PERGROUP_MODE;

    OP_CHECK_IF(inputParams_.aDtype != ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE(context_->GetNodeName(), "x and weight must be FLOAT8_E4M3FN"), return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context_->GetNodeName(), "x_scale and weight_scale must be FLOAT8_E8M0"), return false);
    OP_CHECK_IF(groupListDesc->GetDataType() != ge::DT_INT64,
                OP_LOGE(context_->GetNodeName(), "group_list must be INT64"), return false);
    OP_CHECK_IF(inputParams_.outDataDtype != ge::DT_FLOAT8_E4M3FN || inputParams_.outScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context_->GetNodeName(), "V3 output must be FLOAT8_E4M3FN with FLOAT8_E8M0 scale"),
                return false);
    OP_CHECK_IF(
        inputParams_.aFormat != ge::FORMAT_ND || inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ ||
            inputParams_.scaleFormat != ge::FORMAT_ND || inputParams_.cFormat != ge::FORMAT_ND ||
            static_cast<ge::Format>(ge::GetPrimaryFormat(groupListDesc->GetFormat().GetStorageFormat())) !=
                ge::FORMAT_ND ||
            !IsDenseFormat(static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()))) ||
            !IsDenseFormat(static_cast<ge::Format>(ge::GetPrimaryFormat(yScaleDesc->GetFormat().GetStorageFormat()))),
        OP_LOGE(context_->GetNodeName(), "V3 requires x/scale/output ND and weight FRACTAL_NZ"), return false);
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::CheckShapeAndScale(const gert::Shape &xShape, const gert::Shape &xScaleShape,
                                                             const gert::Shape &weightShape,
                                                             const gert::Shape &weightStorageShape,
                                                             const gert::Shape &weightScaleShape,
                                                             const gert::Shape &groupListShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() != X_DIM_NUM, OP_LOGE(context_->GetNodeName(), "x must be a 2D shape"),
                return false);
    OP_CHECK_IF(groupListShape.GetDimNum() != GROUP_LIST_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "group_list must be a 1D shape"), return false);
    OP_CHECK_IF(weightShape.GetDimNum() != WEIGHT_VIEW_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "weight view must be a 3D shape"), return false);

    const int64_t m = xShape.GetDim(0);
    const int64_t k = xShape.GetDim(1);
    const int64_t e = groupListShape.GetDim(0);
    const int64_t n = transposeWeight_ ? weightShape.GetDim(1) : weightShape.GetDim(2);
    const int64_t weightK = transposeWeight_ ? weightShape.GetDim(2) : weightShape.GetDim(1);
    const int64_t kScale = (k + MX_GROUP_SIZE_V3 - 1) / MX_GROUP_SIZE_V3;
    OP_CHECK_IF(m <= 0 || k <= 0, OP_LOGE(context_->GetNodeName(), "x must be non-empty"), return false);
    OP_CHECK_IF(e <= 0 || e > MAX_GROUP_NUM, OP_LOGE(context_->GetNodeName(), "group_list length must be in [1, 1024]"),
                return false);
    OP_CHECK_IF(weightShape.GetDim(0) != e || weightK != k || n <= 0 || n % 64 != 0 || n % SWIGLU_SPLIT != 0,
                OP_LOGE(context_->GetNodeName(),
                        "weight view must be (E, K, N) or transposed (E, N, K), with N positive and 64 aligned"),
                return false);
    // FRACTAL_NZ follows the V2 MXFP8 contract: non-transposed (E, K, N)
    // views use storage (E, ceil(N/32), ceil(K/16), 16, 32), while
    // transposed (E, N, K) views use (E, ceil(K/32), ceil(N/16), 16, 32).
    const int64_t expectedStorageDim1 =
        transposeWeight_ ? (k + NZ_C0_SIZE - 1) / NZ_C0_SIZE : (n + NZ_C0_SIZE - 1) / NZ_C0_SIZE;
    const int64_t expectedStorageDim2 =
        transposeWeight_ ? (n + NZ_K0_SIZE - 1) / NZ_K0_SIZE : (k + NZ_K0_SIZE - 1) / NZ_K0_SIZE;
    OP_CHECK_IF(weightStorageShape.GetDimNum() != WEIGHT_STORAGE_DIM_NUM || weightStorageShape.GetDim(0) != e ||
                    weightStorageShape.GetDim(1) != expectedStorageDim1 ||
                    weightStorageShape.GetDim(2) != expectedStorageDim2 || weightStorageShape.GetDim(3) != NZ_K0_SIZE ||
                    weightStorageShape.GetDim(4) != NZ_C0_SIZE,
                OP_LOGE(context_->GetNodeName(),
                        "weight storage shape is not the expected FRACTAL_NZ shape for the transpose setting"),
                return false);
    OP_CHECK_IF(xScaleShape.GetDimNum() != X_SCALE_DIM_NUM || xScaleShape.GetDim(0) != m ||
                    xScaleShape.GetDim(1) != kScale || xScaleShape.GetDim(2) != 2,
                OP_LOGE(context_->GetNodeName(), "x_scale must be (M, ceil(K/64), 2)"), return false);
    OP_CHECK_IF(weightScaleShape.GetDimNum() != WEIGHT_SCALE_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "weight_scale must be 4D"), return false);
    const int64_t weightScaleN = transposeWeight_ ? weightScaleShape.GetDim(1) : weightScaleShape.GetDim(2);
    const int64_t weightScaleK = transposeWeight_ ? weightScaleShape.GetDim(2) : weightScaleShape.GetDim(1);
    OP_CHECK_IF(weightScaleShape.GetDimNum() != WEIGHT_SCALE_DIM_NUM || weightScaleShape.GetDim(0) != e ||
                    weightScaleN != n || weightScaleK != kScale || weightScaleShape.GetDim(3) != 2,
                OP_LOGE(context_->GetNodeName(),
                        "weight_scale must be (E, ceil(K/64), N, 2) or transposed (E, N, ceil(K/64), 2)"),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::CheckOutputShape() const
{
    auto xStorage = context_->GetInputShape(X_INDEX);
    auto weightStorage = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto outputStorage = context_->GetOutputShape(Y_INDEX);
    auto outputScaleStorage = context_->GetOutputShape(Y_SCALE_INDEX);
    OP_CHECK_IF(
        xStorage == nullptr || weightStorage == nullptr || outputStorage == nullptr || outputScaleStorage == nullptr,
        OP_LOGE(context_->GetNodeName(), "shape is nullptr"), return false);
    const auto &xShape = xStorage->GetOriginShape();
    const auto &weightShape = weightStorage->GetOriginShape();
    const auto &yShape = outputStorage->GetOriginShape();
    const auto &yScaleShape = outputScaleStorage->GetOriginShape();
    OP_CHECK_IF(xShape.GetDimNum() != X_DIM_NUM || weightShape.GetDimNum() != WEIGHT_VIEW_DIM_NUM ||
                    yShape.GetDimNum() != OUTPUT_DIM_NUM || yScaleShape.GetDimNum() != OUTPUT_SCALE_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "output shape rank is invalid"), return false);
    const int64_t m = xShape.GetDim(0);
    const int64_t n = transposeWeight_ ? weightShape.GetDim(1) : weightShape.GetDim(2);
    const int64_t outputN = n / SWIGLU_SPLIT;
    const int64_t outputScaleN = (outputN + MX_GROUP_SIZE_V3 - 1) / MX_GROUP_SIZE_V3;
    OP_CHECK_IF(yShape.GetDimNum() != OUTPUT_DIM_NUM || yShape.GetDim(0) != m || yShape.GetDim(1) != outputN,
                OP_LOGE(context_->GetNodeName(), "output shape is not (M, N/2)"), return false);
    OP_CHECK_IF(yScaleShape.GetDimNum() != OUTPUT_SCALE_DIM_NUM || yScaleShape.GetDim(0) != m ||
                    yScaleShape.GetDim(1) != outputScaleN || yScaleShape.GetDim(2) != 2,
                OP_LOGE(context_->GetNodeName(), "output_scale shape is not (M, ceil((N/2)/64), 2)"), return false);
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::AnalyzeInputs()
{
    const auto *xStorage = context_->GetInputShape(X_INDEX);
    const auto *xScaleStorage = context_->GetInputShape(X_SCALE_INDEX);
    const auto *groupListStorage = context_->GetInputShape(GROUPLIST_INDEX);
    const auto *weightStorage = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    const auto *weightScaleStorage = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_IF(xStorage == nullptr || xScaleStorage == nullptr || groupListStorage == nullptr ||
                    weightStorage == nullptr || weightScaleStorage == nullptr,
                OP_LOGE(context_->GetNodeName(), "required input shape is nullptr"), return false);
    OP_CHECK_IF(GetDynamicInputCount(context_, WEIGHT_INDEX) != SINGLE_TENSOR_COUNT ||
                    GetDynamicInputCount(context_, WEIGHT_SCALE_INDEX) != SINGLE_TENSOR_COUNT,
                OP_LOGE(context_->GetNodeName(), "V3 only supports a single weight and weight_scale tensor"),
                return false);

    const auto &xShape = xStorage->GetOriginShape();
    const auto &xScaleShape = xScaleStorage->GetOriginShape();
    const auto &groupListShape = groupListStorage->GetOriginShape();
    const auto &weightShape = weightStorage->GetOriginShape();
    const auto &weightNzShape = weightStorage->GetStorageShape();
    const auto &weightScaleShape = weightScaleStorage->GetOriginShape();
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(context_->GetNodeName(), "SetGroupNum failed"), return false);
    OP_CHECK_IF(!SetMKN(xShape, weightShape), OP_LOGE(context_->GetNodeName(), "SetMKN failed"), return false);
    OP_CHECK_IF(!CheckShapeAndScale(xShape, xScaleShape, weightShape, weightNzShape, weightScaleShape, groupListShape),
                OP_LOGE(context_->GetNodeName(), "shape and scale validation failed"), return false);
    OP_CHECK_IF(!CheckOutputShape(), OP_LOGE(context_->GetNodeName(), "output shape validation failed"), return false);
    return true;
}

bool GroupedMatmulSwigluQuantV3Tiling950::IsTensorApiCapable() const
{
    // Match V2 BasicApiTiling950::IsCapable after the V3-specific FP8/NZ,
    // single-tensor and MX scale shape checks above.
    if (aicoreParams_.l1Size == 0 || aicoreParams_.l0cSize == 0 || aivNum_ == 0) {
        return false;
    }
    const uint64_t averageM = inputParams_.mSize / inputParams_.groupNum;
    const bool capable = GroupedMatmulSwigluQuantTensorApiTiling::IsShapeAndPlatformCapable(
        {inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum, aicoreParams_.aicNum,
         aivNum_, transposeWeight_, true});
    OP_LOGD(context_->GetNodeName(),
            "V3 Tensor API capability: M=%lu, N=%lu, K=%lu, groups=%lu, averageM=%lu, "
            "transB=%d, cores=%lu:%lu, capable=%d",
            inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum, averageM,
            transposeWeight_, aicoreParams_.aicNum, aivNum_, capable);
    return capable;
}

bool GroupedMatmulSwigluQuantV3Tiling950::CheckCoreNum() const
{
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV3Tiling950::DoOpTiling()
{
    // Platform resources are populated after AnalyzeInputs by the base tiling lifecycle.
    useTensorApi_ = IsTensorApiCapable();
    tilingData_.gmmSwigluQuantParams.set_groupNum(static_cast<uint32_t>(inputParams_.groupNum));
    tilingData_.gmmSwigluQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
    tilingData_.gmmSwigluQuantParams.set_quantDtype(static_cast<uint8_t>(inputParams_.outDataDtype));
    tilingData_.gmmSwigluQuantParams.set_isMxWeightNzMultiTensor(0);
    tilingData_.gmmSwigluQuantParams.set_dequantDtype(static_cast<uint8_t>(FLOAT_DEQUANT_DTYPE));
    tilingData_.gmmSwigluQuantParams.set_swigluMode(swigluMode_);
    tilingData_.gmmSwigluQuantParams.set_clampLimit(clampLimit_);
    tilingData_.gmmSwigluQuantParams.set_gluAlpha(gluAlpha_);
    tilingData_.gmmSwigluQuantParams.set_gluBias(gluBias_);
    tilingData_.gmmSwigluQuantParams.set_scaleAlg(scaleAlg_);
    tilingData_.gmmSwigluQuantParams.set_dstTypeMax(dstTypeMax_);
    tilingData_.gmmSwigluQuantParams.set_roundMode(roundMode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV3Tiling950::DoLibApiTiling()
{
    CalBasicBlock();
    if (useTensorApi_) {
        GroupedMatmulSwigluQuantTensorApiTiling::PrepareBasicBlockForL1(basicTiling_);
    } else {
        basicTiling_.baseM =
            GroupedMatmul::CeilAlign(std::min(basicTiling_.baseM, static_cast<uint64_t>(128)), GmmConstant::CUBE_BLOCK);
        basicTiling_.baseN =
            GroupedMatmul::CeilAlign(std::min(basicTiling_.baseN, static_cast<uint64_t>(256)), GmmConstant::CUBE_BLOCK);
    }
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed for V3 MXFP8 weight-NZ"), return ge::GRAPH_FAILED);
    if (useTensorApi_) {
        GroupedMatmulSwigluQuantTensorApiTiling::RestoreBasicBlockAfterL1(basicTiling_);
    }

    auto &mm = tilingData_.mmTilingData;
    mm.set_M(inputParams_.mSize);
    mm.set_N(inputParams_.nSize);
    mm.set_Ka(inputParams_.kSize);
    mm.set_Kb(inputParams_.kSize);
    mm.set_usedCoreNum(aicoreParams_.aicNum);
    mm.set_baseM(basicTiling_.baseM);
    mm.set_baseN(basicTiling_.baseN);
    mm.set_baseK(basicTiling_.baseK);
    // The NZ CGMCT kernel consumes the per-core M tile selected by the
    // basic tiler.  Keep this aligned with the V2 NZ path: singleCoreM is
    // the actual base-M tile, not the generic single-core partition value.
    mm.set_singleCoreM(basicTiling_.baseM);
    mm.set_singleCoreN(basicTiling_.singleCoreN);
    mm.set_singleCoreK(basicTiling_.singleCoreK);
    mm.set_depthA1(basicTiling_.depthA1);
    mm.set_depthB1(basicTiling_.depthB1);
    mm.set_stepM(basicTiling_.stepM);
    mm.set_stepN(basicTiling_.stepN);
    mm.set_stepKa(basicTiling_.stepKa);
    mm.set_stepKb(basicTiling_.stepKb);
    mm.set_isBias(0);
    mm.set_iterateOrder(basicTiling_.iterateOrder);
    mm.set_dbL0A(2);
    mm.set_dbL0B(2);
    mm.set_dbL0C(basicTiling_.dbL0c);
    // The factors are bounded by CalL1Tiling's L1 budget. Matmul's K loop
    // handles the residual scale groups, so a K tail must not enlarge these
    // factors after the L1 depth and buffer sizes have been fixed.
    mm.set_mxTypePara((SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                      (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA);
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV3Tiling950::GetTilingKey() const
{
    const uint64_t kernelType =
        useTensorApi_ ? GMM_SWIGLU_QUANT_V3_TENSOR_LEVEL_KERNEL_TYPE : GMM_SWIGLU_QUANT_V3_CGMCT_KERNEL_TYPE;
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(transposeWeight_), GMM_SWIGLU_QUANT_NO_TRANS, kernelType);
}

ge::graphStatus GroupedMatmulSwigluQuantV3Tiling950::GetWorkspaceSize()
{
    auto *workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    workspace[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV3Tiling950::PostTiling()
{
    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        rawTilingData == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "tilingData", "does not support nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetData() == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "tilingData.data",
                                                         "does not support nullptr"),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(aicoreParams_.aicNum);
    context_->SetScheduleMode(1);
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "V3 tiling data is not 8-byte aligned"), return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace GroupedMatmulSwigluQuantV3Tiling
} // namespace optiling
