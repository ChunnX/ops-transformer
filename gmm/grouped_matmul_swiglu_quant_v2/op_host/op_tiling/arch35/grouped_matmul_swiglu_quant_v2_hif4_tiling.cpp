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
 * \file grouped_matmul_swiglu_quant_v2_basic_api_tiling.cpp
 * \brief
 */

#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "grouped_matmul_swiglu_quant_v2_hif4_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmulSwigluQuantParamsV2;
using namespace optiling::GmmConstant;
namespace optiling {
void GroupedMatmulSwigluQuantV2Hif4Tiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    return;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Hif4Tiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmBasicApiTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::IsCapable()
{
    // hif4
    return IsHif4Scale();
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::IsHifp4(ge::DataType dtype) const
{
    return dtype == ge::DT_HIFLOAT4;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::IsHifp4Input() const
{
    return IsHifp4(inputParams_.aDtype) && IsHifp4(inputParams_.bDtype);
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::IsHif4Scale() const
{
    return IsHifp4Input() && inputParams_.scaleDtype == ge::DT_HIFLOAT4_SCALE;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::CheckQuantParamsForHif4TypeM(
    const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const
{
    if (xScaleShape.GetDimNum() != 3 || wScaleShape.GetDimNum() != 4) {
        OP_LOGE(inputParams_.opName, "HIF4 requires xScale rank 3 and weightScale rank 4.");
        return false;
    }
    auto scaleK = static_cast<int64_t>(GroupedMatmul::CeilDiv(inputParams_.kSize, GmmConstant::MXFP_BASEK_FACTOR));
    bool validX = xScaleShape.GetDim(0) == static_cast<int64_t>(inputParams_.mSize) &&
                  xScaleShape.GetDim(1) == scaleK && xScaleShape.GetDim(2) == 1;
    bool validW = wScaleShape.GetDim(0) == static_cast<int64_t>(inputParams_.groupNum) &&
                  wScaleShape.GetDim(inputParams_.transB ? 1 : 2) == static_cast<int64_t>(inputParams_.nSize) &&
                  wScaleShape.GetDim(inputParams_.transB ? 2 : 1) == scaleK && wScaleShape.GetDim(3) == 1;
    if (!validX || !validW) {
        OP_LOGE(inputParams_.opName, "HIF4 xScale or weightScale shape does not match M, N, K and groupNum.");
        return false;
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
        OP_CHECK_IF(!(inputParams_.groupListType == 0 || inputParams_.groupListType == 1),
                    OP_LOGE(context_->GetNodeName(), "GroupListType must be 0 or 1, but actual value is %d.",
                            inputParams_.groupListType), return false);
    }
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    const int64_t *dequantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    OP_CHECK_IF(dequantModePtr == nullptr, OP_LOGE(context_->GetNodeName(), "The dequantModePtr is nullptr."),
                return false);
    OP_CHECK_IF(*dequantModePtr != Hif4QuantMode,
                OP_LOGE(context_->GetNodeName(), "In hifloat4 quant mode, dequantMode should be 3, but actual value is %ld.",
                        *dequantModePtr), return false);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    OP_CHECK_IF(quantModePtr == nullptr, OP_LOGE(context_->GetNodeName(), "The quantModePtr is nullptr."),
                return false);
    OP_CHECK_IF(*quantModePtr != Hif4QuantMode,
                OP_LOGE(context_->GetNodeName(), "In hifloat4 quant mode, quantMode should be 3, but actual value is %ld.",
                        *quantModePtr), return false);
    const int64_t *dequantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
    OP_CHECK_IF(dequantDtypePtr == nullptr, OP_LOGE(context_->GetNodeName(), "The dequantDtypePtr is nullptr."),
                return false);
    ge::DataType dequantDtype = static_cast<ge::DataType>(*dequantDtypePtr);
    OP_CHECK_IF(dequantDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(),
                        "In hifloat4 quant mode, dequantDtype should be DT_FLOAT, but"
                        " actual value is %s.",
                        ge::TypeUtils::DataTypeToSerialString(dequantDtype).c_str()), return false);
    const int64_t *quantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_IF(quantDtypePtr == nullptr, OP_LOGE(context_->GetNodeName(), "The quantDtypePtr is nullptr."),
                return false);
    ge::DataType quantDtype = static_cast<ge::DataType>(*quantDtypePtr);
    OP_CHECK_IF(quantDtype != ge::DT_HIFLOAT4,
                OP_LOGE(context_->GetNodeName(),
                        "In hifloat4 quant mode, quantDtype should be DT_HIFLOAT4, but"
                        " actual value is %s.",
                        ge::TypeUtils::DataTypeToSerialString(dequantDtype).c_str()), return false);
    // gmm quant tiling need groupType to calculate L1 tiling 
  	inputParams_.groupType = SPLIT_M;
    return true;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scaleDesc is nullptr."), return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto outDesc = context_->GetOutputDesc(Y_DATA_INDEX);
    OP_CHECK_IF(outDesc == nullptr, OP_LOGE(context_->GetNodeName(), "OutDesc is nullptr."), return false);
    inputParams_.outDataDtype = outDesc->GetDataType();
    auto outScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(outScaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "OutScaleDesc is nullptr."), return false);
    inputParams_.outScaleDtype = outScaleDesc->GetDataType();
    auto x1ScaleStorageShape = context_->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "XScaleStorageShape is nullptr."),
                return false);
    const gert::Shape &xScaleShape = x1ScaleStorageShape->GetOriginShape();
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetStorageShape();
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!SetQuantModeForGMMSwigluQuant(wScaleShape, xScaleShape),
                OP_LOGE(inputParams_.opName, "SetQuantModeForGMMSwigluQuant failed."), return false);
    return CheckDtype();
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::CheckDtype() const
{
    OP_CHECK_IF(!(IsHifp4(inputParams_.aDtype) && IsHifp4(inputParams_.bDtype)),
                OP_LOGE(inputParams_.opName,
                        "The dtype of x and weight should both be HIFLOAT4, but x dtype is %s, weight dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                return false);
    OP_CHECK_IF(!IsHifp4Input(),
                OP_LOGE(inputParams_.opName,
                        "Only HIFLOAT4 inputs are supported, but x dtype is %s, weight dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_HIFLOAT4_SCALE || inputParams_.perTokenScaleDtype != ge::DT_HIFLOAT4_SCALE,
                OP_LOGE(inputParams_.opName, "Xscale dtype must be HIFLOAT4_SCALE, but actual dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                return false);
    OP_CHECK_IF(!IsHifp4(inputParams_.outDataDtype),
                OP_LOGE(inputParams_.opName, "Only HIFLOAT4 output are supported, but out dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype).c_str()),
                return false);
    OP_CHECK_IF(inputParams_.outScaleDtype != ge::DT_HIFLOAT4_SCALE,
                OP_LOGE(inputParams_.opName, "OutScale dtype must be HIFLOAT4_SCALE, but actual dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.outScaleDtype).c_str()),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::AnalyzeInputs()
{
    OP_CHECK_IF(!CheckCoreNum(),
            OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        scaleDimNum != MX_WEIGHT_SCALE_DIM,
        OP_LOGE(inputParams_.opName, "The dimension of weight_scale should be equal to 4, actual is %zu", scaleDimNum),
        return false);
    auto x1ScaleStorageShape = context_->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "XScaleStorageShape is nullptr."),
                return false);
    const gert::Shape &xScaleShape = x1ScaleStorageShape->GetOriginShape();
    auto xScaleDimNum = xScaleShape.GetDimNum();
    OP_CHECK_IF(
        xScaleDimNum != MX_X_SCALE_DIM,
        OP_LOGE(inputParams_.opName, "The dimension of x_scale should be equal to 3, actual is %zu", xScaleDimNum),
        return false);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!CheckQuantParamsForHif4TypeM(xScaleShape, wScaleShape),
                OP_LOGE(inputParams_.opName, "CheckQuantParamsForHif4TypeM failed."), return false);
    return CheckHif4Dims();
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::CheckHif4Dims() const
{
    auto aInnerSize = inputParams_.transA ? inputParams_.mSize : inputParams_.kSize;
    auto bInnerSize = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;
    OP_CHECK_IF(
        IsHifp4Input() && (aInnerSize % B4_DATACOPY_MIN_NUM != 0 || bInnerSize % B4_DATACOPY_MIN_NUM != 0),
        OP_LOGE(inputParams_.opName, "When inputs are HIFLOAT4, x and weight inner axis element number should be even."),
        return false);

    // HIFLOAT4 场景不支持K=2
    OP_CHECK_IF(IsHifp4Input() && inputParams_.kSize == HIF4_K_MIN_VALUE,
                OP_LOGE(inputParams_.opName,
                        "When the dtypes of x and weight are DT_HIFLOAT4,"
                        " the K value should be greater than 2, but actual value is %lu.",
                        inputParams_.kSize),
                return false);
    // HIFLOAT4 场景下，当输出类型为HIF4时，N需要满足为大于等于4的偶数
    if (IsHifp4Input()) {
        OP_CHECK_IF(inputParams_.nSize < HIF4_N_MIN_VALUE || inputParams_.nSize % EVEN_FACTOR != 0,
                    OP_LOGE(inputParams_.opName,
                            "When inputs and output are HIFLOAT4, N value should be even and greater or equal to 4, "
                            "but actual N is %lu.",
                            inputParams_.nSize),
                    return false);
    }
    // HIFLOAT4 量化场景下，N为128对齐
    OP_CHECK_IF(inputParams_.nSize % GmmConstant::BASIC_BLOCK_SIZE_128 != 0,
                OP_LOGE(inputParams_.opName, "Weight n axis element number should be an integer multiple of 128."),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::CheckCoreNum() const
{
    auto aicNum = context_->GetCompileInfo<GMMSwigluV2CompileInfo>()->aicNum_;
    auto aivNum = context_->GetCompileInfo<GMMSwigluV2CompileInfo>()->aivNum_;
    OP_CHECK_IF(aicNum == 0,
                OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    uint32_t coreRatio = context_->GetCompileInfo<GMMSwigluV2CompileInfo>()->supportMmadF16S8 ?
                             1U : GmmConstant::CORE_RATIO;
    OP_CHECK_IF(aivNum != coreRatio * aicNum,
                OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:%u, actual aicNum: %u, aivNum: %u.", coreRatio,
                        aicNum, aivNum),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Hif4Tiling::SetQuantModeForGMMSwigluQuant(const gert::Shape &wScaleShape,
                                                                          const gert::Shape &xScaleShape)
{
    auto wScaleDims = wScaleShape.GetDimNum();
    if (IsHif4Scale()) {
        inputParams_.bQuantMode = optiling::QuantMode::HIF4_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::HIF4_PERGROUP_MODE;
        return true;
    }
    return false;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Hif4Tiling::DoOpTiling()
{
    tilingData_.gmmSwigluQuantParams.set_groupNum(inputParams_.groupNum);
    tilingData_.gmmSwigluQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *dequantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
        int64_t dequantDtype = dequantDtypeTypePtr != nullptr ? static_cast<int64_t>(*dequantDtypeTypePtr) : 0L;
        const int64_t *quantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
        int64_t quantDtype = quantDtypeTypePtr != nullptr ? static_cast<int64_t>(*quantDtypeTypePtr) : 0L;
        tilingData_.gmmSwigluQuantParams.set_quantDtype(static_cast<uint8_t>(quantDtype));
    }
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Hif4Tiling::DoLibApiTiling()
{
    // Reuse GMM common tiling helpers and keep the output aligned with GroupedQmmBasicApiTiling::DoLibApiTiling.
    // The kernel consumes compact low-level BasicAPI fields instead of TCubeTiling, so scaleKAL1 and scaleKBL1
    GroupedQmmTiling::CalBasicBlock();
    auto baseMModified = std::min(basicTiling_.baseM, GmmConstant::MAX_BASEM_ONE_VECTOR);
    basicTiling_.baseM = GroupedMatmul::CeilAlign(baseMModified, GmmConstant::CUBE_BLOCK);
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
 	                 OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."), return ge::GRAPH_FAILED);
    tilingData_.basicTilingData.set_m(inputParams_.mSize);
    tilingData_.basicTilingData.set_n(inputParams_.nSize);
    tilingData_.basicTilingData.set_k(inputParams_.kSize);
    tilingData_.basicTilingData.set_baseM(basicTiling_.baseM);
    tilingData_.basicTilingData.set_baseN(basicTiling_.baseN);
    tilingData_.basicTilingData.set_baseK(basicTiling_.baseK);
    tilingData_.basicTilingData.set_kAL1(basicTiling_.stepKa * basicTiling_.baseK);
    tilingData_.basicTilingData.set_kBL1(basicTiling_.stepKb * basicTiling_.baseK);
    tilingData_.basicTilingData.set_dbL0C(basicTiling_.dbL0c);
    uint32_t scaleKL1 = std::min(
 	         std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa,
 	                  basicTiling_.scaleFactorB * basicTiling_.stepKb) *
 	             basicTiling_.baseK,
 	         inputParams_.kSize);
    tilingData_.basicTilingData.set_scaleKAL1(scaleKL1);
    tilingData_.basicTilingData.set_scaleKBL1(scaleKL1);
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2Hif4Tiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              static_cast<uint64_t>(GMM_SWIGLU_HIF4_QUANT_KERNEL_TYPE));
}

ge::graphStatus GroupedMatmulSwigluQuantV2Hif4Tiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZES;
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulSwigluQuantV2Hif4Tiling::LogQuantParams(const GMMSwigluQuantParams &params) const
{
    int32_t enable = CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }

    std::ostringstream oss;
    oss << "GMMSwigluQuantParams: groupNum = " << inputParams_.groupNum 
        << ", groupType = " << static_cast<int32_t>(inputParams_.groupType)
        << ", groupListType = " << static_cast<uint32_t>(inputParams_.groupListType);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
}

ge::graphStatus GroupedMatmulSwigluQuantV2Hif4Tiling::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

void GroupedMatmulSwigluQuantV2Hif4Tiling::PrintQuantParams()
{
    LogQuantParams(tilingData_.gmmSwigluQuantParams);
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2Hif4Tiling, 10);
} // namespace optiling
