/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_gelu_tanh_quant_tiling.h"

#include <algorithm>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/grouped_matmul_gelu_tanh_quant_tiling_key.h"

using namespace Ops::Transformer::OpTiling;
using namespace optiling::GmmConstant;

namespace optiling {
namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t X_SCALE_INDEX = 1;
constexpr uint32_t GROUP_LIST_INDEX = 2;
constexpr uint32_t WEIGHT_INDEX = 3;
constexpr uint32_t WEIGHT_SCALE_INDEX = 4;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;
constexpr uint32_t ATTR_INDEX_DEQUANT_MODE = 0;
constexpr uint32_t ATTR_INDEX_DEQUANT_DTYPE = 1;
constexpr uint32_t ATTR_INDEX_QUANT_MODE = 2;
constexpr uint32_t ATTR_INDEX_QUANT_DTYPE = 3;
constexpr uint32_t ATTR_INDEX_TRANS_W = 4;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 5;
constexpr int64_t MX_QUANT_MODE = 2;
constexpr uint32_t WEIGHT_NZ_DIM_NUM = 5;
constexpr uint32_t WEIGHT_NZ_LAST_DIM_FP8 = 32;
constexpr uint32_t WEIGHT_NZ_C0_DIM = 16;
constexpr uint32_t FP8_E4M3FN_VALUE = 36;
constexpr uint32_t FP8_E5M2_VALUE = 35;
} // namespace

void GroupedMatmulGeluTanhQuantTiling950::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

ge::graphStatus GroupedMatmulGeluTanhQuantTiling950::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulGeluTanhQuantTiling950::IsFp8(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

bool GroupedMatmulGeluTanhQuantTiling950::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.transA = false;
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "group_list_type must be 0 or 1, actual is %d.",
                        inputParams_.groupListType),
                return false);

    const int64_t *dequantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    const int64_t *dequantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
    const int64_t *quantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_IF(dequantModePtr == nullptr || quantModePtr == nullptr || dequantDtypePtr == nullptr ||
                    quantDtypePtr == nullptr,
                OP_LOGE(context_->GetNodeName(), "quant attrs must not be nullptr."), return false);
    OP_CHECK_IF(*dequantModePtr != MX_QUANT_MODE || *quantModePtr != MX_QUANT_MODE,
                OP_LOGE(context_->GetNodeName(), "dequant_mode and quant_mode must be 2 in MX mode."),
                return false);
    auto dequantDtype = static_cast<ge::DataType>(*dequantDtypePtr);
    auto quantDtype = static_cast<ge::DataType>(*quantDtypePtr);
    OP_CHECK_IF(dequantDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "dequant_dtype must be FLOAT in MX mode."), return false);
    OP_CHECK_IF(!IsFp8(quantDtype),
                OP_LOGE(context_->GetNodeName(), "quant_dtype must be FLOAT8_E4M3FN or FLOAT8_E5M2."),
                return false);
    inputParams_.groupType = GroupedMatmul::SPLIT_M;
    return true;
}

bool GroupedMatmulGeluTanhQuantTiling950::CheckDtype() const
{
    OP_CHECK_IF(!IsFp8(inputParams_.aDtype) || !IsFp8(inputParams_.bDtype),
                OP_LOGE(context_->GetNodeName(), "x and weight must be FLOAT8_E4M3FN or FLOAT8_E5M2."),
                return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 ||
                    inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context_->GetNodeName(), "x_scale and weight_scale must be FLOAT8_E8M0."),
                return false);
    OP_CHECK_IF(!IsFp8(inputParams_.outDataDtype),
                OP_LOGE(context_->GetNodeName(), "y must be FLOAT8_E4M3FN or FLOAT8_E5M2."), return false);
    OP_CHECK_IF(inputParams_.outScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context_->GetNodeName(), "y_scale must be FLOAT8_E8M0."), return false);
    return true;
}

bool GroupedMatmulGeluTanhQuantTiling950::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    auto xScaleDesc = context_->GetInputDesc(X_SCALE_INDEX);
    auto wScaleDesc = context_->GetInputDesc(WEIGHT_SCALE_INDEX);
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    auto yScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || wDesc == nullptr || xScaleDesc == nullptr || wScaleDesc == nullptr ||
                    yDesc == nullptr || yScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "input or output desc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = wDesc->GetDataType();
    inputParams_.scaleDtype = wScaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = xScaleDesc->GetDataType();
    inputParams_.outDataDtype = yDesc->GetDataType();
    inputParams_.outScaleDtype = yScaleDesc->GetDataType();
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    return CheckDtype();
}

bool GroupedMatmulGeluTanhQuantTiling950::CheckMxScaleShape(const gert::Shape &xScaleShape,
                                                            const gert::Shape &wScaleShape) const
{
    OP_CHECK_IF(xScaleShape.GetDimNum() != MXFP_PER_TOKEN_SCALE_DIM_NUM ||
                    wScaleShape.GetDimNum() != MXFP_TYPE_M_SCALE_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "x_scale must be 3D and weight_scale must be 4D."),
                return false);
    auto expectedKBlocks = static_cast<int64_t>(Ops::Base::CeilDiv(inputParams_.kSize, MXFP_BASEK_FACTOR));
    auto wScaleN = inputParams_.transB ? wScaleShape.GetDim(1) : wScaleShape.GetDim(2);
    auto wScaleK = inputParams_.transB ? wScaleShape.GetDim(2) : wScaleShape.GetDim(1);
    OP_CHECK_IF(xScaleShape.GetDim(0) != static_cast<int64_t>(inputParams_.mSize) ||
                    xScaleShape.GetDim(1) != expectedKBlocks ||
                    xScaleShape.GetDim(2) != static_cast<int64_t>(MXFP_MULTI_BASE_SIZE),
                OP_LOGE(context_->GetNodeName(), "x_scale shape must be (M, ceil(K/64), 2)."), return false);
    OP_CHECK_IF(wScaleShape.GetDim(0) != static_cast<int64_t>(inputParams_.groupNum) ||
                    wScaleN != static_cast<int64_t>(inputParams_.nSize) || wScaleK != expectedKBlocks ||
                    wScaleShape.GetDim(3) != static_cast<int64_t>(MXFP_MULTI_BASE_SIZE),
                OP_LOGE(context_->GetNodeName(),
                        "weight_scale shape must be (E, ceil(K/64), N, 2) or (E, N, ceil(K/64), 2)."),
                return false);
    return true;
}

bool GroupedMatmulGeluTanhQuantTiling950::CheckWeightNzShape(const gert::Shape &wStorageShape) const
{
    OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "weight must use FRACTAL_NZ format."), return false);
    OP_CHECK_IF(wStorageShape.GetDimNum() != WEIGHT_NZ_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "NZ weight storage shape must be 5D."), return false);
    OP_CHECK_IF(wStorageShape.GetDim(3) != WEIGHT_NZ_C0_DIM || wStorageShape.GetDim(4) != WEIGHT_NZ_LAST_DIM_FP8,
                OP_LOGE(context_->GetNodeName(), "FP8 NZ weight storage tail dims must be 16 and 32."),
                return false);
    return true;
}

bool GroupedMatmulGeluTanhQuantTiling950::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto xScaleStorageShape = context_->GetInputShape(X_SCALE_INDEX);
    auto wScaleStorageShape = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_IF(xStorageShape == nullptr || wStorageShape == nullptr || xScaleStorageShape == nullptr ||
                    wScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "input shape is nullptr."), return false);
    OP_CHECK_IF(!SetGroupNum(GROUP_LIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xStorageShape->GetOriginShape(), wStorageShape->GetOriginShape()),
                OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    OP_CHECK_IF(!CheckMxScaleShape(xScaleStorageShape->GetOriginShape(), wScaleStorageShape->GetOriginShape()),
                OP_LOGE(inputParams_.opName, "CheckMxScaleShape failed."), return false);
    OP_CHECK_IF(!CheckWeightNzShape(wStorageShape->GetStorageShape()),
                OP_LOGE(inputParams_.opName, "CheckWeightNzShape failed."), return false);
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool GroupedMatmulGeluTanhQuantTiling950::CheckCoreNum() const
{
    auto compileInfo = context_->GetCompileInfo<GMMCompileInfo>();
    if (compileInfo == nullptr) {
        return true;
    }
    OP_CHECK_IF(compileInfo->aivNum != GmmConstant::CORE_RATIO * compileInfo->aicNum,
                OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual %u:%u.",
                        compileInfo->aicNum, compileInfo->aivNum),
                return false);
    return true;
}

ge::graphStatus GroupedMatmulGeluTanhQuantTiling950::DoOpTiling()
{
    tilingData_.gmmGeluTanhQuantParams.set_groupNum(static_cast<uint32_t>(inputParams_.groupNum));
    tilingData_.gmmGeluTanhQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
    tilingData_.gmmGeluTanhQuantParams.set_quantDtype(
        inputParams_.outDataDtype == ge::DT_FLOAT8_E5M2 ? FP8_E5M2_VALUE : FP8_E4M3FN_VALUE);
    tilingData_.gmmGeluTanhQuantParams.set_dequantDtype(static_cast<uint8_t>(ge::DT_FLOAT));
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulGeluTanhQuantTiling950::FillMmTilingData()
{
    tilingData_.mmTilingData.set_M(inputParams_.mSize);
    tilingData_.mmTilingData.set_N(inputParams_.nSize);
    tilingData_.mmTilingData.set_Ka(inputParams_.kSize);
    tilingData_.mmTilingData.set_Kb(inputParams_.kSize);
    tilingData_.mmTilingData.set_usedCoreNum(aicoreParams_.aicNum);
    tilingData_.mmTilingData.set_baseM(basicTiling_.baseM);
    tilingData_.mmTilingData.set_baseN(basicTiling_.baseN);
    tilingData_.mmTilingData.set_baseK(basicTiling_.baseK);
    tilingData_.mmTilingData.set_singleCoreM(basicTiling_.baseM);
    tilingData_.mmTilingData.set_singleCoreN(basicTiling_.singleCoreN);
    tilingData_.mmTilingData.set_singleCoreK(basicTiling_.singleCoreK);
    tilingData_.mmTilingData.set_depthA1(basicTiling_.depthA1);
    tilingData_.mmTilingData.set_depthB1(basicTiling_.depthB1);
    tilingData_.mmTilingData.set_stepM(basicTiling_.stepM);
    tilingData_.mmTilingData.set_stepN(basicTiling_.stepN);
    tilingData_.mmTilingData.set_stepKa(basicTiling_.stepKa);
    tilingData_.mmTilingData.set_stepKb(basicTiling_.stepKb);
    tilingData_.mmTilingData.set_isBias(0);
    tilingData_.mmTilingData.set_iterateOrder(basicTiling_.iterateOrder);
    tilingData_.mmTilingData.set_dbL0A(2);
    tilingData_.mmTilingData.set_dbL0B(2);
    tilingData_.mmTilingData.set_dbL0C(basicTiling_.dbL0c);
    if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
        basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
        tilingData_.mmTilingData.set_mxTypePara(
            (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
            (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA);
    } else {
        tilingData_.mmTilingData.set_mxTypePara(
            (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
            (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT);
    }
}

ge::graphStatus GroupedMatmulGeluTanhQuantTiling950::DoLibApiTiling()
{
    CalBasicBlock();
    auto baseMModified = std::min(basicTiling_.baseM, static_cast<uint64_t>(128));
    basicTiling_.baseM = GroupedMatmul::CeilAlign(baseMModified, GmmConstant::CUBE_BLOCK);
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."),
                return ge::GRAPH_FAILED);
    FillMmTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulGeluTanhQuantTiling950::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB),
                              static_cast<uint64_t>(inputParams_.transA));
}

ge::graphStatus GroupedMatmulGeluTanhQuantTiling950::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulGeluTanhQuantTiling950::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

void GroupedMatmulGeluTanhQuantTiling950::PrintQuantParams()
{
    OP_LOGD(inputParams_.opName, "GMMGeluTanhQuantParams: groupNum=%lu, groupListType=%d, M=%lu, N=%lu, K=%lu.",
            inputParams_.groupNum, inputParams_.groupListType, inputParams_.mSize, inputParams_.nSize,
            inputParams_.kSize);
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulGeluTanhQuant, GroupedMatmulGeluTanhQuantTiling950, 0);

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGroupedMatmulGeluTanhQuant(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupedMatmulGeluTanhQuantTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedMatmulGeluTanhQuant", "tiling context is nullptr."),
                return ge::GRAPH_FAILED);
    std::vector<int32_t> registerList = {0};
    return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

IMPL_OP_OPTILING(GroupedMatmulGeluTanhQuant)
    .Tiling(GroupedMatmulGeluTanhQuantTilingFunc)
    .TilingParse<GMMCompileInfo>(TilingPrepareForGroupedMatmulGeluTanhQuant);
} // namespace optiling
