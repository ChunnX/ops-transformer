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
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GMMSQArch35WeightQuantTiling;

namespace optiling {

constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t MX_SCALE_BLOCK_SIZE = 64;
constexpr uint32_t ALIGN_SIZE_KN = 32;
constexpr int64_t MIN_KN_SIZE = 32;
constexpr uint32_t DIM_NUM_WEIGHT_NZ = 5;

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetPlatformInfo()
{
    auto compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("GroupedMatmulSwigluQuantV2", "CompileInfo is null"),
                return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;

    OP_LOGI(context_, "Compile info: aicNum(%lu) ubSize(%lu) l1Size(%lu) l0aSize(%lu) l0bSize(%lu) l0cSize(%lu).",
            compileInfoPtr_->aicNum, compileInfoPtr_->ubSize, compileInfoPtr_->l1Size, compileInfoPtr_->l0ASize,
            compileInfoPtr_->l0BSize, compileInfoPtr_->l0CSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OP_LOGD(context_->GetNodeName(), "Start inferring scenario and checking params.");

    OP_CHECK_IF(!InferScenario(), OP_LOGE(inputParams_.opName, "Failed to infer scenario."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Scenario inferred successfully, type=%d.",
            static_cast<int>(scenarioType_));

    OP_CHECK_IF(!RunCheckFunc(), OP_LOGE(inputParams_.opName, "Failed to check input params."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "All input params validated.");

    RunSetInputFunc();
    OP_LOGD(context_->GetNodeName(),
            "Input params set: mSize=%ld, kSize=%ld, nSize=%ld, "
            "groupNum=%ld, outputBS=%ld, hasBias=%d.",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, inputParams_.groupNum, inputParams_.outputBS,
            inputParams_.hasBias);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start setting tiling data.");

    tilingData_.groupListType = inputParams_.groupListType;
    tilingData_.hasBias = inputParams_.hasBias;
    tilingData_.coreNum = compileInfoPtr_->aicNum;
    tilingData_.groupNum = inputParams_.groupNum;
    tilingData_.outputBs = inputParams_.outputBS;
    tilingData_.groupSize = MX_BLOCK_SIZE;
    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;

    OP_LOGD(context_->GetNodeName(),
            "Tiling data set: coreNum=%u, groupNum=%ld, outputBs=%ld, "
            "kSize=%ld, nSize=%ld, hasBias=%d, groupListType=%d.",
            tilingData_.coreNum, tilingData_.groupNum, tilingData_.outputBs, tilingData_.kSize, tilingData_.nSize,
            tilingData_.hasBias, tilingData_.groupListType);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2WeightQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(0, static_cast<bool>(inputParams_.wTrans), static_cast<bool>(inputParams_.hasBias));
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is nullptr."),
                return ge::GRAPH_FAILED);
    workspaces[0] = 16777216U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start posting tiling data, blockDim=%u, tilingSize=%zu.",
            tilingData_.coreNum, sizeof(tilingData_));

    context_->SetBlockDim(tilingData_.coreNum);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "RawTilingData is nullptr."),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
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
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::InferScenario()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input xDesc is nullptr."), return false);
    auto wDesc = context_->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input wDesc is nullptr."), return false);

    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto scaleDesc = context_->GetOptionalInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input scaleDesc is nullptr."), return false);
    auto scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    auto perTokenScaleDtype = pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : ge::DT_INT8;
    auto xDtype = xDesc->GetDataType();
    auto wDtype = wDesc->GetDataType();

    OP_LOGD(context_->GetNodeName(), "Current xDtype: %s, wDtype: %s, scaleDtype: %s, perTokenScaleDtype: %s, wFormat: %s",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(wDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(perTokenScaleDtype).c_str(),
            (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) ? "FRACTAL_NZ" : "ND");

    if (xDtype == ge::DT_FLOAT8_E4M3FN && wDtype == ge::DT_FLOAT4_E2M1 &&
        (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) &&
        scaleDtype == ge::DT_FLOAT8_E8M0 && perTokenScaleDtype == ge::DT_FLOAT8_E8M0) {
        scenarioType_ = ScenarioType::MX_A8W4_WEIGHT_NZ;
        OP_LOGD(context_->GetNodeName(), "Enable MX-A8W4-WEIGHT-NZ mode.");
        return true;
    }

    OP_LOGD(context_->GetNodeName(), "No matching scenario found.");
    return false;
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::AnalyzeAttrs()
{
    const bool *xTrans = context_->GetAttrPointer<bool>(X_TRANS_INDEX);
    const bool *wTrans = context_->GetAttrPointer<bool>(W_TRANS_INDEX);
    const int64_t *groupListType = context_->GetAttrPointer<int64_t>(GROUP_LIST_TYPE_INDEX);

    OP_CHECK_IF(xTrans == nullptr, OP_LOGE(inputParams_.opName, "Attr xTrans is nullptr."), return false);
    OP_CHECK_IF(wTrans == nullptr, OP_LOGE(inputParams_.opName, "Attr wTrans is nullptr."), return false);
    OP_CHECK_IF(groupListType == nullptr, OP_LOGE(inputParams_.opName, "Attr groupListType is nullptr."), return false);

    inputParams_.xTrans = *xTrans;
    inputParams_.wTrans = *wTrans;
    inputParams_.groupListType = *groupListType;

    OP_CHECK_IF(inputParams_.xTrans != false,
                OP_LOGE(inputParams_.opName, "MxA8W4 only supports xTrans=false."), return false);
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(inputParams_.opName, "groupListType must be 0 or 1."), return false);

    return true;
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto wDesc = context_->GetInputDesc(W_INDEX);
    inputParams_.xDtype = xDesc->GetDataType();
    inputParams_.wDtype = wDesc->GetDataType();
    return true;
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::AnalyzeInputs()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto wDesc = context_->GetInputDesc(W_INDEX);
    auto groupListDesc = context_->GetInputDesc(GROUPLIST_INDEX);

    OP_CHECK_IF(xDesc == nullptr || wDesc == nullptr || groupListDesc == nullptr,
                OP_LOGE(inputParams_.opName, "Required input is nullptr."), return false);

    auto xShape = xDesc->GetShape();
    auto wShape = wDesc->GetShape();
    auto groupListShape = groupListDesc->GetShape();

    inputParams_.mSize = xShape.GetDim(0);
    inputParams_.kSize = xShape.GetDim(1);
    inputParams_.groupNum = groupListShape.GetDim(0);

    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    inputParams_.wFormat = wFormat;

    if (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        inputParams_.nSize = wShape.GetDim(1) * 16;
    } else {
        inputParams_.nSize = wShape.GetDim(1);
    }

    inputParams_.outputBS = inputParams_.mSize;
    inputParams_.hasBias = false;

    OP_CHECK_IF(inputParams_.kSize < MIN_KN_SIZE,
                OP_LOGE(inputParams_.opName, "K size %ld < min %d.", inputParams_.kSize, MIN_KN_SIZE), return false);
    OP_CHECK_IF(inputParams_.nSize < MIN_KN_SIZE,
                OP_LOGE(inputParams_.opName, "N size %ld < min %d.", inputParams_.nSize, MIN_KN_SIZE), return false);
    OP_CHECK_IF(inputParams_.kSize % ALIGN_SIZE_KN != 0,
                OP_LOGE(inputParams_.opName, "K size %ld not %d aligned.", inputParams_.kSize, ALIGN_SIZE_KN), return false);
    OP_CHECK_IF(inputParams_.nSize % 128 != 0,
                OP_LOGE(inputParams_.opName, "N size %ld not 128 aligned.", inputParams_.nSize), return false);

    return true;
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::CheckShapeForMxQuant(const gert::Shape &wScaleShape, const gert::Shape &xScaleShape)
{
    if (wScaleShape.GetDimNum() != 4) {
        OP_LOGE(inputParams_.opName, "wScale shape dim num %lu != 4.", wScaleShape.GetDimNum());
        return false;
    }
    if (xScaleShape.GetDimNum() != 3) {
        OP_LOGE(inputParams_.opName, "xScale shape dim num %lu != 3.", xScaleShape.GetDimNum());
        return false;
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::RunCheckFunc()
{
    for (auto &checkFunc : checkConditionFuncs_) {
        OP_CHECK_IF(!checkFunc(context_), OP_LOGE(inputParams_.opName, "Check condition failed."), return false);
    }
    return true;
}

void GroupedMatmulSwigluQuantV2WeightQuantTiling::RunSetInputFunc()
{
    for (auto &setFunc : SetInputFuncs_) {
        setFunc(context_, inputParams_);
    }
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::SetMxA8W4NzInputFunc()
{
    checkConditionFuncs_.push_back([this](gert::TilingContext *context) { return AnalyzeAttrs(); });
    checkConditionFuncs_.push_back([this](gert::TilingContext *context) { return AnalyzeDtype(); });
    checkConditionFuncs_.push_back([this](gert::TilingContext *context) { return AnalyzeInputs(); });
    return true;
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::SetMxA8W4NzConditionFunc()
{
    return true;
}
} // namespace optiling
