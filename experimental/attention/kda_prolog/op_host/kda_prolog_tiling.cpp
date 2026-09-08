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
 * \file kda_prolog_tiling.cpp
 * \brief Orchestrates per-module tiling aligned with kernel components.
 */

#include "kda_prolog_tiling.h"

#include "../op_kernel/kda_prolog_template_tiling_key.h"
#include "../op_kernel/kda_prolog_workspace.h"
#include "kda_prolog_tiling_mm_bgg.h"
#include "kda_prolog_tiling_mx_quant.h"
#include "kda_prolog_tiling_qmm_qkv.h"
#include "kda_prolog_tiling_sigmoid.h"

namespace optiling {
namespace {
constexpr uint32_t DIM_NUM_TWO = 2U;
constexpr uint32_t DIM_NUM_THREE = 3U;
constexpr int64_t MX_BLOCK_SIZE = 64L;
constexpr int64_t SCALE_QKV_PACK_NUM = 2L;

ge::graphStatus InferOutFeatures(const char *opName, const gert::Shape &weightStorage, int64_t hiddenSize,
                                 bool transWeight, const char *weightName, int64_t &outFeatures)
{
    const int64_t dim0 = weightStorage.GetDim(0);
    const int64_t dim1 = weightStorage.GetDim(1);
    OP_CHECK_IF(dim0 <= 0L || dim1 <= 0L, OP_LOGE(opName, "%s shape [%ld, %ld] is invalid.", weightName, dim0, dim1),
                return ge::GRAPH_FAILED);
    const int64_t inFeatures = transWeight ? dim1 : dim0;
    OP_CHECK_IF(inFeatures != hiddenSize,
                OP_LOGE(opName, "%s inFeatures=%ld does not match x.dim1=%ld (trans_weight=%d).", weightName,
                        inFeatures, hiddenSize, static_cast<int32_t>(transWeight)),
                return ge::GRAPH_FAILED);
    outFeatures = transWeight ? dim0 : dim1;
    OP_CHECK_IF(outFeatures <= 0L, OP_LOGE(opName, "%s outFeatures=%ld is invalid.", weightName, outFeatures),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckExpectDataType(ge::DataType actual, ge::DataType expect, const char *tensorName,
                                    const char *expectName, const char *opName)
{
    OP_CHECK_IF(actual != expect, OP_LOGE(opName, "%s dtype must be %s.", tensorName, expectName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus KdaPrologInfoParser::GetOpName()
{
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaProlog", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    const char *opName = context_->GetNodeName();
    OP_CHECK_IF(opName == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaProlog", "Node name is null."),
                return ge::GRAPH_FAILED);
    opName_ = opName;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0U || aivNum == 0U, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);
    aicNum_ = aicNum;

    const auto socVersion = ascendcPlatform.GetSocVersion();
    OP_CHECK_IF(socVersion != platform_ascendc::SocVersion::ASCEND950,
                OP_LOGE(opName_, "SOC Version[%d] is not supported, only ASCEND950 is supported.",
                        static_cast<int32_t>(socVersion)),
                return ge::GRAPH_FAILED);

    l1Size_ = 0UL;
    l0cSize_ = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    OP_CHECK_IF(l1Size_ == 0UL || l0cSize_ == 0UL,
                OP_LOGE(opName_, "l1Size or l0cSize is 0, platform mem info is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void KdaPrologInfoParser::GetInputParaInfo()
{
    opParamInfo_.x.desc = context_->GetInputDesc(X_INDEX);
    opParamInfo_.x.shape = context_->GetInputShape(X_INDEX);
    opParamInfo_.weightQkv.desc = context_->GetInputDesc(WEIGHT_QKV_INDEX);
    opParamInfo_.weightQkv.shape = context_->GetInputShape(WEIGHT_QKV_INDEX);
    opParamInfo_.weightBeta.desc = context_->GetInputDesc(WEIGHT_BETA_INDEX);
    opParamInfo_.weightBeta.shape = context_->GetInputShape(WEIGHT_BETA_INDEX);
    opParamInfo_.weightGate.desc = context_->GetInputDesc(WEIGHT_GATE_INDEX);
    opParamInfo_.weightGate.shape = context_->GetInputShape(WEIGHT_GATE_INDEX);
    opParamInfo_.weightG.desc = context_->GetInputDesc(WEIGHT_G_INDEX);
    opParamInfo_.weightG.shape = context_->GetInputShape(WEIGHT_G_INDEX);
    opParamInfo_.scaleQkv.desc = context_->GetInputDesc(SCALE_QKV_INDEX);
    opParamInfo_.scaleQkv.shape = context_->GetInputShape(SCALE_QKV_INDEX);
}

void KdaPrologInfoParser::GetOutputParaInfo()
{
    opParamInfo_.qkvOutDesc = context_->GetOutputDesc(QKV_INDEX);
    opParamInfo_.betaOutDesc = context_->GetOutputDesc(BETA_INDEX);
    opParamInfo_.gateOutDesc = context_->GetOutputDesc(GATE_INDEX);
    opParamInfo_.gOutDesc = context_->GetOutputDesc(G_INDEX);
}

ge::graphStatus KdaPrologInfoParser::GetAndCheckAttrParaInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        opParamInfo_.transWeightQkv = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_QKV_INDEX);
        opParamInfo_.transWeightBeta = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_BETA_INDEX);
        opParamInfo_.transWeightGate = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_GATE_INDEX);
        opParamInfo_.transWeightG = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_G_INDEX);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.x.shape == nullptr, OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_prolog::X_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.x.desc == nullptr, OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::X_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkv.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_prolog::WEIGHT_QKV_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkv.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::WEIGHT_QKV_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightBeta.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_prolog::WEIGHT_BETA_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightBeta.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::WEIGHT_BETA_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightGate.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_prolog::WEIGHT_GATE_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightGate.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::WEIGHT_GATE_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightG.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_prolog::WEIGHT_G_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightG.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::WEIGHT_G_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleQkv.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_prolog::SCALE_QKV_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleQkv.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::SCALE_QKV_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qkvOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::QKV_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.betaOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::BETA_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.gateOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::GATE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.gOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_prolog::G_NAME), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::GetAndCheckInOutDataType()
{
    const ge::DataType xType = opParamInfo_.x.desc->GetDataType();
    const ge::DataType weightQkvType = opParamInfo_.weightQkv.desc->GetDataType();
    const ge::DataType weightBetaType = opParamInfo_.weightBeta.desc->GetDataType();
    const ge::DataType weightGateType = opParamInfo_.weightGate.desc->GetDataType();
    const ge::DataType weightGType = opParamInfo_.weightG.desc->GetDataType();
    const ge::DataType scaleQkvType = opParamInfo_.scaleQkv.desc->GetDataType();
    const ge::DataType qkvOutType = opParamInfo_.qkvOutDesc->GetDataType();
    const ge::DataType betaOutType = opParamInfo_.betaOutDesc->GetDataType();
    const ge::DataType gateOutType = opParamInfo_.gateOutDesc->GetDataType();
    const ge::DataType gOutType = opParamInfo_.gOutDesc->GetDataType();

    if (CheckExpectDataType(xType, ge::DT_BF16, kda_prolog::X_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightQkvType, ge::DT_FLOAT8_E4M3FN, kda_prolog::WEIGHT_QKV_NAME, "FLOAT8_E4M3FN",
                            opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightBetaType, ge::DT_BF16, kda_prolog::WEIGHT_BETA_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightGateType, ge::DT_BF16, kda_prolog::WEIGHT_GATE_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightGType, ge::DT_BF16, kda_prolog::WEIGHT_G_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(scaleQkvType, ge::DT_FLOAT8_E8M0, kda_prolog::SCALE_QKV_NAME, "FLOAT8_E8M0", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(qkvOutType, ge::DT_BF16, kda_prolog::QKV_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(betaOutType, ge::DT_FLOAT, kda_prolog::BETA_NAME, "FLOAT", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(gateOutType, ge::DT_BF16, kda_prolog::GATE_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(gOutType, ge::DT_BF16, kda_prolog::G_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::CheckShapeDim()
{
    OP_CHECK_IF(opParamInfo_.x.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_prolog::X_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkv.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_prolog::WEIGHT_QKV_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightBeta.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_prolog::WEIGHT_BETA_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightGate.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_prolog::WEIGHT_GATE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightG.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_prolog::WEIGHT_G_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleQkv.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(opName_, "%s must be 3D.", kda_prolog::SCALE_QKV_NAME), return ge::GRAPH_FAILED);

    const int64_t tSize = opParamInfo_.x.shape->GetStorageShape().GetDim(0);
    const int64_t hiddenSize = opParamInfo_.x.shape->GetStorageShape().GetDim(1);
    OP_CHECK_IF(tSize <= 0L || hiddenSize <= 0L,
                OP_LOGE(opName_, "%s shape [%ld, %ld] is invalid.", kda_prolog::X_NAME, tSize, hiddenSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::GetBaseShapeInfo()
{
    const int64_t tSize = opParamInfo_.x.shape->GetStorageShape().GetDim(0);
    const int64_t hiddenSize = opParamInfo_.x.shape->GetStorageShape().GetDim(1);
    const bool transWeightQkv = opParamInfo_.transWeightQkv != nullptr ? *opParamInfo_.transWeightQkv : true;
    const bool transWeightBeta = opParamInfo_.transWeightBeta != nullptr ? *opParamInfo_.transWeightBeta : true;
    const bool transWeightGate = opParamInfo_.transWeightGate != nullptr ? *opParamInfo_.transWeightGate : true;
    const bool transWeightG = opParamInfo_.transWeightG != nullptr ? *opParamInfo_.transWeightG : true;

    int64_t qkvSize = 0L;
    int64_t betaSize = 0L;
    int64_t gateSize = 0L;
    int64_t gSize = 0L;
    if (InferOutFeatures(opName_, opParamInfo_.weightQkv.shape->GetStorageShape(), hiddenSize, transWeightQkv,
                         kda_prolog::WEIGHT_QKV_NAME, qkvSize) != ge::GRAPH_SUCCESS ||
        InferOutFeatures(opName_, opParamInfo_.weightBeta.shape->GetStorageShape(), hiddenSize, transWeightBeta,
                         kda_prolog::WEIGHT_BETA_NAME, betaSize) != ge::GRAPH_SUCCESS ||
        InferOutFeatures(opName_, opParamInfo_.weightGate.shape->GetStorageShape(), hiddenSize, transWeightGate,
                         kda_prolog::WEIGHT_GATE_NAME, gateSize) != ge::GRAPH_SUCCESS ||
        InferOutFeatures(opName_, opParamInfo_.weightG.shape->GetStorageShape(), hiddenSize, transWeightG,
                         kda_prolog::WEIGHT_G_NAME, gSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(tSize > static_cast<int64_t>(UINT32_MAX) || hiddenSize > static_cast<int64_t>(UINT32_MAX) ||
                    qkvSize > static_cast<int64_t>(UINT32_MAX) || betaSize > static_cast<int64_t>(UINT32_MAX) ||
                    gateSize > static_cast<int64_t>(UINT32_MAX) || gSize > static_cast<int64_t>(UINT32_MAX),
                OP_LOGE(opName_, "Shape exceeds uint32 range for tiling base params."), return ge::GRAPH_FAILED);
    baseParams_.tSize = static_cast<uint32_t>(tSize);
    baseParams_.hiddenSize = static_cast<uint32_t>(hiddenSize);
    baseParams_.qkvSize = static_cast<uint32_t>(qkvSize);
    baseParams_.betaSize = static_cast<uint32_t>(betaSize);
    baseParams_.gateSize = static_cast<uint32_t>(gateSize);
    baseParams_.gSize = static_cast<uint32_t>(gSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::ValidateInputShapesMatch()
{
    const bool transWeightQkv = opParamInfo_.transWeightQkv != nullptr ? *opParamInfo_.transWeightQkv : true;
    const int64_t hiddenSize = static_cast<int64_t>(baseParams_.hiddenSize);
    const int64_t qkvSize = static_cast<int64_t>(baseParams_.qkvSize);
    const int64_t mxHiddenSize = (hiddenSize + MX_BLOCK_SIZE - 1L) / MX_BLOCK_SIZE;
    const auto &scaleQkvStorage = opParamInfo_.scaleQkv.shape->GetStorageShape();
    const int64_t scaleDim0 = scaleQkvStorage.GetDim(0);
    const int64_t scaleDim1 = scaleQkvStorage.GetDim(1);
    const int64_t scaleDim2 = scaleQkvStorage.GetDim(2);
    const int64_t expectScaleDim0 = transWeightQkv ? qkvSize : mxHiddenSize;
    const int64_t expectScaleDim1 = transWeightQkv ? mxHiddenSize : qkvSize;
    OP_CHECK_IF(
        scaleDim0 != expectScaleDim0 || scaleDim1 != expectScaleDim1 || scaleDim2 != SCALE_QKV_PACK_NUM,
        OP_LOGE(opName_, "%s shape [%ld, %ld, %ld] does not match expected [%ld, %ld, %ld] (trans_weight_qkv=%d).",
                kda_prolog::SCALE_QKV_NAME, scaleDim0, scaleDim1, scaleDim2, expectScaleDim0, expectScaleDim1,
                SCALE_QKV_PACK_NUM, static_cast<int32_t>(transWeightQkv)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void KdaPrologInfoParser::GenerateInfo(KdaPrologTilingInfo &tilingInfo)
{
    tilingInfo.opName = opName_;
    tilingInfo.platformInfo = platformInfo_;
    tilingInfo.baseParams = baseParams_;
    tilingInfo.transWeightQkv = opParamInfo_.transWeightQkv != nullptr ? *opParamInfo_.transWeightQkv : true;
    tilingInfo.transWeightBeta = opParamInfo_.transWeightBeta != nullptr ? *opParamInfo_.transWeightBeta : true;
    tilingInfo.transWeightGate = opParamInfo_.transWeightGate != nullptr ? *opParamInfo_.transWeightGate : true;
    tilingInfo.transWeightG = opParamInfo_.transWeightG != nullptr ? *opParamInfo_.transWeightG : true;
    tilingInfo.aicNum = aicNum_;
    tilingInfo.l1Size = l1Size_;
    tilingInfo.l0cSize = l0cSize_;

    OP_LOGI(opName_, "KdaProlog ParseAndCheck: T=%u K=%u qkv=%u beta=%u gate=%u g=%u aic=%u l1=%lu l0c=%lu.",
            tilingInfo.baseParams.tSize, tilingInfo.baseParams.hiddenSize, tilingInfo.baseParams.qkvSize,
            tilingInfo.baseParams.betaSize, tilingInfo.baseParams.gateSize, tilingInfo.baseParams.gSize,
            tilingInfo.aicNum, tilingInfo.l1Size, tilingInfo.l0cSize);
}

ge::graphStatus KdaPrologInfoParser::ParseAndCheck(KdaPrologTilingInfo &tilingInfo)
{
    if (GetOpName() != ge::GRAPH_SUCCESS || GetNpuInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetInputParaInfo();
    GetOutputParaInfo();
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || GetAndCheckAttrParaInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckInOutDataType() != ge::GRAPH_SUCCESS || CheckShapeDim() != ge::GRAPH_SUCCESS ||
        GetBaseShapeInfo() != ge::GRAPH_SUCCESS || ValidateInputShapesMatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(tilingInfo);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForKdaProlog(gert::TilingParseContext * /* context */) { return ge::GRAPH_SUCCESS; }

ge::graphStatus KdaPrologTiling::FillBaseParams(const KdaPrologTilingInfo &tilingInfo)
{
    tilingData_.baseParams = tilingInfo.baseParams;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::CalcModuleTilings(const KdaPrologTilingInfo &tilingInfo)
{
    // 与 kernel 四组件一一对应；跨模块 workspace / 核数协调在编排层汇总
    if (KdaPrologMmBggTiling(tilingInfo).CalcTiling(tilingData_.mmBggParams) != ge::GRAPH_SUCCESS ||
        KdaPrologMxQuantTiling(tilingInfo).CalcTiling(tilingData_.mxQuantParams) != ge::GRAPH_SUCCESS ||
        KdaPrologQmmQkvTiling(tilingInfo).CalcTiling(tilingData_.qmmQkvParams) != ge::GRAPH_SUCCESS ||
        KdaPrologSigmoidTiling(tilingInfo).CalcTiling(tilingData_.sigmoidParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::CalcWorkspaceSize(const KdaPrologTilingInfo &tilingInfo)
{
    const auto &base = tilingInfo.baseParams;
    workspaceSize_ = KdaProlog::KdaPrologWorkspace::TotalBytes(base.tSize, base.hiddenSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::SetTilingKey(const KdaPrologTilingInfo &tilingInfo)
{
    // 与 ASCENDC_TPL_ARGS_DECL 顺序一致：TRANS_WEIGHT_{QKV,BETA,GATE,G}
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tilingInfo.transWeightQkv, tilingInfo.transWeightBeta,
                                                  tilingInfo.transWeightGate, tilingInfo.transWeightG);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaProlog",
            "KdaProlog tilingKey=%lu (trans_qkv=%d, trans_beta=%d, trans_gate=%d, trans_g=%d).", tilingKey,
            static_cast<int32_t>(tilingInfo.transWeightQkv), static_cast<int32_t>(tilingInfo.transWeightBeta),
            static_cast<int32_t>(tilingInfo.transWeightGate), static_cast<int32_t>(tilingInfo.transWeightG));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::WriteTilingResult(const KdaPrologTilingInfo &tilingInfo)
{
    const char *opName = tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaProlog";
    OP_CHECK_IF(tilingInfo.platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "platformInfo is null."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_CHECK_IF(blockDim == 0U, OP_LOGE(opName, "blockDim is 0."), return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(1);

    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "GetWorkspaceSizes returned nullptr."),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize_) + static_cast<size_t>(sysWorkspaceSize);

    KdaPrologTilingData *outTiling = context_->GetTilingData<KdaPrologTilingData>();
    OP_CHECK_IF(outTiling == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "GetTilingData returned nullptr."),
                return ge::GRAPH_FAILED);
    *outTiling = tilingData_;

    OP_LOGI(opName, "KdaProlog WriteTilingResult: blockDim=%u workspace=%zu.", blockDim, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::DoTiling(const KdaPrologTilingInfo *tilingInfo)
{
    OP_CHECK_IF(tilingInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaProlog", "Tiling info is null."),
                return ge::GRAPH_FAILED);

    if (FillBaseParams(*tilingInfo) != ge::GRAPH_SUCCESS || CalcModuleTilings(*tilingInfo) != ge::GRAPH_SUCCESS ||
        CalcWorkspaceSize(*tilingInfo) != ge::GRAPH_SUCCESS || SetTilingKey(*tilingInfo) != ge::GRAPH_SUCCESS ||
        WriteTilingResult(*tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForKdaProlog(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaProlog", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    KdaPrologTilingInfo tilingInfo;
    KdaPrologInfoParser infoParser(context);
    if (infoParser.ParseAndCheck(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    KdaPrologTiling tiling(context);
    return tiling.DoTiling(&tilingInfo);
}

IMPL_OP_OPTILING(KdaProlog).Tiling(TilingForKdaProlog).TilingParse<KdaPrologCompileInfo>(TilingPrepareForKdaProlog);
} // namespace optiling
