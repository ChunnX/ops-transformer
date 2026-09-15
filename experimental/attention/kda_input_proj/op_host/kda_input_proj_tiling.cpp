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
 * \file kda_input_proj_tiling.cpp
 * \brief Orchestrates per-module tiling aligned with kernel components.
 */

#include "kda_input_proj_tiling.h"

#include "../op_kernel/kda_input_proj_template_tiling_key.h"
#include "../op_kernel/kda_input_proj_workspace.h"
#include "kda_input_proj_tiling_mm_bgg.h"
#include "kda_input_proj_tiling_mx_quant.h"
#include "kda_input_proj_tiling_qmm_qkv.h"
#include "kda_input_proj_tiling_sigmoid.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
ge::graphStatus KdaInputProjInfoParser::GetOpName()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::GetNpuInfo()
{
    return ge::GRAPH_SUCCESS;
}

void KdaInputProjInfoParser::GetInputParaInfo()
{
}

void KdaInputProjInfoParser::GetOutputParaInfo()
{
}

ge::graphStatus KdaInputProjInfoParser::GetAndCheckAttrParaInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::CheckRequiredInOutExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::GetAndCheckInOutDataType()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::CheckShapeDim()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::GetBaseShapeInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::ValidateInputShapesMatch()
{
    return ge::GRAPH_SUCCESS;
}

void KdaInputProjInfoParser::GenerateInfo(KdaInputProjTilingInfo &tilingInfo)
{
    (void)tilingInfo;
}

ge::graphStatus KdaInputProjInfoParser::ParseAndCheck(KdaInputProjTilingInfo &tilingInfo)
{
    (void)tilingInfo;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForKdaInputProj(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::FillBaseParams(const KdaInputProjTilingInfo &tilingInfo)
{
    tilingData_.baseParams = tilingInfo.baseParams;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::CalcModuleTilings(const KdaInputProjTilingInfo &tilingInfo)
{
    // 与 kernel 四组件一一对应；跨模块 workspace / 核数协调在编排层汇总
    if (KdaInputProjMmBggTiling(tilingInfo).CalcTiling(tilingData_.mmBggParams) != ge::GRAPH_SUCCESS ||
        KdaInputProjMxQuantTiling(tilingInfo).CalcTiling(tilingData_.mxQuantParams) != ge::GRAPH_SUCCESS ||
        KdaInputProjQmmQkvTiling(tilingInfo).CalcTiling(tilingData_.qmmQkvParams) != ge::GRAPH_SUCCESS ||
        KdaInputProjSigmoidTiling(tilingInfo).CalcTiling(tilingData_.sigmoidParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::FillWorkspace(const KdaInputProjTilingInfo &tilingInfo)
{
    const auto &base = tilingInfo.baseParams;
    workspaceSize_ = KdaInputProj::KdaInputProjWorkspace::TotalBytes(base.tSize, base.hiddenSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::SetTilingKey(const KdaInputProjTilingInfo &tilingInfo)
{
    // 与 ASCENDC_TPL_ARGS_DECL 顺序一致：TRANS_WEIGHT_{QKV,BETA,GATE,G}
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(tilingInfo.transWeightQkv, tilingInfo.transWeightBeta, tilingInfo.transWeightGate,
                           tilingInfo.transWeightG);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaInputProj",
            "KdaInputProj tilingKey=%lu (trans_qkv=%d, trans_beta=%d, trans_gate=%d, trans_g=%d).", tilingKey,
            static_cast<int32_t>(tilingInfo.transWeightQkv), static_cast<int32_t>(tilingInfo.transWeightBeta),
            static_cast<int32_t>(tilingInfo.transWeightGate), static_cast<int32_t>(tilingInfo.transWeightG));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::WriteTilingResult(const KdaInputProjTilingInfo &tilingInfo)
{
    uint32_t sysWorkspaceSize = 0U;
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        sysWorkspaceSize = platform_ascendc::PlatformAscendC(platformInfo).GetLibApiWorkSpaceSize();
    }
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaInputProj",
                                            "GetWorkspaceSizes returned nullptr."),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize_) + static_cast<size_t>(sysWorkspaceSize);

    // TODO: SetBlockDim / 写出 tilingData_
    (void)blockDim_;
    (void)tilingData_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::DoTiling(const KdaInputProjTilingInfo *tilingInfo)
{
    OP_CHECK_IF(tilingInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaInputProj", "Tiling info is null."),
                return ge::GRAPH_FAILED);

    if (FillBaseParams(*tilingInfo) != ge::GRAPH_SUCCESS || CalcModuleTilings(*tilingInfo) != ge::GRAPH_SUCCESS ||
        FillWorkspace(*tilingInfo) != ge::GRAPH_SUCCESS || SetTilingKey(*tilingInfo) != ge::GRAPH_SUCCESS ||
        WriteTilingResult(*tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForKdaInputProj(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaInputProj", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    KdaInputProjTilingInfo tilingInfo;
    KdaInputProjInfoParser infoParser(context);
    if (infoParser.ParseAndCheck(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    KdaInputProjTiling tiling(context);
    return tiling.DoTiling(&tilingInfo);
}

IMPL_OP_OPTILING(KdaInputProj)
    .Tiling(TilingForKdaInputProj)
    .TilingParse<KdaInputProjCompileInfo>(TilingPrepareForKdaInputProj);
} // namespace optiling
