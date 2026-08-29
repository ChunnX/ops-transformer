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
#include "kda_prolog_tiling_mm_bgg.h"
#include "kda_prolog_tiling_mx_quant.h"
#include "kda_prolog_tiling_qmm_qkv.h"
#include "kda_prolog_tiling_sigmoid.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
ge::graphStatus KdaPrologInfoParser::GetOpName()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::GetNpuInfo()
{
    return ge::GRAPH_SUCCESS;
}

void KdaPrologInfoParser::GetInputParaInfo()
{
}

void KdaPrologInfoParser::GetOutputParaInfo()
{
}

ge::graphStatus KdaPrologInfoParser::GetAndCheckAttrParaInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::CheckRequiredInOutExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::GetAndCheckInOutDataType()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::CheckShapeDim()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::GetBaseShapeInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologInfoParser::ValidateInputShapesMatch()
{
    return ge::GRAPH_SUCCESS;
}

void KdaPrologInfoParser::GenerateInfo(KdaPrologTilingInfo &tilingInfo)
{
    (void)tilingInfo;
}

ge::graphStatus KdaPrologInfoParser::ParseAndCheck(KdaPrologTilingInfo &tilingInfo)
{
    (void)tilingInfo;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForKdaProlog(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

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

ge::graphStatus KdaPrologTiling::SetTilingKey(const KdaPrologTilingInfo &tilingInfo)
{
    // 与 ASCENDC_TPL_ARGS_DECL 顺序一致：TRANS_WEIGHT_{QKV,BETA,GATE,G}
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(tilingInfo.transWeightQkv, tilingInfo.transWeightBeta, tilingInfo.transWeightGate,
                           tilingInfo.transWeightG);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaProlog",
            "KdaProlog tilingKey=%lu (trans_qkv=%d, trans_beta=%d, trans_gate=%d, trans_g=%d).", tilingKey,
            static_cast<int32_t>(tilingInfo.transWeightQkv), static_cast<int32_t>(tilingInfo.transWeightBeta),
            static_cast<int32_t>(tilingInfo.transWeightGate), static_cast<int32_t>(tilingInfo.transWeightG));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::WriteTilingResult(const KdaPrologTilingInfo &tilingInfo)
{
    (void)tilingInfo;
    // TODO: SetBlockDim / SetWorkspaceSizes / 写出 tilingData_
    (void)blockDim_;
    (void)workspaceSize_;
    (void)tilingData_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaPrologTiling::DoTiling(const KdaPrologTilingInfo *tilingInfo)
{
    OP_CHECK_IF(tilingInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaProlog", "Tiling info is null."),
                return ge::GRAPH_FAILED);

    if (FillBaseParams(*tilingInfo) != ge::GRAPH_SUCCESS || CalcModuleTilings(*tilingInfo) != ge::GRAPH_SUCCESS ||
        SetTilingKey(*tilingInfo) != ge::GRAPH_SUCCESS || WriteTilingResult(*tilingInfo) != ge::GRAPH_SUCCESS) {
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

IMPL_OP_OPTILING(KdaProlog)
    .Tiling(TilingForKdaProlog)
    .TilingParse<KdaPrologCompileInfo>(TilingPrepareForKdaProlog);
} // namespace optiling
