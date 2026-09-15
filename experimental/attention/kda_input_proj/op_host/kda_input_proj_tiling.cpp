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
 * \brief
 */

#include "kda_input_proj_tiling.h"

#include "../op_kernel/kda_input_proj_template_tiling_key.h"

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

ge::graphStatus KdaInputProjTiling::DoTiling(const KdaInputProjTilingInfo *tilingInfo)
{
    (void)tilingInfo;
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
