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
 * \brief
 */

#include "kda_prolog_tiling.h"

#include "../op_kernel/kda_prolog_template_tiling_key.h"

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

ge::graphStatus KdaPrologTiling::DoTiling(const KdaPrologTilingInfo *tilingInfo)
{
    (void)tilingInfo;
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
