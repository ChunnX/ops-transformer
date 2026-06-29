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
 * \file cloud_causal_conv1d_fn_tiling.cpp
 * \brief Main tiling entry for CloudCausalConv1dFn decode operator.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "cloud_causal_conv1d_fn_cut_bh_tiling_arch35.h"

namespace optiling {
struct CloudCausalConv1dFnCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// Main tiling function for the decode-only BH implementation.
static ge::graphStatus TilingCloudCausalConv1dFn(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "CloudCausalConv1dFnTiling tiling start");

    auto xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "x shape is null"), return ge::GRAPH_FAILED);
    int64_t xDimNum = static_cast<int64_t>(xShape->GetOriginShape().GetDimNum());
    OP_CHECK_IF(xDimNum != 2 && xDimNum != 3,
                OP_LOGE(context->GetNodeName(), "decode only supports 2D or 3D x, got dimNum=%ld", xDimNum),
                return ge::GRAPH_FAILED);

    CloudCausalConv1dFnCutBHTiling tilingImpl(context);
    ge::graphStatus status = tilingImpl.DoTiling();

    OP_LOGD(context->GetNodeName(), "CloudCausalConv1dFnTiling tiling end, status=%d", status);
    return status;
}

static ge::graphStatus TilingPrepareCloudCausalConv1dFn(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("CloudCausalConv1dFn", "context is null"), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Register the main tiling entry point for unified CloudCausalConv1dFn operator
IMPL_OP_OPTILING(CloudCausalConv1dFn)
    .Tiling(TilingCloudCausalConv1dFn)
    .TilingParse<CloudCausalConv1dFnCompileInfo>(TilingPrepareCloudCausalConv1dFn);
} // namespace optiling
