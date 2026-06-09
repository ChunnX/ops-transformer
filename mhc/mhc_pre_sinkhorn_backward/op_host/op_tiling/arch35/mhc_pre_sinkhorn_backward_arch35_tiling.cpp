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
 * \file mhc_pre_sinkhorn_backward.cc
 * \brief MhcPreSinkhornBackward tiling file
 */
#include "mhc_pre_sinkhorn_backward_arch35_tiling_base.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"


namespace optiling {
using namespace Ops::Base;
using Ops::NN::Optiling::TilingRegistry;

static ge::graphStatus Tiling4MhcPreSinkhornBackwardArch35(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornBackward rt2.0 is running.");
    auto compile_info = static_cast<const MhcPreSinkhornBackwardCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornBackward compile info is ready, running tiling registry.");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MhcPreSinkhornBackwardArch35(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init TilingPrepare4MhcPreSinkhornBackward.");
    auto ci = context->GetCompiledInfo<MhcPreSinkhornBackwardCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF((ci->core_num <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ub_size = ubSize;
    OP_CHECK_IF(
        (ci->ubSize == 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the IndexPutV2 op.
IMPL_OP_OPTILING(MhcPreSinkhornBackward)
    .Tiling(Tiling4MhcPreSinkhornBackwardArch35)
    .TilingParse<MhcPreSinkhornBackwardCompileInfo>(TilingPrepare4MhcPreSinkhornBackwardArch35);
} // namespace optiling