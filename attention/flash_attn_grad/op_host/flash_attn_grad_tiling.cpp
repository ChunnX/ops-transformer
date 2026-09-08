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
 * \file flash_attn_grad_tiling.cpp
 * \brief L0 入口：解析 -> 校验 -> 规划 -> 写回。业务判断都在下层，这里不做决策。
 */

#include "flash_attn_grad_tiling.h"

#include <cmath>

#include "flash_attn_grad_tiling_check.h"
#include "flash_attn_grad_tiling_info.h"
#include "flash_attn_grad_tiling_info_parser.h"
#include "flash_attn_grad_tiling_plan.h"
#include "log/log.h"

namespace optiling {
namespace {

void CommitTilingData(gert::TilingContext *context, const FagParsedInfo &info)
{
    FlashAttnGradTilingData *tilingData = context->GetTilingData<FlashAttnGradTilingData>();
    tilingData->b = info.b;
    tilingData->s1 = info.s1;
    tilingData->s2 = info.s2;
    tilingData->n1 = info.n1;
    tilingData->n2 = info.n2;
    tilingData->d = info.d;
    tilingData->dv = info.dv;
    tilingData->scaleValue = info.scaleValue == 0.0f ? 1.0f / sqrt(static_cast<float>(info.d)) : info.scaleValue;
    tilingData->gSize = info.g;
    tilingData->viewD0Q = info.viewD0Q;
    tilingData->viewD2Q = info.viewD2Q;
    tilingData->coefB0Q = info.coefB0Q;
    tilingData->viewD0KV = info.viewD0KV;
    tilingData->viewD2KV = info.viewD2KV;
    tilingData->coefB0KV = info.coefB0KV;
    tilingData->coefN0 = info.coefN0;
    tilingData->coefN2 = info.coefN2;
}

void CommitPlan(gert::TilingContext *context, const FagParsedInfo &info, const FagTilingPlan &plan)
{
    context->SetBlockDim(plan.schedule.blockDim);
    // BN2GS1S2 的 pre/post 用 sync_all，启动核数不是满片时必须 batch mode，
    // 否则多流下可能卡死。BN2 没有 sync_all。
    if (plan.kernel.tmpl == TMPL_BN2GS1S2) {
        context->SetScheduleMode(1);
    }

    CommitTilingData(context, info);
    context->SetTilingKey(plan.tilingKey);
    context->GetWorkspaceSizes(1)[0] = plan.workspace.totalSize;
}

void LogPlan(gert::TilingContext *context, const FagParsedInfo &info, const FagTilingPlan &plan,
             const FagRouteTrace &trace)
{
    OP_LOGI(context->GetNodeName(),
            "FlashAttnGrad tiling: usedCube=%u, usedBlockDim=%u, useActualCores=%d, s1Outer=%ld, s2Outer=%ld, "
            "layoutStr=%s, layout=%ld, maskMode=%ld, hasAttenMask=%d, hasSeqused=%d, hasSinks=%d, isEmptyShape=%d, "
            "template=%ld, isBn2MultiBlk=%ld, swizzle=%ld, dAlign=%ld, dvAlign=%ld, "
            "B=%ld, S1=%ld, S2=%ld, N1=%ld, N2=%ld, G=%ld, D=%ld, Dv=%ld, "
            "viewQ=[%ld,S,%ld,D] b0=%ld, viewKV=[%ld,S,%ld,D] b0=%ld, coefN=(%ld,%ld), "
            "tilingKey=%llu, workspaceSize=%zu.",
            plan.schedule.usedCube, plan.schedule.blockDim, static_cast<int>(plan.schedule.useActualCores),
            plan.schedule.s1Outer, plan.schedule.s2Outer, info.layoutQ.c_str(), info.layout, info.maskMode,
            static_cast<int>(info.hasAttenMask), static_cast<int>(info.hasSeqused), static_cast<int>(info.hasSinks),
            static_cast<int>(IsEmptyShape(info)), plan.kernel.tmpl, plan.kernel.isBn2MultiBlk, plan.schedule.swizzle,
            plan.kernel.dAlign, plan.kernel.dvAlign, info.b, info.s1, info.s2, info.n1, info.n2, info.g, info.d,
            info.dv, info.viewD0Q, info.viewD2Q, info.coefB0Q, info.viewD0KV, info.viewD2KV, info.coefB0KV,
            info.coefN0, info.coefN2, plan.tilingKey, plan.workspace.totalSize);

    // 决策追溯：上板发现别的模板更快时，先看这行就知道今天为什么选了它。
    OP_LOGI(context->GetNodeName(), "FlashAttnGrad route: chosen=%s, swizzle=%s.", trace.chosen, trace.swizzleReject);
    for (size_t i = 0; i < trace.rejectedNum; ++i) {
        OP_LOGI(context->GetNodeName(), "FlashAttnGrad route: rejected %s by %s.", trace.rejectedName[i],
                trace.rejectedReason[i]);
    }
}

ge::graphStatus FlashAttnGradTilingFunc(gert::TilingContext *context)
{
    FagParsedInfo info;
    ge::graphStatus ret = ParsePlatform(context, info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = FlashAttnGradCheck::CheckParams(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckParams failed."),
                return ge::GRAPH_PARAM_INVALID);

    ret = ParseFlashAttnGradInfo(context, info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    FagTilingPlan plan;
    FagRouteTrace trace;
    ret = BuildTilingPlan(context, info, plan, trace);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    CommitPlan(context, info, plan);
    LogPlan(context, info, plan, trace);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForFlashAttnGrad([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(FlashAttnGrad)
    .Tiling(FlashAttnGradTilingFunc)
    .TilingParse<FlashAttnGradCompileInfo>(TilingParseForFlashAttnGrad);

} // namespace optiling
