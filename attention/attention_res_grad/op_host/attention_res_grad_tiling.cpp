/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "attention_res_grad_tiling.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/attention_res_grad_tiling_key.h"
namespace optiling {
using namespace Ops::Transformer::OpTiling;
constexpr uint32_t ALIGN_512B = 512;

static void PrintInfo(gert::TilingContext *ctx, AttentionResGradTilingData& td) {
    OP_LOGD(ctx, " B=%ld N=%ld N1=%ld H=%ld cores=%ld perWksp=%lu",
            td.get_batchSize(), td.get_numBlocks(), td.get_totalBlocks(), td.get_hiddenSize(),
            td.get_coreNum(), td.get_perCoreWkspBytes());
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* ctx, uint64_t& ub, uint32_t& coreNum) {
    auto ci = ctx->GetCompileInfo<AttentionResGradCompileInfo>();
    coreNum = ci->coreNum;
    ub = ci->ubSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Check(gert::TilingContext* ctx) {
    auto ps = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape());
    auto br = EnsureNotScalar(ctx->GetInputShape(1)->GetStorageShape());
    OP_CHECK_IF(ps.GetDimNum() != 2, OP_LOGE(ctx, "prefix_sum must be 2D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(br.GetDimNum() != 3, OP_LOGE(ctx, "block_residual must be 3D"), return ge::GRAPH_FAILED);

    int64_t B  = ps.GetDim(0);
    int64_t H  = ps.GetDim(1);
    int64_t Bb = br.GetDim(0);
    int64_t N  = br.GetDim(1);
    int64_t Hb = br.GetDim(2);
    OP_CHECK_IF(B != Bb || H != Hb, OP_LOGE(ctx, "shape mismatch"), return ge::GRAPH_FAILED);

    OP_LOGD(ctx, "shape: B=%ld N=%ld H=%ld", B, N, H);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(gert::TilingContext* ctx, int64_t B, int64_t N, int64_t H, uint32_t coreNum)
{
    // number of blocks (including the prefix-sum block)
    int64_t N1 = N + 1;

    // per-core workspace: H floats, 512B-aligned for DMA efficiency
    uint64_t perWksp = (static_cast<uint64_t>(H) * sizeof(float) + ALIGN_512B - 1) / ALIGN_512B * ALIGN_512B;

    AttentionResGradTilingData td;
    td.set_batchSize(B);
    td.set_numBlocks(N);
    td.set_totalBlocks(N1);
    td.set_hiddenSize(H);
    // per-core batch range and workspace offset are computed in kernel via GetBlockIdx()
    td.set_coreBatchStart(0);
    td.set_coreBatchEnd(B);
    td.set_coreBatchCount(0);
    td.set_gradScoreWeightWkspOff(0);
    td.set_coreNum(static_cast<int64_t>(coreNum));
    td.set_perCoreWkspBytes(perWksp);

    // Hard SyncAll requires all launched cores to start together in batch mode.
    ctx->SetScheduleMode(1);
    ctx->SetBlockDim(coreNum);
    td.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(td.GetDataSize());

    PrintInfo(ctx, td);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcWorkspaceSize(gert::TilingContext* ctx) {
    int64_t H = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape()).GetDim(1);
    auto ci = ctx->GetCompileInfo<AttentionResGradCompileInfo>();
    uint64_t per = (static_cast<uint64_t>(H) * sizeof(float) + ALIGN_512B - 1) / ALIGN_512B * ALIGN_512B;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *ws = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ws);
    ws[0] = static_cast<size_t>(per * static_cast<uint64_t>(ci->coreNum) + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* ctx) {
    uint64_t ub = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(ctx, ub, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "get platform info failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(Check(ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "check shape failed"), return ge::GRAPH_FAILED);

    auto inputDesc = ctx->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TPL_SCH_MODE_FP16);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TPL_SCH_MODE_BF16);
    } else {
        OP_LOGE(ctx, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    ctx->SetTilingKey(tilingKey);

    auto ps = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape());
    auto br = EnsureNotScalar(ctx->GetInputShape(1)->GetStorageShape());
    OP_CHECK_IF(DoTiling(ctx, ps.GetDim(0), br.GetDim(1), ps.GetDim(1), coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "tiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcWorkspaceSize(ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "workspace failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Parse(gert::TilingParseContext* ctx) {
    auto ci = ctx->GetCompiledInfo<AttentionResGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ci);

    auto plat = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint32_t aivNum = plat.GetCoreNumAiv();
    ci->coreNum = aivNum;
    OP_CHECK_IF(ci->coreNum == 0, OP_LOGE(ctx, "coreNum is 0"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ci->ubSize);
    OP_CHECK_IF(ci->ubSize == 0, OP_LOGE(ctx, "ubSize is 0"), return ge::GRAPH_FAILED);

    OP_LOGD(ctx, "parse: ub=%luB coreNum=%u", ci->ubSize, ci->coreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionResGrad).Tiling(TilingFunc).TilingParse<AttentionResGradCompileInfo>(Parse);
}
