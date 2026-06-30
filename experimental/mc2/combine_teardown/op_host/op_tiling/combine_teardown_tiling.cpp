/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  combine_teardown_tiling.cpp —— CombineTeardown(mxfp8 多核) tiling。
 *  输入序：0 combine_data[GE*bs,h](fp8) / 1 combine_scale[GE*bs,h/32](e8m0) /
 *          2 expert_scales[bs,topk](fp32) / 3 expand_idx[bs,topk] / 4 expert_ids[bs,topk]
 *  attr 序：0 rank_num(R) / 1 expert_per_rank(E)；GE=R*E。
 *  blockDim = min(aivNum, bs*numChunks)，numChunks=ceil(h/1024)。
 */
#include "combine_teardown_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t COMBINE_CHUNK = 1024U;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    CombineTeardownTilingData tiling;

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) { return ge::GRAPH_FAILED; }
    const int64_t *rankNumPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *expertPerRankPtr = attrs->GetAttrPointer<int64_t>(1);
    if (rankNumPtr == nullptr || expertPerRankPtr == nullptr) { return ge::GRAPH_FAILED; }
    const uint32_t R = static_cast<uint32_t>(*rankNumPtr);
    const uint32_t E = static_cast<uint32_t>(*expertPerRankPtr);

    auto cdShape = context->GetInputShape(0);        // [GE*bs, h]
    auto scalesShape = context->GetInputShape(2);    // [bs, topk]
    if (cdShape == nullptr || scalesShape == nullptr) { return ge::GRAPH_FAILED; }
    const uint32_t h = static_cast<uint32_t>(cdShape->GetStorageShape().GetDim(1));
    const uint32_t bs = static_cast<uint32_t>(scalesShape->GetStorageShape().GetDim(0));
    const uint32_t topk = static_cast<uint32_t>(scalesShape->GetStorageShape().GetDim(1));
    if (h % 32U != 0U) { return ge::GRAPH_FAILED; }        // mxfp8 group=32 硬前提
    const uint32_t scaleH = h / 32U;

    const uint32_t numChunks = (h + COMBINE_CHUNK - 1U) / COMBINE_CHUNK;
    const uint32_t workItems = bs * numChunks;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (aivNum == 0U) { aivNum = 1U; }
    uint32_t blockDim = (workItems < aivNum) ? workItems : aivNum;
    if (blockDim == 0U) { blockDim = 1U; }

    context->SetBlockDim(blockDim);

    tiling.set_bs(bs);
    tiling.set_topk(topk);
    tiling.set_h(h);
    tiling.set_scaleH(scaleH);
    tiling.set_globalExperts(R * E);
    tiling.set_aivNum(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0U;
    context->SetTilingKey(0UL);
    return ge::GRAPH_SUCCESS;
}

struct CombineTeardownCompileInfo {};
static ge::graphStatus TilingParseFunc(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CombineTeardown)
    .Tiling(TilingFunc)
    .TilingParse<CombineTeardownCompileInfo>(TilingParseFunc);

}  // namespace optiling
