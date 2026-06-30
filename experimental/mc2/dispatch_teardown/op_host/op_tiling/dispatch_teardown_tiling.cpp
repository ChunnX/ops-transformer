/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown_tiling.cpp —— DispatchTeardown(mxfp8 多核) tiling。
 *  输入序：0 token_data[R*E*bs,h](fp8) / 1 token_scale[R*E*bs,h/32](e8m0) / 2 wish_cnt[R*E] / 3 expert_ids[bs,topk]
 *  attr 序：0 rank_num(R) / 1 expert_per_rank(E)
 *  blockDim = min(aivNum, A)：A 是 expandX 行上限，kernel 内按实收行(sum wishCnt)再均分。
 */
#include "dispatch_teardown_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DispatchTeardownTilingData tiling;

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) { return ge::GRAPH_FAILED; }
    const int64_t *rankNumPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *expertPerRankPtr = attrs->GetAttrPointer<int64_t>(1);
    if (rankNumPtr == nullptr || expertPerRankPtr == nullptr) { return ge::GRAPH_FAILED; }
    const uint32_t R = static_cast<uint32_t>(*rankNumPtr);
    const uint32_t E = static_cast<uint32_t>(*expertPerRankPtr);

    auto tokenShape = context->GetInputShape(0);
    auto expertIdsShape = context->GetInputShape(3);
    if (tokenShape == nullptr || expertIdsShape == nullptr) { return ge::GRAPH_FAILED; }
    const uint32_t h = static_cast<uint32_t>(tokenShape->GetStorageShape().GetDim(1));
    const uint32_t bs = static_cast<uint32_t>(expertIdsShape->GetStorageShape().GetDim(0));
    const uint32_t topk = static_cast<uint32_t>(expertIdsShape->GetStorageShape().GetDim(1));
    if (h % 32U != 0U) { return ge::GRAPH_FAILED; }        // mxfp8 group=32 硬前提，否则 scale 排布非法
    const uint32_t scaleH = h / 32U;                       // mxfp8 group=32

    const uint32_t minKE = (topk < E) ? topk : E;
    const uint32_t A = R * bs * minKE;

    // 平台 AIV 核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (aivNum == 0U) { aivNum = 1U; }
    uint32_t blockDim = (A < aivNum) ? A : aivNum;          // 行数不足时少开核
    if (blockDim == 0U) { blockDim = 1U; }

    context->SetBlockDim(blockDim);

    tiling.set_bs(bs);
    tiling.set_topk(topk);
    tiling.set_h(h);
    tiling.set_scaleH(scaleH);
    tiling.set_E(E);
    tiling.set_R(R);
    tiling.set_aivNum(blockDim);
    tiling.set_A(A);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0U;
    context->SetTilingKey(0UL);
    return ge::GRAPH_SUCCESS;
}

struct DispatchTeardownCompileInfo {};
static ge::graphStatus TilingParseFunc(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchTeardown)
    .Tiling(TilingFunc)
    .TilingParse<DispatchTeardownCompileInfo>(TilingParseFunc);

}  // namespace optiling
