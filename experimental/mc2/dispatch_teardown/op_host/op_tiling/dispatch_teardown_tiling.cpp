/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown_tiling.cpp —— 最小 tiling：单核轮询，无额外 workspace。
 *  注册形态对齐 all_gather：IMPL_OP_OPTILING(Op).Tiling(fn).TilingParse<Info>(fn)。
 */
#include "dispatch_teardown_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr uint32_t DISPATCH_TEARDOWN_BLOCK_DIM = 1U;  // 单 AIV 核轮询足够

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DispatchTeardownTilingData tiling;
    context->SetBlockDim(DISPATCH_TEARDOWN_BLOCK_DIM);

    // status 输入的元素个数 = 状态区 int32 槽数（单 die 通常 1）。
    uint32_t totalCnt = 1U;
    auto statusShape = context->GetInputShape(0);
    if (statusShape != nullptr) {
        int64_t n = statusShape->GetStorageShape().GetShapeSize();
        if (n > 0) {
            totalCnt = static_cast<uint32_t>(n);
        }
    }
    tiling.set_totalCnt(totalCnt);
    tiling.set_expectVal(1U);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 本算子只读状态区 + 写 1 个 done，无需用户 workspace（系统 workspace 由框架补）。
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0U;

    context->SetTilingKey(1UL);
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
