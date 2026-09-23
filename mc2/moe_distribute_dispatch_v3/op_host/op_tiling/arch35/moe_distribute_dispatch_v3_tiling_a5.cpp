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
 * \file moe_distribute_dispatch_v3_tiling_a5.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v3_tiling_a5.h"
#include "../../../../moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.h"

namespace optiling {
ge::graphStatus MoeDistributeDispatchV3TilingFuncA5::MoeDistributeDispatchA3TilingFuncImplPublic(
    gert::TilingContext *context, DispatchV2Config &config)
{
    MoeDistributeDispatchV2TilingFuncA5 impl;
    return impl.MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV3TilingFuncImplA5(gert::TilingContext *context)
{
    MoeDistributeDispatchV3TilingFuncA5 impl;
    ge::graphStatus ret = impl.MoeDistributeDispatchV3TilingFunc(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // A5 上 V2/V3/V4 单算子 API 的 fullmesh 通路均路由本算子——CCU 控制面使能与 V2 入口同款
    // （fullmesh_v2 + env MOE_DISPATCH_USE_CCU_CTRL=1 时配置，tiling key 复用不变，否则显式关）。
    // V3 属性布局与 V2 不同：commAlg=attr12（对齐 v3_tiling.cpp attrCommAlgIndex）、无 groupEp 属性
    // （通信绑定经 mc2Context 输入，groupEp 传空串同 SetHcommCfg 空 group 先例）
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto commAlgPtr = attrs->GetAttrPointer<char>(12); // 12: V3 布局 commAlg 索引
    MoeDistributeDispatchV2TilingFuncA5 v2Impl;
    return v2Impl.TryEnableCcuCtrl(context, commAlgPtr, "");
}
IMPL_OP_OPTILING(MoeDistributeDispatchV3)
    .Tiling(MoeDistributeDispatchV3TilingFuncImplA5)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV3);
} // namespace optiling
