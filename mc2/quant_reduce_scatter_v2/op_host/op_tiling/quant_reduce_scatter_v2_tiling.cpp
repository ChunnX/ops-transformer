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
 * \file quant_reduce_scatter_v2_tiling.cpp
 * \brief host侧tiling实现
 */
#include <register/op_def_registry.h>
#include "mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_tiling_key.h"
#include "mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_tiling_data.h"
#include "mc2/quant_reduce_scatter/op_host/op_tiling/common/quant_reduce_scatter_util_tiling.h"

namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

/**
 * @brief quant_reduce_scatter_v2算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QuantReduceScatterV2TilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("quant_reduce_scatter_v2", "failed to get tiling context in quant_reduce_scatter_v2."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("quant_reduce_scatter_v2", "failed to get nodeName in quant_reduce_scatter_v2."),
                    return ge::GRAPH_FAILED);

    QuantReduceScatterTilingData *tilingData = context->GetTilingData<QuantReduceScatterTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in quant_reduce_scatter_v2."),
                    return ge::GRAPH_FAILED);

    TilingRunInfo runInfo = {};
    OP_TILING_CHECK(QuantReduceScatterUtilTiling::CheckNpuArch(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "NpuArch is invalid in quant_reduce_scatter_v2."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(QuantReduceScatterUtilTiling::CheckTilingFunc(context, runInfo, OpType::OP_QUANT_REDUCE_SCATTER) !=
                    ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tiling check failed in quant_reduce_scatter_v2."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetHcommCfg(context, tilingData, runInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "SetHCommCfg failed."),
                    return ge::GRAPH_FAILED);
    SetTilingData(context, *tilingData);
    SetXPerBlock(*tilingData, runInfo);
    SetTilingKey(context);
    PrintTilingDataInfo(context, *tilingData);
    return ge::GRAPH_SUCCESS;
}

struct QuantReduceScatterV2CompileInfo {};
 
ge::graphStatus TilingParseForQuantReduceScatterV2(gert::TilingParseContext *context) {
    (void)context;
  	return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantReduceScatterV2)
    .Tiling(QuantReduceScatterV2TilingFunc)
    .TilingParse<QuantReduceScatterV2CompileInfo>(TilingParseForQuantReduceScatterV2);

}  // namespace MC2Tiling
