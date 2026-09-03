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
 * \file matmul_reduce_scatter_v2_tiling_common.cpp
 * \brief
 */
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "matmul_reduce_scatter_v2_tiling_common.h"
#if MC2_DFX_ENABLE
#include "../../../common/utils/mc2_exception_dump.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_c_tiling.h"
#endif

namespace optiling {
ge::graphStatus MatmulReduceScatterTilingV2Func(gert::TilingContext *context);
struct MatmulReduceScatterV2CompileInfo {};
ge::graphStatus TilingParseForMatmulReduceScatterV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulReduceScatterV2)
    .Tiling(MatmulReduceScatterTilingV2Func)
    .TilingParse<MatmulReduceScatterV2CompileInfo>(TilingParseForMatmulReduceScatterV2);

#if MC2_DFX_ENABLE
// Register exception dump func
inline void MatmulReduceScatterV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    OP_LOGE("MatmulReduceScatterV2", "DFX callback begin: soc=%s, nonQuantSize=%zu, quantSize=%zu.",
            socName == nullptr ? "null" : socName, sizeof(Mc2Tiling::MatmulReduceScatterV2TilingData),
            sizeof(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData));
    if (args == nullptr || socName == nullptr || std::strstr(socName, "Ascend950") == nullptr) {
        OP_LOGE("MatmulReduceScatterV2", "DFX callback skipped: null exception info or unsupported SoC.");
        return;
    }
    static_assert(offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, quantMode) ==
                  offsetof(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData, quantMode),
                  "MMRS quant/non-quant tiling must share the quantMode header offset");
    // Save operator evidence before accessing the potentially-faulting CCU context.
    // The hidden HCCL context is args[0], so workspace/tiling are args[9]/args[10].
    Mc2Exception::Mc2DumpTilingData(args, "MatmulReduceScatterV2",
        10U,   // tilingGmArgIdx
        0U,    // argsOffset
        9U,    // workspaceGmArgIdx
        offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, quantMode),                   // quantModeOffset
        offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, workspaceLayout),             // wsLayoutOffsetNonQuant
        offsetof(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData, workspaceLayout)); // wsLayoutOffsetQuant
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "MatmulReduceScatterV2");
    OP_LOGE("MatmulReduceScatterV2", "DFX callback end.");
}
 
__attribute__((constructor)) void RegisterMatmulReduceScatterV2ExceptionFunc()
{
    IMPL_OP(MatmulReduceScatterV2)
        .ExceptionDumpParseFunc(MatmulReduceScatterV2ExceptionImplWrapper);
}
#endif // MC2_DFX_ENABLE

} // namespace optiling
