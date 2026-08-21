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
#include "mc2_exception_dump.h"
#include "../../../op_kernel/matmul_reduce_scatter_v2_c_tiling.h"
#include <cstddef>
#include <cstring>

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

// Register exception dump func
inline void MatmulReduceScatterV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "MatmulReduceScatterV2");

    // TODO: 待 MatmulReduceScatterV2 TilingData 加入 x1QuantMode 后，传入 quantModeOffset 和两个不同 offset
    Mc2Exception::Mc2DumpTilingData(args, "MatmulReduceScatterV2",
        10U,   // tilingGmArgIdx
        0U,    // argsOffset
        9U,    // workspaceGmArgIdx
        0U,    // quantModeOffset (暂无)
        offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, workspaceLayout),  // wsLayoutOffsetNonQuant
        offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, workspaceLayout)); // wsLayoutOffsetQuant (暂同)
}

__attribute__((constructor)) void RegisterMatmulReduceScatterV2ExceptionFunc()
{
    IMPL_OP(MatmulReduceScatterV2)
        .ExceptionDumpParseFunc(MatmulReduceScatterV2ExceptionImplWrapper);
}

} // namespace optiling
