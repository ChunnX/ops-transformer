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
 * \file all_gather_matmul_v2_tiling_common.cpp
 * \brief
 */

#include "mc2_log.h"
#include "op_host/tiling_templates_registry.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "all_gather_matmul_v2_tiling_common.h"
#include "mc2_exception_dump.h"
#include "../../op_kernel/all_gather_matmul_tiling_data.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
ge::graphStatus AllGatherMatmulTilingV2Func(gert::TilingContext *context);
struct AllGatherMatmulCompileInfo {};
ge::graphStatus TilingParseForAllGatherMatmulV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherMatmulV2)
    .Tiling(AllGatherMatmulTilingV2Func)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmulV2);

#if MC2_DFX_ENABLE
// Register exception dump func
inline void AllGatherMatmulV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AllGatherMatmulV2");

    Mc2Exception::Mc2DumpTilingData(
        args, "AllGatherMatmulV2",
        11U,                                                                 // tilingGmArgIdx
        0U,                                                                  // argsOffset
        10U,                                                                 // workspaceGmArgIdx
        offsetof(Mc2Tiling::AllGatherMatmulTilingDataV2, quantMode),         // quantModeOffset
        offsetof(Mc2Tiling::AllGatherMatmulTilingDataV2, workspaceLayout),   // wsLayoutOffsetNonQuant
        offsetof(Mc2Tiling::AllGatherMatmulTilingDataFp8, workspaceLayout)); // wsLayoutOffsetQuant
}

__attribute__((constructor)) void RegisterAllGatherMatmulV2ExceptionFunc()
{
    IMPL_OP(AllGatherMatmulV2).ExceptionDumpParseFunc(AllGatherMatmulV2ExceptionImplWrapper);
}
#endif // MC2_DFX_ENABLE

} // namespace optiling
