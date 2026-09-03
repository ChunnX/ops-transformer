/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_matmul_tiling.cpp
 * \brief hosttiling
 */
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>

#include "allto_all_matmul_tiling_base.h"
#include "op_host/util/op_const_def.h"
#include "mc2_exception_dump.h"
#include "../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"

using namespace ge;
using Ops::Transformer::OpTiling::TilingRegistryArch;
using Ops::Transformer::OpTiling::TilingRegistryNew;

namespace MC2Tiling {

static ge::graphStatus AlltoAllMatmulTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == Ops::Base::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmul(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulCompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmul)
    .Tiling(AlltoAllMatmulTilingFunc)
    .TilingParse<AlltoAllMatmulCompileInfo>(TilingParseForAlltoAllMatmul);
    
#if MC2_DFX_ENABLE
// Register exception dump func
inline void AlltoAllMatmulExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AlltoAllMatmul");
    Mc2Exception::Mc2DumpTilingData(args, "AlltoAllMatmul",
        12U,   // tilingGmArgIdx
        0U,    // argsOffset
        11U,   // workspaceGmArgIdx
        offsetof(AlltoAllMatmulTilingData, alltoAllMatmulTilingInfo.x1QuantMode),  // quantModeOffset
        offsetof(AlltoAllMatmulTilingData, workspaceLayout),                       // wsLayoutOffsetNonQuant
        offsetof(AlltoAllQuantMatmulTilingData, workspaceLayout),                 // wsLayoutOffsetQuant
        offsetof(Apace::hcommAllToAllMatmulTilingData, workspaceLayout));        // wsLayoutOffsetApace
}
 
__attribute__((constructor)) void RegisterAlltoAllMatmulExceptionFunc()
{
    IMPL_OP(AlltoAllMatmul)
        .ExceptionDumpParseFunc(AlltoAllMatmulExceptionImplWrapper);
}
#endif // MC2_DFX_ENABLE
} // namespace MC2Tiling
