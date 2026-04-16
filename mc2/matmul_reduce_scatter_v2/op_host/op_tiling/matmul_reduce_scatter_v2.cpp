/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_v2.cpp
 * \brief
 */
#include "matmul_reduce_scatter_v2_tiling.h"
#include "quant_bmm_reduce_scatter_tiling.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(MatmulReduceScatterV2, MatmulReduceScatterV2Tiling, 0);
REGISTER_OPS_TILING_TEMPLATE(MatmulReduceScatterV2, QuantBmmReduceScatterTiling, 1);
constexpr uint32_t ATTR_COMMMODE = 10;
ge::graphStatus MatmulReduceScatterTilingV2Func(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    if (socVersion == "Ascend910B" || socVersion == "Ascend910_93") {
        auto attrs = context->GetAttrs();
        auto commModePtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMMMODE));
        OP_TILING_CHECK((commModePtr == nullptr || !(std::strcmp(commModePtr, "aiv") == 0)),
            OP_LOGE(context->GetNodeName(), "AivModeTiling commMode is invalid. commMode is %s", commModePtr), return ge::GRAPH_FAILED);
        if (std::strcmp(commModePtr, "aiv") == 0) {
            return MatmulReduceScatterTilingV2AivModeFunc(context);
        }
    }
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct MatmulReduceScatterV2CompileInfo {};
ge::graphStatus TilingParseForMatmulReduceScatterV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulReduceScatterV2)
    .Tiling(MatmulReduceScatterTilingV2Func)
    .TilingParse<MatmulReduceScatterV2CompileInfo>(TilingParseForMatmulReduceScatterV2);
}  // namespace optiling

