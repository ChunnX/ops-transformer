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
 * \file grouped_mat_mul_allto_allv_tiling_a5.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "grouped_mat_mul_allto_allv_tiling_a5.h"
#include "grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace optiling::Mc2GroupedMatmul;

namespace optiling {

bool GmmAlltoAllvTilingA5::IsCapable()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return false);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto npuArch = ascendcPlatform.GetCurNpuArch();

    if (npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_RESV) {
        OP_LOGI(opName_, "GmmAlltoAllvTilingA5 capable for A5 architecture.");
        return true;
    }
    OP_LOGI(opName_, "Skip GmmAlltoAllvTilingA5 for non-A5 architecture.");
    return false;
}

ge::graphStatus GmmAlltoAllvTilingA5::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t GmmAlltoAllvTilingA5::GetTilingKey() const
{
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, 0, 0);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, 0, 0, tilingKey);
    return tilingKey;
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatMulAlltoAllv, GmmAlltoAllvTilingA5, 0);

} // namespace optiling