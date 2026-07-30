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
 * \file allto_all_matmul_apace_tiling.cpp
 * \brief Tiling implementation for AlltoAllMatmulApace
 */

#include "allto_all_matmul_apace_tiling.h"
#include "arch35/allto_all_matmul_apace_tiling_base.h"

#include <cstdio>
#include <string>
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"
#include "apace/tiling/quant_matmul_mx_tiling_swat.h"
#include "common/utils/mc2_log.h"

using namespace ge;
using namespace Mc2Log;

uint64_t GetMatmulTilingKey()
{
    return 16;
}

void BuildCommTilingData(uint64_t m, uint64_t ka, uint64_t headMSize, CommTilingData &commData)
{
    uint64_t tileCnt = m / headMSize;
    commData.splitAxisTileSize = headMSize;
    commData.splitAxisTileCnt = tileCnt;
    commData.splitAxisTailSize = 0;
    commData.splitAxisTailCnt = 0;
    commData.nonSplitAxisSize = ka;

    OP_LOGI("AlltoAllMatmulApace", "CommTiling: tileSize=%lu, tileCnt=%lu, nonSplitAxisSize=%lu",
            commData.splitAxisTileSize, commData.splitAxisTileCnt, commData.nonSplitAxisSize);
}

void BuildMatmulTilingData(uint64_t m, uint64_t n, uint64_t ka, uint32_t aDtype, uint32_t bDtype,
                           QuantMatmulTilingData &matmulData)
{
    if (ka == 0) {
        OP_LOGE("AlltoAllMatmulApace", "BuildMatmulTilingData: ka=0, invalid K axis after world_size split");
        return;
    }
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
    tilingEngine.GetTilingData(m, n, ka, false, true, matmulData);
}

AlltoAllDtypeCombo GetDtypeCombo(uint32_t x1Dtype, uint32_t x2Dtype)
{
    if (x1Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN) &&
        x2Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN)) {
        return E4M3_E4M3;
    }
    if (x1Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E5M2) && x2Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E5M2)) {
        return E5M2_E5M2;
    }
    if (x1Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN) &&
        x2Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E5M2)) {
        return E4M3_E5M2;
    }
    if (x1Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E5M2) &&
        x2Dtype == static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN)) {
        return E5M2_E4M3;
    }
    OP_LOGW("AlltoAllMatmulApace", "Unknown dtype combo x1=%u x2=%u, falling back to E4M3_E4M3", x1Dtype, x2Dtype);
    return E4M3_E4M3;
}

namespace MC2Tiling {

using Ops::Transformer::OpTiling::TilingRegistryArch;
using Ops::Transformer::OpTiling::TilingRegistryNew;

static ge::graphStatus AlltoAllMatmulApaceTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmulApace(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulApaceCompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmulApace)
    .Tiling(AlltoAllMatmulApaceTilingFunc)
    .TilingParse<AlltoAllMatmulApaceCompileInfo>(TilingParseForAlltoAllMatmulApace);

} // namespace MC2Tiling
