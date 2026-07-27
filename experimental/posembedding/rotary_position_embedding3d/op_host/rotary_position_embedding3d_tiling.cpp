/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/rotary_position_embedding3d_tiling_data.h"
#include "../op_kernel/rotary_position_embedding3d_tiling_key.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

constexpr uint32_t BLOCK_DIM = 8;
constexpr int64_t TILE_NUM = 1;
constexpr float FREQ_BASE = 10000.0f;

struct CompileInfo {};

static void ComputeBandDims(int64_t headDim, int64_t &tBand, int64_t &hBand, int64_t &wBand)
{
    // 2:1:1 ratio: T = D/2, H = D/4, W = D/4, all even
    int64_t unit = headDim / 4;
    if (unit % 2 != 0) {
        unit -= 1;
    }
    tBand = unit * 2;
    hBand = unit;
    wBand = headDim - tBand - hBand;
    if (wBand % 2 != 0) {
        wBand -= 1;
        tBand += 1;
        if (tBand % 2 != 0) {
            tBand -= 1;
            hBand += 1;
        }
    }
    if (hBand % 2 != 0) {
        hBand -= 1;
        tBand += 1;
    }
    if (tBand % 2 != 0) {
        tBand -= 1;
        hBand += 1;
    }
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &coreNum)
{
    auto *platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto plat = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = plat.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseInputs(gert::TilingContext *context, int64_t &totalLength, int64_t &headDim,
                                   int64_t &seqLen)
{
    auto inputShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    int64_t dimNum = inputShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "input dim < 2"), return ge::GRAPH_FAILED);
    headDim = static_cast<int64_t>(inputShape.GetDim(dimNum - 1));
    totalLength = 1;
    for (int64_t i = 0; i < dimNum; i++) {
        totalLength *= static_cast<int64_t>(inputShape.GetDim(i));
    }
    seqLen = 1;
    for (int64_t i = 1; i < dimNum - 1; i++) {
        seqLen *= static_cast<int64_t>(inputShape.GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalLength;
    int64_t headDim;
    int64_t seqLen;
    OP_CHECK_IF(ParseInputs(context, totalLength, headDim, seqLen) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ParseInputs error"), return ge::GRAPH_FAILED);

    auto *td = context->GetTilingData<RotaryPositionEmbedding3dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    memset_s(td, sizeof(RotaryPositionEmbedding3dTilingData), 0, sizeof(RotaryPositionEmbedding3dTilingData));

    int64_t tBand = 0;
    int64_t hBand = 0;
    int64_t wBand = 0;
    ComputeBandDims(headDim, tBand, hBand, wBand);

    td->totalLength = totalLength;
    td->headDim = headDim;
    td->seqLen = seqLen;
    td->tBand = tBand;
    td->hBand = hBand;
    td->tileNum = TILE_NUM;
    td->blockLength = totalLength / BLOCK_DIM;
    td->freqBase = FREQ_BASE;
    td->rT = static_cast<float>(std::pow(FREQ_BASE, -2.0 / static_cast<double>(tBand)));
    td->rH = static_cast<float>(std::pow(FREQ_BASE, -2.0 / static_cast<double>(hBand)));
    td->rW = static_cast<float>(std::pow(FREQ_BASE, -2.0 / static_cast<double>(wBand)));

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 16U * 1024U * 1024U;

    context->SetBlockDim(BLOCK_DIM);
    auto inputDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t tilingKey = 0;
    if (inputDtype == ge::DT_FLOAT16) {
        tilingKey = 1;
    } else if (inputDtype == ge::DT_BF16) {
        tilingKey = 2;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(RotaryPositionEmbedding3d).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
} // namespace optiling
