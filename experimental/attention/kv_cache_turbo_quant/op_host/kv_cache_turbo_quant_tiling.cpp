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
 * \file kv_cache_turbo_quant_tiling.cpp
 * \brief Host tiling for the five-kernel TurboQuant encode pipeline.
 */
#include "kv_cache_turbo_quant_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

namespace optiling {

constexpr int64_t kHeadDim = 128;
constexpr uint32_t kMinRowsPerVec = 8;
constexpr size_t kMseBitsAttr = 0;

static uint32_t SelectCubeCores(int64_t rows, uint32_t aic)
{
    if (rows < kHeadDim || aic == 0) {
        return 1;
    }
    for (uint32_t c = aic; c >= 1; --c) {
        if (rows % static_cast<int64_t>(c) != 0) {
            continue;
        }
        const int64_t per = rows / static_cast<int64_t>(c);
        if (per % kHeadDim == 0 || per < kHeadDim) {
            return c;
        }
    }
    return 1;
}

static uint32_t SelectVecCores(int64_t rows, uint32_t aiv)
{
    uint32_t cores = aiv == 0 ? 1 : aiv;
    if (rows > 0 && static_cast<uint32_t>(rows) < cores) {
        cores = static_cast<uint32_t>(rows);
    }
    while (cores > 1 && static_cast<uint32_t>(rows) / cores < kMinRowsPerVec) {
        cores /= 2;
    }
    return cores == 0 ? 1 : cores;
}

static ge::graphStatus TilingKvCacheTurboQuant(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *kvShape = context->GetInputShape(0);
    const gert::StorageShape *rotShape = context->GetInputShape(1);
    const gert::StorageShape *qjlShape = context->GetInputShape(2);
    if (kvShape == nullptr || rotShape == nullptr || qjlShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &kv = kvShape->GetStorageShape();
    const gert::Shape &rot = rotShape->GetStorageShape();
    const gert::Shape &qjl = qjlShape->GetStorageShape();
    if (kv.GetDimNum() != 3 || rot.GetDimNum() != 2 || qjl.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = kv.GetDim(0);
    const int64_t heads = kv.GetDim(1);
    const int64_t headDim = kv.GetDim(2);
    const int64_t qjlDim = qjl.GetDim(0);
    if (tokens <= 0 || heads < 4 || heads > 32 || headDim != kHeadDim || qjlDim != kHeadDim) {
        return ge::GRAPH_FAILED;
    }
    if (rot.GetDim(0) != kHeadDim || rot.GetDim(1) != kHeadDim || qjl.GetDim(1) != kHeadDim) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t bits = 3;
    if (attrs != nullptr) {
        const int64_t *bitsAttr = attrs->GetAttrPointer<int64_t>(kMseBitsAttr);
        if (bitsAttr != nullptr) {
            bits = *bitsAttr;
        }
    }
    if (bits != 2 && bits != 3 && bits != 4) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aiv = platform.GetCoreNumAiv();
    const uint32_t aic = platform.GetCoreNumAic();
    const int64_t rows = tokens * heads;
    const uint32_t vecCores = SelectVecCores(rows, aiv);
    const uint32_t cubeCores = SelectCubeCores(rows, aic);

    KvCacheTurboQuantTilingData tiling;
    tiling.set_rowCount(rows);
    tiling.set_headDim(headDim);
    tiling.set_qjlDim(qjlDim);
    tiling.set_mseBits(bits);
    tiling.set_vecCoreNum(vecCores);
    tiling.set_cubeCoreNum(cubeCores);
    tiling.set_idxBytes(static_cast<uint32_t>((headDim * bits) / 8));
    tiling.set_qjlBytes(static_cast<uint32_t>(qjlDim / 8));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(cubeCores);
    context->SetTilingKey(bits);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const size_t inter = static_cast<size_t>(rows) * static_cast<size_t>(kHeadDim) * sizeof(float) * 4;
    const size_t normFp = static_cast<size_t>(rows) * sizeof(float) * 3;
    constexpr size_t kMatmulWorkspace = 32U * 1024U * 1024U;
    workspaces[0] = inter + normFp + kMatmulWorkspace + 4096U;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheTurboQuant).Tiling(TilingKvCacheTurboQuant);

} // namespace optiling
