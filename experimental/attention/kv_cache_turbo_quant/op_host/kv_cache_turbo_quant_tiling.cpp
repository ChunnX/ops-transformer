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
 * \brief MIX Cube：每 AIV 分片；Kernel 内按 64 行 Iterate。
 */
#include "kv_cache_turbo_quant_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

namespace {
constexpr int64_t kHeadDim = 128;
constexpr int64_t kMseBitsMin = 2;
constexpr int64_t kMseBitsMax = 4;
constexpr size_t kMatmulSysWorkspace = 32UL * 1024UL * 1024UL;
constexpr uint32_t kGemmTile = 64U;
} // namespace

static ge::graphStatus TilingKvCacheTurboQuant(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *mseBitsPtr = attrs->GetAttrPointer<int64_t>(0);
    int64_t mseBits = (mseBitsPtr == nullptr) ? 3 : *mseBitsPtr;
    if (mseBits < kMseBitsMin || mseBits > kMseBitsMax) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *kvShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    const auto &kv = kvShape->GetStorageShape();
    if (kv.GetDimNum() != 3 || kv.GetDim(2) != kHeadDim) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = kv.GetDim(0);
    const int64_t numKvHeads = kv.GetDim(1);

    const gert::StorageShape *rotShape = context->GetInputShape(1);
    const gert::StorageShape *qjlShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, rotShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);
    const auto &rot = rotShape->GetStorageShape();
    const auto &qjl = qjlShape->GetStorageShape();
    if (rot.GetDimNum() != 2 || rot.GetDim(0) != kHeadDim || rot.GetDim(1) != kHeadDim) {
        return ge::GRAPH_FAILED;
    }
    if (qjl.GetDimNum() != 2 || qjl.GetDim(0) != kHeadDim || qjl.GetDim(1) != kHeadDim) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 0U) {
        aicNum = 20U;
    }
    const uint32_t vecCoreNum = aicNum * 2U;
    const int64_t totalVectors = numTokens * numKvHeads;
    if (totalVectors <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalRows = static_cast<uint32_t>(totalVectors);
    const uint32_t coreRows = (totalRows + vecCoreNum - 1U) / vecCoreNum;
    const uint32_t tileRows = (coreRows > kGemmTile) ? kGemmTile : coreRows;

    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetShape(tileRows, static_cast<uint32_t>(kHeadDim), static_cast<uint32_t>(kHeadDim));
    mm.SetOrgShape(tileRows, static_cast<uint32_t>(kHeadDim), static_cast<uint32_t>(kHeadDim));
    mm.SetBias(false);
    mm.SetBufferSpace(-1, -1, -1);

    KvCacheTurboQuantTilingData tiling;
    if (mm.GetTiling(tiling.cubeTiling) != 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_numTokens(numTokens);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_headDim(kHeadDim);
    tiling.set_mseBits(mseBits);
    tiling.set_qjlDim(kHeadDim);
    tiling.set_idxBytes((kHeadDim * mseBits + 7) / 8);
    tiling.set_qjlBytes((kHeadDim + 7) / 8);
    tiling.set_coreNum(aicNum);
    tiling.set_totalRows(totalRows);
    tiling.set_coreRows(coreRows);
    tiling.set_vecCoreNum(vecCoreNum);

    context->SetBlockDim(aicNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    const size_t rowMatBytes = static_cast<size_t>(totalRows) * static_cast<size_t>(kHeadDim) * sizeof(float);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = kMatmulSysWorkspace + 2UL * rowMatBytes + 16UL * 1024UL * 1024UL;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheTurboQuant).Tiling(TilingKvCacheTurboQuant);

} // namespace optiling
