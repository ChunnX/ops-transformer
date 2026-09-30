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
 * \brief
 */
#include "kv_cache_turbo_quant_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

using namespace ge;

namespace optiling {

// Cube tiling for the matmul-lib calls: z = x @ (Hh|Hl)^T and p = r @ (Sh|Sl)^T
// per tri-core group, fp16 in / fp32 out, C straight to GM. Mirrors the
// conservative 64-capped tile shape proven by compute_wy on dav m200.
static ge::graphStatus FillCubeTiling(gert::TilingContext* context, TCubeTiling& out)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetOrgShape(64, 128, 128);
    mm.SetShape(64, 128, 128);
    mm.SetBufferSpace(-1, -1, -1);
    if (mm.GetTiling(out) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

constexpr uint32_t KV_INPUT_INDEX = 0;
constexpr uint32_t QJL_INPUT_INDEX = 2;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t KV_DIMS = 3;
constexpr int64_t MATRIX_DIMS = 2;
constexpr int64_t MIN_KV_HEADS = 4;
constexpr int64_t MAX_KV_HEADS = 32;
constexpr int64_t QJL_DIM_ALIGN = 8;
constexpr int64_t CUBE_MIN_VEC = 1024;
// mix kernel limits: 128-row chunks, round-robin over 24 groups, 22 chunks per
// group max (flag-cell capacity) -> totalVec <= 67584
constexpr int64_t CUBE_CHUNK_ROWS = 128;
constexpr int64_t CUBE_MAX_LOCAL_CHUNKS = 22; // per-group flag-cell capacity in the kernel

static ge::graphStatus TilingKvCacheTurboQuant(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* kvShape = context->GetInputShape(KV_INPUT_INDEX);
    if (kvShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& kv = kvShape->GetStorageShape();
    if (kv.GetDimNum() != KV_DIMS || kv.GetDim(2) != HEAD_DIM) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = kv.GetDim(0);
    const int64_t numKvHeads = kv.GetDim(1);
    if (numTokens <= 0 || numKvHeads < MIN_KV_HEADS || numKvHeads > MAX_KV_HEADS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalVec = numTokens * numKvHeads;

    const gert::StorageShape* qjlShape = context->GetInputShape(QJL_INPUT_INDEX);
    if (qjlShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& qjl = qjlShape->GetStorageShape();
    if (qjl.GetDimNum() != MATRIX_DIMS || qjl.GetDim(1) != HEAD_DIM) {
        return ge::GRAPH_FAILED;
    }
    const int64_t qjlDim = qjl.GetDim(0);
    if (qjlDim <= 0 || qjlDim > HEAD_DIM || (qjlDim % QJL_DIM_ALIGN) != 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    const int64_t mseBits = (pBits != nullptr) ? *pBits : 3;
    if (mseBits != 2 && mseBits != 3 && mseBits != 4) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 0) {
        aicNum = 1;
    }
    // per-chip cap: chunkNum must fit (aicNum groups x 22 chunk slots); on a
    // 20-AIC part (910B4) 24x22x128 would overflow the kernel's flag cells
    const int64_t cubeMaxVec = static_cast<int64_t>(aicNum) * CUBE_MAX_LOCAL_CHUNKS * CUBE_CHUNK_ROWS;
    const bool useCube = (totalVec >= CUBE_MIN_VEC) && (totalVec <= cubeMaxVec) && (qjlDim == HEAD_DIM);

    KvCacheTurboQuantTilingData tiling;
    tiling.set_totalVec(totalVec);
    tiling.set_qjlDim(qjlDim);
    tiling.set_mseBits(mseBits);
    if (FillCubeTiling(context, tiling.mmTiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t usedCoreNum = 1;
    // single tiling key 0 with runtime dispatch; always launch the full tri-core
    // grid (out-of-range groups idle out in-kernel). usedCoreNum for key-0 vector
    // path = AIV count (2*aicNum), for the mix path = logical tri-core groups.
    // Measured: fewer cube groups is SLOWER (8 groups -> 1345us vs 24 -> 494us
    // for t2048) - the grid-launch overhead is not block-count dominated and
    // fewer groups overloads the serial AIV vector phase, so use all AICs.
    const uint32_t usedAic = aicNum;
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(2 * usedAic, usedAic, 2 * usedAic));
    usedCoreNum = useCube ? usedAic : (2 * usedAic);
    context->SetTilingKey(0);
    tiling.set_coreNum(usedCoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // NOTE: the MIX-channel wrapper auto-generates matmul::clearWorkspace on AIC
    // blocks, so a non-null workspace is required even for the vector-only path.
    // The MIX runtime (KFC channel) owns the first GetLibApiWorkSpaceSize() bytes
    // of the workspace; the kernel skips them via GetUserWorkspace(). Launching
    // with less than that reserved prefix crashes the runtime MTE before any
    // user code runs (bisected 2026-09-24).
    size_t wsBytes = 32 * 1024;
    if (useCube) {
        // flags 264KB + splits 160KB (H bf16-triple + S fp16-pair) +
        // per-padded-vector: z 512B (fp32) + r_hat 512B (rh|rl fp16) +
        // p1/p2/p3 3x256B (fp16) = 1792B; x is read directly (no xh plane)
        const int64_t wsVec = (totalVec + CUBE_CHUNK_ROWS - 1) / CUBE_CHUNK_ROWS * CUBE_CHUNK_ROWS;
        wsBytes = static_cast<size_t>(424 * 1024) + static_cast<size_t>(wsVec) * 1792 + 8 * 1024;
    }
    workspaces[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) + wsBytes;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheTurboQuant).Tiling(TilingKvCacheTurboQuant);

} // namespace optiling
