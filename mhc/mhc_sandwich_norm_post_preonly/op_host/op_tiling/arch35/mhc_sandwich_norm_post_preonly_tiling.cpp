/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "mhc_sandwich_norm_post_preonly_tiling.h"
#include <algorithm>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {

// Host-side constants (mirrors op_kernel/mhc_sandwich_norm_post_preonly_common.h)
constexpr uint32_t HOST_D_MAX             = 5120;
constexpr uint32_t HOST_N_MAX             = 4;
constexpr uint32_t HOST_D_TILE_MAX        = 2560;
constexpr uint32_t HOST_D_TILE_ALIGN_ELEM = 128;
constexpr uint32_t HOST_DUAL_CORE_COUNT   = 2;
constexpr uint32_t HOST_MAX_TOKENS        = 1024 * 1024;    // B*S 上限
constexpr uint32_t HOST_WS_HIN_BASE       = 16;    // workspace pair header (sync blocks)
constexpr uint32_t HOST_WS_SYNC_BLOCK     = 8;     // floats per sync block
constexpr uint32_t HOST_WS_NUM_EVENTS     = 1;     // sense-reversing barrier: 1 flag per core
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr size_t   HOST_MIN_WORKSPACE     = 64;    // 最小 workspace 字节数

static ge::graphStatus TilingMhcSandwichNormPostPreonly(gert::TilingContext *context)
{
    // --- Platform info ---
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    if (coreNumAiv == 0) coreNumAiv = 1;

    // --- Get shapes (support both [B,S,D]/[T,D] and [B,S,N,D]/[T,N,D]) ---
    auto h_out_shape = context->GetInputShape(0);  // [B,S,D] or [T,D]
    auto res_shape = context->GetInputShape(1);     // [B,S,N,D] or [T,N,D]
    auto h_post_shape = context->GetInputShape(2);  // [B,S,N] or [T,N]
    auto h_res_shape = context->GetInputShape(3);   // [B,S,N,N] or [T,N,N]
    auto phi_shape = context->GetInputShape(4);     // [N, N*D]
    if (h_out_shape == nullptr || res_shape == nullptr || h_post_shape == nullptr ||
        h_res_shape == nullptr || phi_shape == nullptr) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "GetInputShape is return nullptr");
        return ge::GRAPH_FAILED;
    }

    const auto &hs = h_out_shape->GetStorageShape();
    const auto &rs = res_shape->GetStorageShape();
    const auto &hps = h_post_shape->GetStorageShape();
    const auto &hrs = h_res_shape->GetStorageShape();
    const auto &ps = phi_shape->GetStorageShape();

    // --- Format consistency: residual.ndim must be h_out.ndim + 1 ---
    size_t hDimNum = hs.GetDimNum();
    size_t rDimNum = rs.GetDimNum();
    if (hDimNum != 2U && hDimNum != 3U) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "h_out must be 2D [T,D] or 3D [B,S,D], got %zuD", hDimNum);
        return ge::GRAPH_FAILED;
    }
    if (rDimNum != hDimNum + 1U) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "residual.ndim (%zu) must be h_out.ndim (%zu) + 1", rDimNum, hDimNum);
        return ge::GRAPH_FAILED;
    }
    // h_post is 2D [T,N] / 3D [B,S,N]; h_res is 3D [T,N,N] / 4D [B,S,N,N]
    size_t hpDimNum = hps.GetDimNum();
    size_t hrDimNum = hrs.GetDimNum();
    if (hpDimNum != 2U && hpDimNum != 3U) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "h_post must be 2D [T,N] or 3D [B,S,N], got %zuD", hpDimNum);
        return ge::GRAPH_FAILED;
    }
    if (hrDimNum != 3U && hrDimNum != 4U) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "h_res must be 3D [T,N,N] or 4D [B,S,N,N], got %zuD", hrDimNum);
        return ge::GRAPH_FAILED;
    }

    uint32_t totalTokens;
    uint32_t D;
    uint32_t N;
    if (hDimNum == 2U) {
        // [T, D] format
        totalTokens = static_cast<uint32_t>(hs.GetDim(0));
        D = static_cast<uint32_t>(hs.GetDim(1));
        N = static_cast<uint32_t>(rs.GetDim(1));  // [T, N, D]
    } else {
        // [B, S, D] format
        uint32_t B = static_cast<uint32_t>(hs.GetDim(0));
        uint32_t S = static_cast<uint32_t>(hs.GetDim(1));
        D = static_cast<uint32_t>(hs.GetDim(NUM_TWO));
        N = static_cast<uint32_t>(rs.GetDim(NUM_TWO));  // [B, S, N, D]
        totalTokens = B * S;
    }

    if (ps.GetDimNum() != 2U) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "phi must be 2D [N, N*D], got %zuD", ps.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    uint32_t phiRows = static_cast<uint32_t>(ps.GetDim(0));  // N (pre-branch only)
    uint32_t phiCols = static_cast<uint32_t>(ps.GetDim(1));  // N*D

    // --- Parameter validation ---
    if (totalTokens == 0 || totalTokens > HOST_MAX_TOKENS) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "totalTokens (B*S) must be in [1, %u], got %u", HOST_MAX_TOKENS, totalTokens);
        return ge::GRAPH_FAILED;
    }
    if (N != HOST_N_MAX) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "headCount (N) must be %u, got %u", HOST_N_MAX, N);
        return ge::GRAPH_FAILED;
    }
    if (D != HOST_D_TILE_MAX && D != HOST_D_MAX) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "embedDim (D) must be %u or %u, got %u", HOST_D_TILE_MAX, HOST_D_MAX, D);
        return ge::GRAPH_FAILED;
    }
    if (phiRows != N || phiCols != N * D) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "phi shape invalid: expected [%u, %u], got [%u, %u]",
                  N, N * D, phiRows, phiCols);
        return ge::GRAPH_FAILED;
    }

    // --- Attributes (all optional with defaults) ---
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    float normEps = 1e-6F;
    float mhcPreNormEps = 1e-6F;
    float hcEps = 1e-6F;
    bool returnHInF32 = false;
    if (attrs != nullptr) {
        // With op_host_aclnn (auto-generated aclnn wrapper), attrs are stored as float
        // matching the .Float() declaration in _def.cpp.
        const float *eps0 = attrs->GetAttrPointer<float>(0);
        if (eps0 != nullptr) {
            normEps = *eps0;
        }
        const float *preEps = attrs->GetAttrPointer<float>(1);
        if (preEps != nullptr) {
            mhcPreNormEps = *preEps;
        }
        const float *eps1 = attrs->GetAttrPointer<float>(2);
        if (eps1 != nullptr) {
            hcEps = *eps1;
        }

        const bool *returnHInF32Ptr = attrs->GetBool(3);
        if (returnHInF32Ptr != nullptr) {
            returnHInF32 = *returnHInF32Ptr;
        }
    }

    // --- Validate eps ---
    if (normEps < 0.0F) {
        OP_LOGE("MhcSandwichNormPostPreonly", "norm_eps must be >= 0, got %e", normEps);
        return ge::GRAPH_FAILED;
    }
    if (hcEps < 0.0F) {
        OP_LOGE("MhcSandwichNormPostPreonly", "hc_eps must be >= 0, got %e", hcEps);
        return ge::GRAPH_FAILED;
    }
    if (mhcPreNormEps < 0.0F) {
        OP_LOGE("MhcSandwichNormPostPreonly", "mhc_pre_norm_eps must be >= 0, got %e", mhcPreNormEps);
        return ge::GRAPH_FAILED;
    }

    // --- Detect gamma_2 optional input (index 9) ---
    // If gamma_2 is provided, enable RMSNorm_mid; otherwise skip it
    auto gamma2_shape = context->GetOptionalInputShape(9);
    bool useRmsMid = (gamma2_shape != nullptr);

    // --- D-axis tiling (must compute BEFORE core mapping) ---
    const uint32_t alignElem = HOST_D_TILE_ALIGN_ELEM;
    uint32_t dimTile = D;
    if (dimTile > HOST_D_TILE_MAX) {
        dimTile = HOST_D_TILE_MAX;
    }
    dimTile = (dimTile / alignElem) * alignElem;
    if (dimTile == 0) dimTile = alignElem;

    uint32_t lastDimTile = D % dimTile;
    uint32_t dimLoop = D / dimTile + (lastDimTile ? 1 : 0);

    // --- [R9] Core mapping: dual-core N-split for dimLoop<=2 ---
    uint32_t coresPerToken = 1;
    if (N >= HOST_DUAL_CORE_COUNT && dimLoop <= HOST_DUAL_CORE_COUNT 
        && totalTokens <= (coreNumAiv / HOST_DUAL_CORE_COUNT)) {
        coresPerToken = HOST_DUAL_CORE_COUNT;
    }
    uint32_t myHeadCount = N / coresPerToken;

    uint32_t logicalTokens = totalTokens * coresPerToken;
    uint32_t usedCoreNum = std::min(logicalTokens, coreNumAiv);

    // Ensure core count is multiple of coresPerToken
    if (coresPerToken == HOST_DUAL_CORE_COUNT) {
        usedCoreNum = (usedCoreNum / HOST_DUAL_CORE_COUNT) * HOST_DUAL_CORE_COUNT;
        if (usedCoreNum == 0) usedCoreNum = HOST_DUAL_CORE_COUNT;
    }

    // Token distribution
    uint32_t numUnits = usedCoreNum / coresPerToken;
    if (numUnits == 0) {
        numUnits = 1;
        usedCoreNum = coresPerToken;
    }

    uint32_t rowsPerCore = totalTokens / numUnits;
    uint32_t remainTokens = totalTokens % numUnits;
    uint32_t rowsPerCoreSp = rowsPerCore + 1;
    uint32_t blockPivot = remainTokens;

    // --- Fill tiling data ---
    MhcSandwichNormPostPreonlyTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_headCount(N);
    tiling.set_embedDim(D);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerCoreSp(rowsPerCoreSp);
    tiling.set_blockPivot(blockPivot);
    tiling.set_dimTile(dimTile);
    tiling.set_dimLoop(dimLoop);
    tiling.set_lastDimTile(lastDimTile);
    tiling.set_phiRows(phiRows);
    tiling.set_phiCols(phiCols);
    tiling.set_normEps(normEps);
    tiling.set_mhcPreNormEps(mhcPreNormEps);
    tiling.set_hcEps(hcEps);
    tiling.set_invD(1.0F / static_cast<float>(D));
    tiling.set_invK(1.0F / static_cast<float>(N * D));
    tiling.set_useRmsMid(useRmsMid ? 1U : 0U);
    tiling.set_coresPerToken(coresPerToken);
    tiling.set_myHeadCount(myHeadCount);
    tiling.set_maxTokensPerPair(rowsPerCoreSp);  // max tokens any pair processes
    tiling.set_returnHInF32(static_cast<int8_t>(returnHInF32)); 

    // --- Workspace sizing (also determines sync region start offset) ---
    // Dual-core: per pair = 16 + N*D floats (sync blocks + x2 data from both cores)
    // Single-core: per core = (N+1)*D floats
    size_t pairWsFloats = HOST_WS_HIN_BASE + (size_t)HOST_N_MAX * HOST_D_MAX;
    size_t singleWsFloats = (size_t)(HOST_N_MAX + 1) * HOST_D_MAX;
    size_t wsPerUnit = std::max(pairWsFloats, singleWsFloats);
    size_t dataWsBytes = (size_t)coreNumAiv * wsPerUnit * sizeof(float);
    // Cross-core sync region comes right after the data region
    size_t syncWsBytes = (size_t)coreNumAiv * HOST_WS_SYNC_BLOCK * HOST_WS_NUM_EVENTS * sizeof(int32_t);
    size_t sysWsBytes = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t totalWsBytes = sysWsBytes + dataWsBytes + syncWsBytes;
    if (totalWsBytes < HOST_MIN_WORKSPACE) {
        totalWsBytes = HOST_MIN_WORKSPACE;
    }
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE("MhcSandwichNormPostPreonly",
                  "GetWorkspaceSizes is nullptr");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = totalWsBytes;

    // Tell kernel where the sync region starts (float offset from workspace base).
    // Matches dataWsBytes/sizeof(float) exactly so kernel writes sync flags
    // into the region host reserved — not into the middle of the data region.
    // Previously kernel computed this itself from runtime `usedCoreNum * wsPerUnit_runtime`,
    // which only worked because usedCoreNum << coreNumAiv left unused data slots.
    tiling.set_syncGmOffsetFloats(static_cast<uint32_t>(dataWsBytes / sizeof(float)));

    // --- Emit tiling buffer ---
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    OP_LOGD(context->GetNodeName(), "returnHInF32:  %s", (returnHInF32 ? "true" : "false"));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMhcSandwichNormPostPreonly(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcSandwichNormPostPreonly)
    .Tiling(TilingMhcSandwichNormPostPreonly)
    .TilingParse<MhcSandwichNormPostPreonlyCompileInfo>(TilingPrepareForMhcSandwichNormPostPreonly);
}  // namespace optiling
