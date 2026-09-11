/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lrn_tiling.cpp
 * \brief LRN host tiling: shape/attr validation, blockDim (pure-vector
 *        grid-stride launch = min(aivNum, total_tasks), the Stage-4 kernel's
 *        launch geometry), tiling data fill, tiling key and the system
 *        workspace sizing. Mirrors the validated PyPTO-Pro Stage-4 launch
 *        config (custom/LRN/test_LRN.py).
 */

#include <algorithm>
#include <string>

#include "lrn_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/math_util.h"

using namespace ge;

namespace optiling {

namespace {
// Shared kernel constants (must match op_kernel/lrn.py). The tile
// geometry rides the TilingKey (LRNTilingKey in op_kernel/lrn.py): the
// host mirrors the candidate sets below, picks one per launch, packs it
// with GET_TPL_TILING_KEY and the build enumerates one kernel instance
// per (key, datatype) combination with tile shapes / UB address map
// folded from c_ext = CChunk + 2*RTier. The feasibility check below and
// the baked address map are therefore same-source same-tier.
constexpr int64_t LRN_R_MAX = 48;        // depth_radius contract upper bound
// hw_tile: fixed at the a5 VF register width — 256B per vector register,
// 2B per fp16 element -> 128 lanes per tile row. The kernel cannot query
// the vreg width itself, so this literal mirrors the HwTile TilingKey
// candidate in op_kernel/lrn.py — same-literal contract as LRN_C_CHUNK
// below. Porting to another platform = change BOTH sides together (or
// reintroduce a runtime Ops::Base::GetVRegSize query then); the packed
// -key membership validation fail-closes any one-sided drift.
constexpr int64_t LRN_HW_TILE = 128;
// c_chunk: CChunk key candidate (output-channel rows per task) — a
// scheduling choice, NOT a UB maximization: more (smaller) tasks split
// better across cores (multi-core first) instead of filling the UB per
// task.
constexpr int64_t LRN_C_CHUNK = 64;
// RTier key candidates: the smallest candidate >= the ACTUAL
// depth_radius selects the instance, so the baked x-slot reservation
// (c_chunk + 2*tier rows) shrinks with the real window instead of the
// worst case (2*LRN_R_MAX). Mirrors RTier in op_kernel/lrn.py.
constexpr int64_t LRN_R_TIERS[] = {0, 1, 2, 4, 8, 16, 32, 48};
constexpr int64_t LRN_R_TIER_COUNT = 8;
constexpr uint32_t LRN_IDX_OF_X = 0;
// Element widths of the two dtype specializations (bytes).
constexpr int64_t LRN_FP16_BYTES_PER_ELEM = 2;
constexpr int64_t LRN_FP32_BYTES_PER_ELEM = 4;
// UB span factors per dtype combo (DESIGN.md §3 layouts):
//   fp32 combo: x32 rotating x2 + y32 rotating x2, all 4 B/elem;
//   fp16 combo: x16 rotating x2 + y16 rotating x2 at 2 B/elem plus one
//               resident y32 staging slot at 4 B/elem.
constexpr int64_t LRN_X_TILE_SLOTS = 2;
constexpr int64_t LRN_Y_ROT_SLOTS = 2;
constexpr int64_t LRN_Y32_RESIDENT_SLOTS = 1;
constexpr int64_t LRN_HALO_SIDES = 2;     // halo rows above + below the chunk
constexpr int64_t LRN_X_RANK_NCHW = 4;    // N, C, H, W
// Attr indices — fixed declaration order in lrn_def.cpp.
constexpr uint32_t LRN_ATTR_DEPTH_RADIUS = 0;
constexpr uint32_t LRN_ATTR_BIAS = 1;
constexpr uint32_t LRN_ATTR_ALPHA = 2;
constexpr uint32_t LRN_ATTR_BETA = 3;
constexpr uint32_t LRN_ATTR_NORM_REGION = 4;
} // namespace

static ge::graphStatus LRNTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of AIV core obtained is 0."),
                return ge::GRAPH_PARAM_INVALID);

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGE(context->GetNodeName(), "SOC Version is not support.");
        return ge::GRAPH_PARAM_INVALID;
    }

    // ---- tile values (same-literal mirrors of the TilingKey candidates;
    //      see the constants block for provenance) ----
    // hw_tile is fixed at LRN_HW_TILE. c_chunk is the design-frozen scheduling constant. The
    // packed-key membership validation below (GET_TPL_TILING_KEY ->
    // INVALID_TILING_KEY) doubles as the cross-side consistency guard: a
    // one-sided edit of either literal fails the launch closed instead of
    // selecting a mismatched kernel instance.
    const int64_t hwTile = LRN_HW_TILE;   // fp16 lanes per tile row
    // c_chunk: its UB feasibility (per tier and dtype) is checked below.
    const int64_t cChunk = LRN_C_CHUNK;

    // Runtime UB size (for the feasibility check below; not a table).
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE(context->GetNodeName(), "UB size obtained from platform is 0."),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    // ---- shape: x must be 4-D NCHW (no per-axis engineering-envelope bounds:
    //      the grid-stride task decomposition is shape-agnostic) ----
    auto xShape = context->GetInputShape(LRN_IDX_OF_X);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE(context->GetNodeName(), "x shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != LRN_X_RANK_NCHW,
                OP_LOGE(context->GetNodeName(), "x must be 4D NCHW, but got %zuD.",
                        xShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);

    auto xDesc = context->GetInputDesc(LRN_IDX_OF_X);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "x desc is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "x dtype must be float16 or float32, but got %d.",
                        static_cast<int32_t>(xDtype)),
                return ge::GRAPH_PARAM_INVALID);

    const int64_t nSize = xShape->GetStorageShape().GetDim(0);
    const int64_t cSize = xShape->GetStorageShape().GetDim(1);
    const int64_t hSize = xShape->GetStorageShape().GetDim(2);
    const int64_t wSize = xShape->GetStorageShape().GetDim(3);
    const int64_t hwSize = hSize * wSize;

    // ---- attrs (declaration order: depth_radius, bias, alpha, beta, norm_region) ----
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "GetAttrs is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    const int64_t *depthRadiusPtr = attrs->GetAttrPointer<int64_t>(LRN_ATTR_DEPTH_RADIUS);
    OP_CHECK_IF(depthRadiusPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "depth_radius attr is required."),
                return ge::GRAPH_PARAM_INVALID);
    const int64_t depthRadius = *depthRadiusPtr;
    OP_CHECK_IF(depthRadius < 0 || depthRadius > LRN_R_MAX,
                OP_LOGE(context->GetNodeName(), "depth_radius must be in [0, %ld], but got %ld.",
                        LRN_R_MAX, depthRadius),
                return ge::GRAPH_PARAM_INVALID);

    const float *biasPtr = attrs->GetAttrPointer<float>(LRN_ATTR_BIAS);
    OP_CHECK_IF(biasPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "bias attr is required."),
                return ge::GRAPH_PARAM_INVALID);
    const float bias = *biasPtr;
    OP_CHECK_IF(bias <= 0.0f,
                OP_LOGE(context->GetNodeName(), "bias must be > 0 (norm > 0), but got %f.", bias),
                return ge::GRAPH_PARAM_INVALID);

    const float *alphaPtr = attrs->GetAttrPointer<float>(LRN_ATTR_ALPHA);
    OP_CHECK_IF(alphaPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "alpha attr is required."),
                return ge::GRAPH_PARAM_INVALID);
    const float alpha = *alphaPtr;
    OP_CHECK_IF(alpha <= 0.0f,
                OP_LOGE(context->GetNodeName(), "alpha must be > 0, but got %f.", alpha),
                return ge::GRAPH_PARAM_INVALID);

    const float *betaPtr = attrs->GetAttrPointer<float>(LRN_ATTR_BETA);
    OP_CHECK_IF(betaPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "beta attr is required."),
                return ge::GRAPH_PARAM_INVALID);
    const float beta = *betaPtr;

    const char *normRegionPtr = attrs->GetAttrPointer<char>(LRN_ATTR_NORM_REGION);
    OP_CHECK_IF(normRegionPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "norm_region attr is required."),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(std::string(normRegionPtr) != "ACROSS_CHANNELS",
                OP_LOGE(context->GetNodeName(), "norm_region only supports ACROSS_CHANNELS, but got %s.",
                        normRegionPtr),
                return ge::GRAPH_PARAM_INVALID);

    // ---- RTier selection + per-(tier, dtype) exact UB feasibility ----
    // tier = smallest RTier candidate >= the ACTUAL depth_radius. The
    // selected key instance bakes c_ext = cChunk + 2*tier into its tile
    // shapes and UB address map, so the check below (the exact span of
    // that instance's address table for the active dtype) is necessary
    // AND sufficient — same-source same-tier as the baked map. This
    // resolves the former packed-model gap (dev log Q14: the old
    // actual-r check modeled shrunk slots while the baked map reserved
    // the worst-case halo, so it was necessary-only).
    //   fp32 combo span = (2*c_ext + 2*c_chunk) * hw_tile * 4
    //   fp16 combo span = (2*c_ext*2 + 1*c_chunk*4 + 2*c_chunk*2) * hw_tile
    // On a5 the worst tier (48) spans 229376B (fp32) / 147456B (fp16)
    // <= 253952B, so every tier is admitted and this check never fires.
    int64_t rTier = -1;
    for (int64_t i = 0; i < LRN_R_TIER_COUNT; i++) {
        if (LRN_R_TIERS[i] >= depthRadius) {
            rTier = LRN_R_TIERS[i];
            break;
        }
    }
    // depth_radius is validated in [0, LRN_R_MAX] and the ladder ends at
    // LRN_R_MAX, so a tier always exists; the guard stays fail-closed.
    OP_CHECK_IF(rTier < 0,
                OP_LOGE(context->GetNodeName(), "no RTier candidate covers depth_radius %ld.",
                        depthRadius),
                return ge::GRAPH_PARAM_INVALID);
    const int64_t cExt = cChunk + LRN_HALO_SIDES * rTier;
    const int64_t ubNeed = (xDtype == ge::DT_FLOAT)
        ? (LRN_X_TILE_SLOTS * cExt + LRN_Y_ROT_SLOTS * cChunk) * hwTile * LRN_FP32_BYTES_PER_ELEM
        : hwTile * (LRN_X_TILE_SLOTS * cExt * LRN_FP16_BYTES_PER_ELEM
                    + LRN_Y32_RESIDENT_SLOTS * cChunk * LRN_FP32_BYTES_PER_ELEM
                    + LRN_Y_ROT_SLOTS * cChunk * LRN_FP16_BYTES_PER_ELEM);
    OP_CHECK_IF(static_cast<int64_t>(ubSize) < ubNeed,
                OP_LOGE(context->GetNodeName(),
                        "tile instance (cChunk=%ld, hwTile=%ld, rTier=%ld, cExt=%ld) for dtype %d "
                        "needs %ld B of UB, exceeding the platform UB size %lu B.",
                        cChunk, hwTile, rTier, cExt, static_cast<int32_t>(xDtype), ubNeed,
                        static_cast<unsigned long>(ubSize)),
                return ge::GRAPH_PARAM_INVALID);

    // ---- tiling data (LRNTiling comes from the pypto kernel via force-include) ----
    LRNTiling *tilingData = context->GetTilingData<LRNTiling>();
    tilingData->n = nSize;
    tilingData->c = cSize;
    tilingData->hw = hwSize;
    tilingData->r = depthRadius;
    tilingData->bias = bias;
    tilingData->alpha = alpha;
    tilingData->beta = beta;

    // ---- tiling key: tile-template selection (HwTile, CChunk, RTier) ----
    // GET_TPL_TILING_KEY takes the ACTUAL candidate values in TilingKey
    // field declaration order and encodes the candidate indices into the
    // packed 64-bit key. With hwTile/cChunk fixed on both sides, its
    // membership validation is the cross-side drift guard (a one-sided
    // literal change -> INVALID_TILING_KEY, rejected fail-closed here);
    // rTier is the only value actually selected at runtime. The build has
    // enumerated one kernel instance per (key, datatype) combination, so
    // the packed key selects the instance whose tile shapes / UB address
    // map match these values.
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(hwTile),
                                                  static_cast<uint64_t>(cChunk),
                                                  static_cast<uint64_t>(rTier));
    OP_CHECK_IF(tilingKey == INVALID_TILING_KEY,
                OP_LOGE(context->GetNodeName(),
                        "tile values (hwTile=%ld, cChunk=%ld, rTier=%ld) are not a valid TilingKey "
                        "combination; extend the candidate set in the kernel key schema "
                        "(op_kernel/lrn.py LRNTilingKey) and rebuild for this platform.",
                        hwTile, cChunk, rTier),
                return ge::GRAPH_PARAM_INVALID);
    context->SetTilingKey(tilingKey);

    // ---- workspace: system reserve only (LRN has no custom workspace) ----
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // ---- blockDim: pure-vector grid-stride launch ----
    // The kernel contains only section_vector (no cube API), so cube cores
    // must not enter the launch config (aic = 0 in CalcTschBlockDim). The
    // kernel walks the flattened task grid N * ceil(C/c_chunk) *
    // ceil(HW/hw_tile) — computed from the same key-frozen values gated
    // above — with grid-stride pl.range(core_id, total_tasks, num_cores),
    // so any blockDim is safe: a zero / over-sized launch width just means
    // the grid-stride walk iterates fewer (or zero) times. No explicit
    // empty-tensor branch is needed either — empty tensors are
    // short-circuited by the GE framework after infershape (probe-verified:
    // tiling is never entered), and should any path reach tiling with an
    // empty shape, totalTasks = 0 collapses the kernel walk to a no-op.
    const int64_t cBlks = Ops::Base::CeilDiv(cSize, cChunk);
    const int64_t hwBlks = Ops::Base::CeilDiv(hwSize, hwTile);
    const int64_t totalTasks = nSize * cBlks * hwBlks;
    const uint32_t launchBlocks = static_cast<uint32_t>(
        std::min<int64_t>(totalTasks, static_cast<int64_t>(aivNum)));
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(launchBlocks, 0, aivNum);
    context->SetBlockDim(blockDim);

    OP_LOGI(context->GetNodeName(),
            "LRN tiling completed: blockDim=%u, aivNum=%u, N=%ld, C=%ld, H=%ld, W=%ld, HW=%ld, "
            "totalTasks=%ld, depth_radius=%ld, bias=%f, alpha=%f, beta=%f, workspaceSize=%zu, "
            "hwTile=%ld (fixed), cChunk=%ld (fixed), rTier=%ld, cExt=%ld, ubNeed=%ld, ubSize=%lu.",
            blockDim, aivNum, nSize, cSize, hSize, wSize, hwSize, totalTasks, depthRadius, bias,
            alpha, beta, workspaceSize, hwTile, cChunk, rTier, cExt, ubNeed,
            static_cast<unsigned long>(ubSize));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLRN(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "TilingParseForLRN received null context."),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null."),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<LRNCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LRN)
    .Tiling(LRNTilingFunc)
    .TilingParse<LRNCompileInfo>(TilingParseForLRN);

} // namespace optiling
