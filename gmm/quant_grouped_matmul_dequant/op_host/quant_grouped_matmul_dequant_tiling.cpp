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
 * \file quant_grouped_matmul_dequant_tiling.cpp
 * \brief Unified tiling for the qgmmd kernel.
 *
 *   Weight format auto-detection:
 *     rank 3   : ND   [E, N, K]            (grouped)
 *     rank 4   : ZN   [K/32, N/16, 16, 32] (non-grouped standard)
 *     rank 5   : ZN   [E, K/32, N/16, 16, 32] (grouped standard)
 *     rank 6   : ZN_BLOCKED [E, num_blocks, K/32, blockN1, 16, 32] (grouped N>65280)
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "platform/platform_infos_def.h"
#include "quant_grouped_matmul_dequant_tiling.h"
// Algorithmic constants (candidate arrays, MMAD_K_MAX, BLOCKED_ZN_NG,
// SMALL_M_THRESHOLD, DQ_MAX_EPS, fractal sizes) - SoC-invariant, single
// source. Memory sizes (UB/L0A/L0B/L0C/L1) are runtime-queried via
// _Params and NEVER sourced from _config.h on the host side.
#include "../op_kernel/quant_grouped_matmul_dequant_config.h"

namespace optiling {

static bool AddWorkspace(gert::TilingContext* context, uint64_t workspace)
{
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
    *workspace_size = static_cast<size_t>(workspace);
    return true;
}

// ============================================================
// Shape / attribute validation
// ============================================================

bool QuantMatmulDequantTiling::CheckInOutShapes(gert::TilingContext* context)
{
    uint32_t idx = 0;

    // x : [M, K] fp16 ND
    auto x = context->GetInputShape(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    auto shape = x->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd x shape ndim must be 2, got %zu", shape.GetDimNum()),
                return false);
    _Params.originM = static_cast<uint64_t>(shape.GetDim(0));
    _Params.originK = static_cast<uint64_t>(shape.GetDim(1));
    _Params.fracK   = (_Params.originK + K_FRACTAL_INT8 - 1) / K_FRACTAL_INT8;
    _Params.fracM   = (_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;

    // quantized_weight : auto-detect 3-D ND / 4-D ZN-standard / 5-D ZN-standard / 6-D ZN-blocked
    auto quantized_weight = context->GetInputShape(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantized_weight);
    shape = quantized_weight->GetStorageShape();
    const size_t wRank = shape.GetDimNum();
    _Params.weightBlockN1 = 0;

    if (isGrouped && wRank == 5) {
        // [E, K/32, N/16, 16, 32]
        _Params.weightFormat = QgmmdWeightFormat::ZN_STANDARD;
        _Params.originE = static_cast<uint64_t>(shape.GetDim(0));
        _Params.fracN   = static_cast<uint64_t>(shape.GetDim(2));
        OP_CHECK_IF(static_cast<uint64_t>(shape.GetDim(1)) != _Params.fracK ||
                    shape.GetDim(3) != NM_FRACTAL_INT8 || shape.GetDim(4) != K_FRACTAL_INT8,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd weight 5-D must be [E, K/32, N/16, 16, 32]"),
                    return false);
    } else if (isGrouped && wRank == 6) {
        // [E, num_blocks, K/32, blockN1, 16, 32] - blocked ZN for N > 65280
        _Params.weightFormat = QgmmdWeightFormat::ZN_BLOCKED;
        _Params.originE = static_cast<uint64_t>(shape.GetDim(0));
        uint64_t numBlocks = static_cast<uint64_t>(shape.GetDim(1));
        uint64_t blockN1   = static_cast<uint64_t>(shape.GetDim(3));
        _Params.weightBlockN1 = static_cast<uint32_t>(blockN1);
        _Params.fracN   = numBlocks * blockN1;  // padded count (for validation below)
        OP_CHECK_IF(static_cast<uint64_t>(shape.GetDim(2)) != _Params.fracK ||
                    shape.GetDim(4) != NM_FRACTAL_INT8 || shape.GetDim(5) != K_FRACTAL_INT8,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd weight 6-D must be [E, num_blocks, K/32, blockN1, 16, 32]"),
                    return false);
        OP_CHECK_IF(blockN1 != AscendC::BLOCKED_ZN_NG,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd blocked ZN blockN1 must be %u, got %lu",
                            AscendC::BLOCKED_ZN_NG, blockN1),
                    return false);
    } else if (!isGrouped && wRank == 4) {
        _Params.weightFormat = QgmmdWeightFormat::ZN_NON_GROUPED;
        _Params.originE = 1;
        _Params.fracN   = static_cast<uint64_t>(shape.GetDim(1));
        OP_CHECK_IF(static_cast<uint64_t>(shape.GetDim(0)) != _Params.fracK ||
                    shape.GetDim(2) != NM_FRACTAL_INT8 || shape.GetDim(3) != K_FRACTAL_INT8,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd weight 4-D must be [K/32, N/16, 16, 32]"),
                    return false);
    } else {
        OP_LOGE(context->GetNodeName(),
                "qgmmd weight rank %zu unsupported (expect 4/5 for ZN, 6 for blocked ZN)", wRank);
        return false;
    }

    // weight_scale : grouped [E, N]; non-grouped [N]
    auto weight_scale = context->GetInputShape(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight_scale);
    shape = weight_scale->GetStorageShape();
    if (isGrouped) {
        OP_CHECK_IF(shape.GetDimNum() != 2 || static_cast<uint64_t>(shape.GetDim(0)) != _Params.originE,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd weight_scale grouped must be [E, N]"),
                    return false);
        _Params.originN = static_cast<uint64_t>(shape.GetDim(1));
    } else {
        OP_CHECK_IF(shape.GetDimNum() != 1,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd weight_scale non-grouped must be [N]"),
                    return false);
        _Params.originN = static_cast<uint64_t>(shape.GetDim(0));
    }
    OP_CHECK_IF((_Params.originN + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 > _Params.fracN,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd weight_scale N-dim vs weight mismatch"),
                return false);
    // Compact blocked-ZN flag. For ZN_BLOCKED, fracN defaults to PADDED
    // (num_blocks * blockN1) to match the ND->TransData output buffer
    // (zero-padded to multiples of 65280). For COMPACT uploads (E*K*N
    // bytes, last block has actualBlockNG < blockN1) set env
    // QGMMD_WEIGHT_COMPACT=1; tiling then sets fracN = compact count so
    // the kernel reads weight with the correct partial-block stride.
    if (_Params.weightFormat == QgmmdWeightFormat::ZN_BLOCKED) {
        const char *compactEnv = getenv("QGMMD_WEIGHT_COMPACT");
        if (compactEnv != nullptr && compactEnv[0] == '1') {
            _Params.fracN = (_Params.originN + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        }
    }

    // Detect weight_scale dtype -> int64 (VDEQ16 direct) vs fp32 (needs V-pipe encode).
    auto scale_desc = context->GetInputDesc(idx - 1);
    _Params.scaleIsInt64 = false;
    if (scale_desc != nullptr) {
        _Params.scaleIsInt64 = (scale_desc->GetDataType() == ge::DT_INT64);
    }

    if (isGrouped) {
        auto group_list = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, group_list);
        shape = group_list->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<uint64_t>(shape.GetDim(0)) != _Params.originE,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd group_list must be [E]"),
                    return false);
    }

    // bias : currently unsupported
    auto bias = context->GetOptionalInputShape(idx++);
    OP_CHECK_IF(bias != nullptr,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd bias not supported"),
                return false);
    _Params.hasBias = false;

    // x_scale : optional; absence => dynamic quant
    auto x_scale = context->GetOptionalInputShape(idx++);
    _Params.xScaleIsHalf = false;
    if (x_scale == nullptr) {
        OP_CHECK_IF(!_Params.perToken,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd dynamic-quant requires pertoken mode"),
                    return false);
        _Params.dynamicQuant = true;
    } else {
        _Params.dynamicQuant = false;
        auto xs_desc = context->GetOptionalInputDesc(idx - 1);
        if (xs_desc != nullptr) {
            _Params.xScaleIsHalf = (xs_desc->GetDataType() == ge::DT_FLOAT16);
        }
    }

    // x_offset : unsupported
    auto x_offset = context->GetOptionalInputShape(idx++);
    OP_CHECK_IF(x_offset != nullptr,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd x_offset not supported"),
                return false);

    // smooth_scale : optional fp16 [K]
    auto smooth_scale = context->GetOptionalInputShape(idx++);
    _Params.smoothScale = false;
    if (smooth_scale != nullptr) {
        shape = smooth_scale->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1 ||
                    static_cast<uint64_t>(shape.GetDim(0)) != _Params.originK,
                    OP_LOGE(context->GetNodeName(),
                            "qgmmd smooth_scale must be [K]"),
                    return false);
        _Params.smoothScale = true;
    }

    // y : [M, N] fp16 ND
    auto y = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y);
    shape = y->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2 ||
                static_cast<uint64_t>(shape.GetDim(0)) != _Params.originM ||
                static_cast<uint64_t>(shape.GetDim(1)) != _Params.originN,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd y shape must be [M, N]"),
                return false);

    return true;
}

bool QuantMatmulDequantTiling::GetCheckAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const std::string xQuantMode = static_cast<std::string>(attrs->GetStr(0));
    const bool *transposeWeight = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_IF(xQuantMode != "pertoken" && xQuantMode != "pertensor",
                OP_LOGE(context->GetNodeName(),
                        "qgmmd x_quant_mode must be pertoken or pertensor"),
                return false);
    OP_CHECK_IF(*transposeWeight == false,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd transpose_weight must be true"),
                return false);
    _Params.perToken = (xQuantMode == "pertoken");
    return true;
}

bool QuantMatmulDequantTiling::GetPlatformInfo(gert::TilingContext* context)
{
    auto compileInfoPtr = reinterpret_cast<const QuantMatmulDequantCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    _Params.coreNum  = compileInfoPtr->coreNum;
    _Params.ubSize   = compileInfoPtr->ubSize;
    _Params.l1Size   = compileInfoPtr->l1Size;
    _Params.l0ASize  = compileInfoPtr->l0ASize;
    _Params.l0BSize  = compileInfoPtr->l0BSize;
    _Params.l0CSize  = compileInfoPtr->l0CSize;
    return true;
}

// ============================================================
// Unified tiling algorithm: enumerate (Nb, Mb) candidates, prune on
// L0A/L0B/L1/UB budgets for the given shape, score the survivors.
// ============================================================

static uint64_t BuildUbKMask(uint64_t originK, uint64_t kAligned32)
{
    uint64_t mask = 0;
    uint64_t kTail = kAligned32 - originK;
    for (uint32_t i = 0; i < K_FRACTAL_INT8; ++i) {
        mask <<= 1;
        if (i < kTail) {
            mask |= 1;
        }
    }
    return mask;
}

static void PickCoreGrid(uint64_t fracM, uint64_t fracN, uint64_t originE,
                         uint32_t &MCoreNum, uint32_t &NCoreNum)
{
    // E>1 with fracM<aicNum splits N over all cores while every core
    // duplicates all iters, so force M-only split there. Only E==1 with
    // fracM==1 gains from N-only; the kernel three-way handles it without
    // changing the picker.
    if (originE > 1u || fracM >= static_cast<uint64_t>(NUM_OF_AICORE)) {
        NCoreNum = 1u;
        MCoreNum = static_cast<uint32_t>(NUM_OF_AICORE);
        return;
    }
    // Proxy: min_i{fracN*2^(3-i) + fracM*2^i}.
    uint32_t best = 0;
    uint64_t minCost = (fracN << 3) + fracM;
    for (uint32_t i = 1; i < 4; ++i) {
        uint64_t cost = (fracN << (3 - i)) + (fracM << i);
        if (cost < minCost) {
            best = i;
            minCost = cost;
        }
    }
    NCoreNum = 1u << best;
    MCoreNum = static_cast<uint32_t>(NUM_OF_AICORE) / NCoreNum;
}

bool QuantMatmulDequantTiling::GetTilingData(gert::TilingContext* context)
{
    _Params.originKAligned32  = (_Params.originK + K_FRACTAL_INT8 - 1) / K_FRACTAL_INT8 * K_FRACTAL_INT8;
    _Params.originKAligned512 = (_Params.originK + L0_ADDR_ALIGN - 1) / L0_ADDR_ALIGN * L0_ADDR_ALIGN;
    _Params.ubKMask = BuildUbKMask(_Params.originK, _Params.originKAligned32);

    OP_CHECK_IF(_Params.originK % FLOAT16_PERBLOCK != 0,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd K must be multiple of 16"),
                return false);
    OP_CHECK_IF(_Params.originN % FLOAT16_PERBLOCK != 0,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd N must be multiple of 16"),
                return false);
    OP_CHECK_IF(_Params.originM == 0,
                OP_LOGE(context->GetNodeName(),
                        "qgmmd M must not be 0"),
                return false);

    // Cost-model candidate enumeration: enumerate (Nb, Mb) from the
    // candidate arrays, prune on L0/L1/UB fit using *runtime-queried*
    // memory sizes (_Params.ubSize / l0ASize / l0BSize / l0CSize /
    // l1Size), score by per-core weight bytes. Tie-break: larger Mb, then
    // larger Nb. Budgets come from the platform query so the model serves
    // any SoC the runtime advertises; candidate arrays are SoC-invariant.
    const uint64_t kBytes  = _Params.originKAligned32;
    const uint64_t ubSize  = _Params.ubSize;
    // Halve L0A/L0B for spatial ping-pong: each Mmad's per-kp L0A and L0B
    // occupy ONE half (32 KB) so the next kp's MTE1 LoadData fills the
    // OTHER half concurrent with the running Mmad. Parity-based event IDs
    // (EVENT_ID0/1) keep the two halves' M<->MTE1 chains independent. kp
    // count doubles but per-kp load volume halves and overlaps with Cube.
    const uint64_t l0aTotal = _Params.l0ASize / 2;
    const uint64_t l0bTotal = _Params.l0BSize / 2;
    const uint64_t l0cSize = _Params.l0CSize;
    const uint64_t l1Size  = _Params.l1Size;

    uint32_t bestMb = 16, bestNb = 16;
    uint64_t bestScore = UINT64_MAX;

    // Phase X streams K through fixed-size chunks. int8Scratch is sized
    // MbAligned*chunk (one chunk across all rows), so the slot scales with
    // chunk, NOT kBytes. Chunk cap 2048 keeps Mb>=64 within UB on K>=4096
    // (chunk=4096 + Mb=64 gives int8Scratch=256KB, exceeding UB). Min 128
    // for vector-mask alignment.
    uint32_t phaseXChunkUB = 128;
    while ((phaseXChunkUB << 1u) <= kBytes && phaseXChunkUB < 2048) {
        phaseXChunkUB <<= 1u;
    }

    auto ubFits = [&](uint32_t mb, uint32_t nb) -> bool {
        // Persistent (must match InitUbLayout in op_kernel/.h):
        //   pertoken (Mb*4) + outPing/Pong (2*Mb*Nb*2) + scaleU64 ping/pong
        //   (2*Nb*8). When !ScaleIsU64: scatterIdx (Nb*4). When HasSmooth:
        //   smoothScale (K*2) + smoothFp32 (chunk*4).
        const uint64_t chunk = phaseXChunkUB;
        uint64_t persistent = 2ULL * mb * nb * sizeof(uint16_t)
                            + 2ULL * nb * sizeof(uint64_t)
                            + mb * sizeof(float);
        // Persistent pertokenBcast buffer (Mb*Nb fp16), gated on E==1
        // (dense). For E>1 the runtime allocation is skipped (kernel falls
        // back to per-row Muls), so the picker pays no UB cost there.
        if (_Params.perToken && _Params.originE == 1ULL) {
            persistent += static_cast<uint64_t>(mb) * static_cast<uint64_t>(nb) * sizeof(uint16_t);
        }
        if (!_Params.scaleIsInt64)   persistent += nb * sizeof(uint32_t);   // scatterIdxLT_
        if (_Params.smoothScale)     persistent += kBytes * sizeof(uint16_t)
                                                  + chunk * sizeof(float);  // smoothScale + smoothFp32
        // Phase X region (larger of the two aliased regions). All buffers
        // sized by phaseXChunkUB: int8Scratch holds ONE chunk of int8
        // across mAligned rows (NOT full Mb*kBytes), and Pass 2 of
        // PhaseXDynamicQuant scatters each chunk to xL1 directly so
        // int8Scratch is reused for the next chunk.
        const uint64_t mbAlignedPhX = (mb < NM_FRACTAL_INT8)
                                       ? NM_FRACTAL_INT8 : mb;
        uint64_t phaseX = mbAlignedPhX * chunk                  // int8 scratch (1 chunk)
                        + 2ULL * chunk * sizeof(uint16_t)      // xRaw ping/pong
                        + chunk * sizeof(uint16_t)             // absLT
                        + 2ULL * chunk * sizeof(float)         // xCastFloat + xScaled
                        + chunk * sizeof(int32_t)              // xInt32
                        + chunk * sizeof(uint16_t);            // xHalf
        // Phase D region (aliased with Phase X - take max). VDEQ16 writes
        // mAligned (>= M_FRACTAL) rows into dequantScratch, so size by
        // max(Mb, M_FRACTAL) for SmallM tiles. When perToken is on, the Mul
        // runs fp32xfp32->fp32 so the region needs an fp32 widen of
        // copyOutBuf + fp32 pertoken broadcast + fp32 Mul output.
        //   fp16 dequantScratch (2) + fp16 copyOutBuf (2) + fp32 copyOutFp32
        //   (4) + fp32 pertokenBroadcast (4) + fp32 finalOutFp32 (4)
        //   = 16 bytes per (mbAligned, nb) unit.
        const uint64_t mbAligned = (mb < NM_FRACTAL_INT8)
                                   ? NM_FRACTAL_INT8 : mb;
        // 2-buffer aliased Phase D: just dequantScratch in aliased region;
        // cb-alternating intermediate reuses persistent outPing/Pong.
        // Aliased footprint: 2 B/elem.
        uint64_t phaseD = 1ULL * mbAligned * nb * sizeof(uint16_t);
        if (!_Params.scaleIsInt64)   phaseD += nb * sizeof(float);          // scaleFp32StagingLT_
        uint64_t total  = persistent + (phaseX > phaseD ? phaseX : phaseD) + 2048;
        return total <= ubSize;
    };

    auto l0Fits = [&](uint32_t mb, uint32_t nb, uint32_t kb) -> bool {
        return (static_cast<uint64_t>(mb) * kb <= l0aTotal)
            && (static_cast<uint64_t>(kb) * nb <= l0bTotal)
            && (static_cast<uint64_t>(mb) * nb * sizeof(int32_t) <= l0cSize);
    };

    auto l1Fits = [&](uint32_t mb, uint32_t nb, uint32_t kb) -> bool {
        // xL1 holds a ZZ-fractal mTile; when mb < M_FRACTAL the layout still
        // reserves a full 16-row turn per K-fractal. Budget by mbAligned or
        // the kernel overruns L1.
        const uint64_t mbAligned = (mb < NM_FRACTAL_INT8) ? NM_FRACTAL_INT8 : mb;
        uint64_t xL1 = mbAligned * kBytes;
        uint64_t wL1 = 2ULL * kb * nb;
        return xL1 + wL1 <= l1Size;
    };

    for (uint32_t nbI = 0; nbI < AscendC::NB_CANDIDATE_COUNT; ++nbI) {
        uint32_t nb = AscendC::NB_CANDIDATES[nbI];
        if (nb > _Params.originN) continue;
        for (uint32_t mbI = 0; mbI < AscendC::MB_CANDIDATE_COUNT; ++mbI) {
            uint32_t mb = AscendC::MB_CANDIDATES[mbI];
            // Hard reject Mb < 32. The cost model favours tiny Mb to
            // minimize cube cycles, but Mb < M_FRACTAL=16 has a broken
            // SmallM path and Mb=16 on K=2048 mis-accumulates (per-chunk
            // scatter at mAligned=16 vs kpasses mmad). Restrict to Mb>=32
            // where the Mmad pipeline runs cleanly.
            if (mb < 32U) continue;
            // Per-pass K = min(padded K, L0A total / mb, L0B total / nb,
            // MMAD_K_MAX). L0A/L0B are reused TEMPORALLY across kp's (M_MTE1
            // flag sync), so the FULL 64 KB is available per kp, not half.
            uint64_t kbRaw = std::min<uint64_t>(
                {kBytes, l0aTotal / mb, l0bTotal / nb,
                 static_cast<uint64_t>(AscendC::MMAD_K_MAX)});
            uint32_t kb = static_cast<uint32_t>((kbRaw / K_FRACTAL_INT8) * K_FRACTAL_INT8);
            if (kb == 0) continue;
            if (!l0Fits(mb, nb, kb)) continue;
            if (!l1Fits(mb, nb, kb)) continue;
            if (!ubFits(mb, nb))     continue;

            // Models both pipe regimes: a max-of-pipes path saturates Cube,
            // but small-K sparse-MoE shapes starve Cube and become
            // issue-bound. Two terms cover the issue-bound regime:
            //   1. MTE2 per-burst prologue * burst count per core. Each
            //      cb-kp loads a fresh weight slab, so bursts =
            //      perCoreIters * cbPerIter * kpasses.
            //   2. MTE1 per-LoadData issue (~32 cyc/call): one LoadData(B)
            //      + mTurns=mb/16 LoadData(A) per cb-kp.
            // Score: cube-bound (cube > non-cube-sum) uses cube; otherwise
            // the SUM of non-cube terms.
            //
            // 310P3 platform constants:
            //   MTE2 (DDR -> on-chip): 17 B/cyc, 422-cyc prologue per burst
            //   MTE1 L1 -> L0A:        512 B/cyc + ~32 cyc issue/LoadData
            //   MTE1 L1 -> L0B:        256 B/cyc + ~32 cyc issue/LoadData
            //   Cube (int8 mmad):      ~mb*nb*kb / 2048 cycles per Mmad call
            //   MTE3 (UB -> GM):       17 B/cyc
            const uint64_t weightBytes = static_cast<uint64_t>(_Params.originN)
                                       * static_cast<uint64_t>(kBytes);
            const uint64_t iters       =
                (static_cast<uint64_t>(_Params.originM) + mb - 1) / mb;
            const uint64_t cbPerIter   =
                (static_cast<uint64_t>(_Params.originN) + nb - 1) / nb;
            const uint64_t coreNum     = static_cast<uint64_t>(NUM_OF_AICORE);
            const uint64_t perCoreIters =
                (iters + coreNum - 1) / coreNum;
            const uint64_t kpasses =
                (static_cast<uint64_t>(kBytes) + kb - 1) / kb;
            // MTE2 weight: BW term (amortized across cores; L2-hit assumed
            // best case) + per-burst prologue * actual bursts per core.
            const uint64_t mte2Bursts  = perCoreIters * cbPerIter * kpasses;
            const uint64_t cyclesMte2W =
                ((weightBytes + coreNum - 1) / coreNum) / 17ULL
                + 422ULL * mte2Bursts;
            const uint64_t cyclesMte1A =
                perCoreIters * cbPerIter * kpasses
                * (static_cast<uint64_t>(mb) * kb) / 512ULL;
            const uint64_t cyclesMte1B =
                perCoreIters * cbPerIter * kpasses
                * (static_cast<uint64_t>(kb) * nb) / 256ULL;
            // MTE1 per-LoadData issue: one LoadData(B) call + mTurns
            // LoadData(A) calls per inner (cb-kp). At ~32 cyc per call.
            const uint64_t mte1Calls   =
                perCoreIters * cbPerIter * kpasses
                * (static_cast<uint64_t>(mb) / 16ULL + 1ULL);
            const uint64_t cyclesMte1Issue = 32ULL * mte1Calls;
            const uint64_t cyclesCube  =
                perCoreIters * cbPerIter * kpasses
                * (static_cast<uint64_t>(mb) * nb * kb) / 2048ULL;
            const uint64_t cyclesMte3  =
                perCoreIters * cbPerIter
                * (static_cast<uint64_t>(mb) * nb * 2ULL) / 17ULL;
            const uint64_t cyclesMte1  = cyclesMte1A + cyclesMte1B + cyclesMte1Issue;
            // Regime-aware aggregation: cube cycles >= feeder sum means
            // cube-bound and max(...) is correct. Otherwise issue/BW-bound,
            // feeders run mostly serially, so sum is closer than max.
            const uint64_t feedSum     = cyclesMte2W + cyclesMte1 + cyclesMte3;
            uint64_t score;
            if (cyclesCube >= feedSum) {
                score = cyclesCube;
            } else {
                score = feedSum;
                if (cyclesCube > score) score = cyclesCube;
            }
            if (score < bestScore
                || (score == bestScore && mb > bestMb)
                || (score == bestScore && mb == bestMb && nb > bestNb)) {
                bestScore = score;
                bestMb = mb;
                bestNb = nb;
            }
        }
    }

    _Params.Mb = bestMb;
    _Params.Nb = bestNb;
    {
        uint64_t kbRaw = std::min<uint64_t>(
            {kBytes, l0aTotal / bestMb, l0bTotal / bestNb,
             static_cast<uint64_t>(AscendC::MMAD_K_MAX)});
        _Params.Kb = static_cast<uint32_t>((kbRaw / K_FRACTAL_INT8) * K_FRACTAL_INT8);
    }
    _Params.kPasses = static_cast<uint32_t>(
        (kBytes + _Params.Kb - 1) / _Params.Kb);

    // wChunkKPasses bundles multiple K-passes into one MTE2 weight burst
    // to amortize the 422-cyc DMA prologue. Per-cb W L1 size grows from
    // 2*Kb*Nb to 2*wChunk*Kb*Nb. Pick the largest power-of-2 wChunk that
    // divides kPasses cleanly and keeps xL1 + 2*wChunk*Kb*Nb <= L1.
    // Falls back to 1 when budget is tight or kPasses==1.
    {
        const uint64_t mbAlignedXL1 = (_Params.Mb < NM_FRACTAL_INT8)
                                      ? NM_FRACTAL_INT8 : _Params.Mb;
        const uint64_t xL1Bytes_    = mbAlignedXL1 * kBytes;
        const uint64_t wL1OnePair   = 2ULL * _Params.Kb * _Params.Nb;
        const uint64_t l1Free       = (l1Size > xL1Bytes_) ? (l1Size - xL1Bytes_) : 0ULL;
        uint32_t wChunk = 1;
        // Double while it fits L1 and stays within kPasses. wChunk must
        // divide kPasses so the inner loop is uniform (simpler control
        // flow); a ragged-last-chunk variant could come later if needed.
        for (uint32_t cand = 2; cand <= _Params.kPasses; cand <<= 1u) {
            if ((_Params.kPasses % cand) != 0) break;
            if (cand * wL1OnePair > l1Free) break;
            wChunk = cand;
        }
        _Params.wChunkKPasses = wChunk;
    }

    // Phase X chunk - already computed above for ubFits. Re-emit for
    // kernel consumption (phaseXFullPairs / phaseXTail derived from
    // aligned K bytes).
    _Params.phaseXChunk = phaseXChunkUB;
    _Params.phaseXFullPairs = static_cast<uint32_t>(kBytes / phaseXChunkUB);
    _Params.phaseXTail      = static_cast<uint32_t>(kBytes % phaseXChunkUB);

    _Params.maxOneTurnToken     = _Params.Mb;
    _Params.maxWeightColOneTurn = _Params.Nb;
    _Params.smallM = (_Params.Mb <= AscendC::SMALL_M_THRESHOLD);

    // Core grid: proxy minimizes MTE2 cost.
    uint32_t mCore = 1, nCore = 1;
    PickCoreGrid(_Params.fracM, _Params.fracN, _Params.originE, mCore, nCore);
    _Params.MCoreNum = mCore;
    _Params.NCoreNum = nCore;

    // Iter-level distribution (flat split across cores).
    uint64_t totalIters = 0;
    {
        uint64_t expertIter = (_Params.originM + _Params.Mb - 1) / _Params.Mb;
        totalIters = expertIter;   // Grouped: summed over experts at kernel runtime.
    }
    _Params.perCoreIters = static_cast<uint32_t>(
        (totalIters + _Params.coreNum - 1) / _Params.coreNum);
    _Params.remainderIters = static_cast<uint32_t>(totalIters % _Params.coreNum);

    // Tile-placement L2 hint: set when the full weight fits in L2; the
    // kernel's useL2Hint is stubbed off until a device probe confirms it.
    _Params.l2WeightFits = (static_cast<uint64_t>(_Params.originK) *
                            static_cast<uint64_t>(_Params.originN) <= L2_HEADROOM_BYTES);

    // Cross-iter xRawL1 pre-load. Size = MbAligned * kBytes * sizeof(half)
    // bytes, single-buffered. Enable only when xL1 + 2*wL1 + xRawL1 <= L1
    // total (covers most small shapes at Mb <= 128).
    const uint64_t xRawMbAligned = (_Params.Mb < NM_FRACTAL_INT8)
                                   ? NM_FRACTAL_INT8 : _Params.Mb;
    const uint64_t xL1Bytes      = xRawMbAligned * kBytes;
    const uint64_t wL1Bytes      = 2ULL * _Params.wChunkKPasses * _Params.Kb * _Params.Nb;
    const uint64_t xRawL1Bytes   = xRawMbAligned * kBytes * sizeof(uint16_t);
    _Params.hasXRawL1 =
        (xL1Bytes + wL1Bytes + xRawL1Bytes <= _Params.l1Size);

    _Params.useL2Hint = false;   // stub off until device probe confirms.

    // Enable L0C ping-pong when one cb's int32 output fits half of L0C.
    // L0C = 256 KB; half = 128 KB = 32 K int32 elems. Mb=128,Nb=128 gives
    // 64 KB (fits half); Mb=256,Nb=256 gives 256 KB (single buffer only).
    {
        const uint64_t l0cPerCbBytes =
            static_cast<uint64_t>(_Params.Mb) * _Params.Nb * sizeof(int32_t);
        const uint64_t l0cHalf = static_cast<uint64_t>(l0cSize) / 2ULL;
        _Params.dbL0c = (l0cPerCbBytes <= l0cHalf);
    }

    // Stage quantized x as ZN in GM workspace when the M*K cost amortizes
    // the cross-pipe MySyncAll cost:
    //   E==1 && M*K >= 256 KB                        (dense FFN / prefill)
    //   E in [2..7] && M_per_expert * K >= 18 KB     (small-MoE per-expert)
    // Below those thresholds, the synchronization overhead exceeds the
    // PhaseX reuse benefit - fall back to per-core PhaseX into xL1.
    {
        const uint64_t mTimesK = _Params.originM * _Params.originKAligned32;
        // Stage E==1, M>=16, N>=256. N<256 is excluded (xZN round-trip cost
        // exceeds the gain); M16 is the floor for the staged path.
        const bool stageXZN = (_Params.originE == 1 && _Params.originM >= 16 &&
                               _Params.originN >= 256);
        _Params.useXZNWorkspace = stageXZN;
        // alignedM = ceil(originM, M_FRACTAL) so PhaseX-out stride matches.
        const uint64_t alignedM = ((_Params.originM + NM_FRACTAL_INT8 - 1)
                                   / NM_FRACTAL_INT8) * NM_FRACTAL_INT8;
        _Params.xZNWorkspaceBytes = stageXZN ? (alignedM * _Params.originKAligned32) : 0ULL;
    }

    // Tiling-key (forward compat only - kernel routes all into unified class).
    if (_Params.originM <= 8 && !isGrouped) {
        tilingKey = QuantMatmulDequantTilingKey::GEMV;
    } else if (!isGrouped) {
        tilingKey = QuantMatmulDequantTilingKey::NORMAL;
    } else {
        tilingKey = QuantMatmulDequantTilingKey::GROUPED;
    }

    return true;
}

bool QuantMatmulDequantTiling::SetTilingData(gert::TilingContext* context)
{
    tilingData.set_coreNum(static_cast<uint32_t>(_Params.coreNum));
    tilingData.set_MCoreNum(_Params.MCoreNum);
    tilingData.set_NCoreNum(_Params.NCoreNum);
    tilingData.set_M(static_cast<uint32_t>(_Params.originM));
    tilingData.set_K(static_cast<uint32_t>(_Params.originK));
    tilingData.set_N(static_cast<uint32_t>(_Params.originN));
    tilingData.set_E(static_cast<uint32_t>(_Params.originE));
    tilingData.set_originKAligned32(static_cast<uint32_t>(_Params.originKAligned32));
    tilingData.set_originKAligned512(static_cast<uint32_t>(_Params.originKAligned512));
    tilingData.set_fracM(static_cast<uint32_t>(_Params.fracM));
    tilingData.set_fracK(static_cast<uint32_t>(_Params.fracK));
    tilingData.set_fracN(static_cast<uint32_t>(_Params.fracN));
    tilingData.set_Mb(_Params.Mb);
    tilingData.set_Nb(_Params.Nb);
    tilingData.set_Kb(_Params.Kb);
    tilingData.set_kPasses(_Params.kPasses);
    tilingData.set_wChunkKPasses(_Params.wChunkKPasses);
    tilingData.set_phaseXChunk(_Params.phaseXChunk);
    tilingData.set_phaseXFullPairs(_Params.phaseXFullPairs);
    tilingData.set_phaseXTail(_Params.phaseXTail);
    tilingData.set_maxOneTurnToken(_Params.maxOneTurnToken);
    tilingData.set_maxWeightColOneTurn(_Params.maxWeightColOneTurn);
    tilingData.set_perCoreIters(_Params.perCoreIters);
    tilingData.set_remainderIters(_Params.remainderIters);
    tilingData.set_weightBlockN1(_Params.weightBlockN1);

    // nBlockSize = how many cbs of (Nb*K bytes) of weight fit in L2
    // simultaneously, so 8 cores stagger cb iteration by (coreIdx %
    // nBlockSize) and all hit L2. L2_EFF_BYTES = L2_HEADROOM (14 MB) *
    // L2_WEIGHT_SHARE (70%) ~ 9.8 MB. Clamped to [1, totalColBatch].
    {
        const uint32_t totalColBatch =
            static_cast<uint32_t>((_Params.originN + _Params.Nb - 1) / _Params.Nb);
        const uint64_t cbBytes =
            static_cast<uint64_t>(_Params.Nb) * _Params.originK;  // int8 weight cb
        const uint64_t l2_eff_bytes =
            (uint64_t)AscendC::L2_HEADROOM_BYTES *
            AscendC::L2_WEIGHT_SHARE_PCT / 100;
        uint32_t nBlockSize = (cbBytes > 0)
            ? static_cast<uint32_t>(l2_eff_bytes / cbBytes) : 1;
        if (nBlockSize == 0) nBlockSize = 1;
        if (nBlockSize > totalColBatch) nBlockSize = totalColBatch;
        _Params.nBlockSize = nBlockSize;
    }
    tilingData.set_nBlockSize(_Params.nBlockSize);

    uint32_t flags = 0;
    if (_Params.dynamicQuant) flags |= QGMMD_FLAG_DYNAMIC_QUANT;
    if (_Params.smoothScale)  flags |= QGMMD_FLAG_HAS_SMOOTH;
    if (_Params.perToken)     flags |= QGMMD_FLAG_PER_TOKEN;
    if (_Params.scaleIsInt64) flags |= QGMMD_FLAG_SCALE_IS_INT64;
    if (_Params.weightBlockN1 != 0) flags |= QGMMD_FLAG_BLOCKED_ZN;
    if (_Params.smallM)       flags |= QGMMD_FLAG_SMALL_M;
    if (_Params.l2WeightFits) flags |= QGMMD_FLAG_L2_WEIGHT_FITS;
    if (_Params.hasXRawL1)    flags |= QGMMD_FLAG_HAS_X_RAW_L1;
    if (_Params.useL2Hint)    flags |= QGMMD_FLAG_USE_L2_HINT;
    if (_Params.dbL0c)        flags |= QGMMD_FLAG_DBL0C;
    if (_Params.useXZNWorkspace) flags |= QGMMD_FLAG_USE_XZN_WORKSPACE;
    tilingData.set_flags(flags);

    tilingData.set_ubKMask(_Params.ubKMask);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return true;
}

bool QuantMatmulDequantTiling::SetLaunchInfo(gert::TilingContext* context)
{
    context->SetBlockDim(static_cast<uint32_t>(_Params.coreNum));
    // The unified kernel compiles to a single entry `_0`; the runtime
    // selects binaries via the simplifiedKey / staticKey (input dtypes),
    // and our entry dispatcher reads TilingData->flags for the remaining
    // template bits. `tilingKey` (NORMAL/GROUPED/...) is kept host-only;
    // the kernel entry index is 0.
    context->SetTilingKey(0);
    (void)tilingKey;
    // SYS_WORKSPACE + xZN GM workspace (when staging enabled). The staged
    // path also needs a sync slab (coreNum*32) + dynamic xScale slab
    // (originM*32, dyn+perToken).
    uint64_t ws = static_cast<uint64_t>(SYS_WORKSPACE_310P) + _Params.xZNWorkspaceBytes;
    if (_Params.useXZNWorkspace) {
        ws += _Params.coreNum * BLOCK_SIZE;
        ws += (_Params.dynamicQuant && _Params.perToken) ? _Params.originM * BLOCK_SIZE : 0ULL;
    }
    AddWorkspace(context, ws);
    return true;
}

ge::graphStatus QuantMatmulDequantTiling::runTiling(gert::TilingContext* context, bool is_grouped)
{
    isGrouped = is_grouped;
    _Params = QuantMatmulDequantParam{};

    OP_CHECK_IF(!GetPlatformInfo(context),
                OP_LOGE(context->GetNodeName(), "qgmmd platform info fail"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GetCheckAttr(context),
                OP_LOGE(context->GetNodeName(), "qgmmd attr check fail"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInOutShapes(context),
                OP_LOGE(context->GetNodeName(), "qgmmd shape check fail"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GetTilingData(context),
                OP_LOGE(context->GetNodeName(), "qgmmd tiling compute fail"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SetTilingData(context),
                OP_LOGE(context->GetNodeName(), "qgmmd tiling emit fail"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SetLaunchInfo(context),
                OP_LOGE(context->GetNodeName(), "qgmmd launch info fail"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForQuantGroupedMatmulDequant(gert::TilingContext* context)
{
    QuantMatmulDequantTiling tiling_handle;
    context->SetScheduleMode(BATCH_MODE);
    return tiling_handle.runTiling(context, true);
}

ge::graphStatus TilingPrepareForQuantGroupedMatmulDequant(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantMatmulDequantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    std::string val;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    OP_CHECK_IF(!val.empty(),
                OP_LOGE(context->GetNodeName(), "qgmmd supports ASCEND310P only"),
                return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,   compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,   compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,   compileInfoPtr->l2Size);

    OP_CHECK_IF((compileInfoPtr->coreNum == 0 || compileInfoPtr->ubSize == 0 ||
                 compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 ||
                 compileInfoPtr->l0ASize == 0 || compileInfoPtr->l0BSize == 0),
                OP_LOGE(context->GetNodeName(),
                        "qgmmd platform invalid, coreNum=%u, ubSize=%lu, l1Size=%lu, "
                        "l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
                        compileInfoPtr->coreNum, compileInfoPtr->ubSize,
                        compileInfoPtr->l1Size, compileInfoPtr->l0CSize,
                        compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
                return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantGroupedMatmulDequant)
    .Tiling(TilingForQuantGroupedMatmulDequant)
    .TilingParse<QuantMatmulDequantCompileInfo>(TilingPrepareForQuantGroupedMatmulDequant);

}  // namespace optiling
