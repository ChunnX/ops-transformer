/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_fit_balance_tiling.h
 * \brief AllToAll + quant-matmul fusion: fit-balance adapter over the shared core.
 *
 * This adapter owns only what is specific to the fusion:
 *  - per-row shipped bytes (data + MX scale layout, fp8 / fp4 packing);
 *  - AllToAll fanout (rankDim - 1) and total-K folding (kTotal = rankDim * ka);
 *  - dual-channel (data + scale) CommTilingData emission.
 * Wave maths, latency models, the k search and all guards live in
 * apace/tiling/fit_balance_tiling.h and are shared across fusion ops.
 */

#ifndef __APACE_ALL_TO_ALL_FIT_BALANCE_TILING_H__
#define __APACE_ALL_TO_ALL_FIT_BALANCE_TILING_H__

#pragma once

#include "../comm_tiling_data.h"
#include "../fit_balance_tiling.h"
#include "../../utils/apace_constant.h"

namespace apace {
namespace all_to_all_quant_matmul {

/*!
 * \brief Per-row bytes shipped by one AllToAll round (data + scale channels merged).
 *
 * MX quantised layouts (ka is per-rank K, ka % 64 == 0 is enforced by the op):
 *   data : fp8 -> ka * 1B ; fp4 -> ka / 2B (two elements packed per byte)
 *   scale: ceil(ka / 64) groups * 2B
 */
inline double CalcPerRowBytes(uint64_t ka, bool isFp4)
{
    const double dataBytes = isFp4 ? (static_cast<double>(ka) / 2.0) : static_cast<double>(ka);
    const double scaleBytes = static_cast<double>((ka + 63UL) / 64UL) * 2.0;
    return dataBytes + scaleBytes;
}

/*!
 * \brief Op-facing input: SWAT matmul tiling result + calibrated models.
 */
struct AllToAllFitBalanceInput {
    // Per-rank problem shape (m is per-rank M, ka is per-rank K).
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t ka = 0;
    uint32_t rankDim = 1;
    bool isFp4 = false;

    // Matmul tiling result from QuantMatmulTilingSwat.
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t usedCoreNum = 1;

    // Calibrated models chosen by the op host (see the op-side params header).
    CommLinearModel comm;
    MatmulTheoryModel mm;
    FitBalanceConfig cfg;
};

/*!
 * \brief Scheduling result: the shared cut plus per-channel non-split sizes.
 */
struct AllToAllFitBalanceResult {
    FitBalanceCut cut;
    uint64_t dataNonSplitAxis = 0;  // bytes per row on the data channel (fp4 packs 2 elements per byte)
    uint64_t scaleNonSplitAxis = 0; // bytes per row on the scale channel
};

/*!
 * \brief Run the shared fit-balance core for the AllToAll quant-matmul fusion.
 */
inline AllToAllFitBalanceResult ScheduleAllToAllFitBalance(const AllToAllFitBalanceInput &in)
{
    AllToAllFitBalanceResult res;
    FitBalanceScheduleInput core;
    core.m = in.m;
    core.n = in.n;
    core.kTotal = static_cast<uint64_t>(in.rankDim) * in.ka;
    core.baseM = in.baseM;
    core.baseN = in.baseN;
    core.usedCoreNum = in.usedCoreNum;
    core.commFanout = (in.rankDim > 1U) ? (static_cast<uint64_t>(in.rankDim) - 1UL) : 1UL;
    core.perRowBytes = CalcPerRowBytes(in.ka, in.isFp4);
    core.comm = in.comm;
    core.mm = in.mm;
    core.mm.coreNum = in.usedCoreNum; // compute runs on the cores the matmul tiling picked
    core.cfg = in.cfg;
    res.cut = FitBalanceSchedule(core);
    res.dataNonSplitAxis = in.isFp4 ? (in.ka / 2UL) : in.ka;
    res.scaleNonSplitAxis = in.ka / MX_GROUP_SIZE;
    return res;
}

/*!
 * \brief Write the scheduling result into the kernel-side dual-channel CommTilingData.
 */
inline void FillCommTilingData(const AllToAllFitBalanceResult &res, CommTilingData &dataTiling,
                               CommTilingData &scaleTiling)
{
    dataTiling.splitAxisTileSize = res.cut.headSize;
    dataTiling.splitAxisTileCnt = res.cut.headCnt;
    dataTiling.splitAxisTailSize = res.cut.tailSize;
    dataTiling.splitAxisTailCnt = res.cut.tailCnt;
    dataTiling.nonSplitAxisSize = res.dataNonSplitAxis;
    scaleTiling = dataTiling;
    scaleTiling.nonSplitAxisSize = res.scaleNonSplitAxis;
}

} // namespace all_to_all_quant_matmul
} // namespace apace

#endif // __APACE_ALL_TO_ALL_FIT_BALANCE_TILING_H__
