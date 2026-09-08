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
 * \file fit_balance_tiling.h
 * \brief Fit-balance scheduling core shared by apace comm+compute fusion ops.
 *
 * This header is the apace-side extraction of the fit-balance idea used by the
 * legacy mc2/common/op_host/op_tiling fit-balance tiling family, reworked for
 * the compute-first apace pipeline:
 *
 *  - Matmul tiling (base blocks / core partition / tail handling) is owned by
 *    the quant matmul tiling engine and stays optimal; this layer never
 *    restructures compute blocks.
 *  - The M span of one full compute wave is
 *        waveM = ceil(usedCoreNum / nTile) * baseM,  nTile = ceil(n / baseN).
 *    Communication rounds are expressed as k waves packed together
 *    (head = k * waveM, always a multiple of baseM), so the AIC
 *    block-to-round dependency mapping (mPos / headTileSize) never splits a
 *    block across two communication rounds.
 *  - Latency models (all coefficients injected by the op host side):
 *      communication:  t = slope * x + intercept   (x = per-rank MB shipped in one round)
 *      matmul:         t = MACs / (effMacPerCycle * cyclePerUs * coreNum)
 *  - The candidate k (powers of two) minimising the estimated critical path
 *        T(k) = comm(head) + (R - 1) * max(comm(head), mm(head)) + mm(tail)
 *    wins, subject to a derived round cap (totalCommTime / roundCost).
 *
 * Legacy correspondence (mc2 fit-balance v1 -> here):
 *   EstimateMMCommTime / ratioCalcComm_  -> CalcCommCalcRatio + FitBalanceSchedule
 *   maxTilingNum = commTime / roundCost -> CalcMaxRounds
 *   minCutTime total-time gate           -> cfg.minCutTimeUs
 *   short/long tile free-length solving  -> integer wave packing (head = k * waveM)
 *   tiny back-tile merge                 -> cfg.enableTailMerge / cfg.tinyTailRatio:
 *        a sub-threshold tail round is folded into the last head round
 *        (headCnt - 1 uniform rounds + one bigger tail round). The kernel-side
 *        dependency mapping stays correct because (a) the merged round still
 *        starts at a multiple of headSize (no block straddles two rounds) and
 *        (b) the fused urma kernel takes the explicit round count
 *        (LocalParams.roundCnt) instead of deriving ceil(m / head), which
 *        would overcount by one for a bigger-than-head tail.
 */

#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace apace {

/*!
 * \brief Linear communication latency model: y(us) = slope * x(MB) + intercept(us).
 */
struct CommLinearModel {
    double slopeUsPerMb = 0.0;
    double interceptUs = 0.0;

    double RoundTimeUs(double roundSizeMb) const
    {
        return slopeUsPerMb * roundSizeMb + interceptUs;
    }
};

/*!
 * \brief Theoretical matmul latency model: t(us) = MACs / (effMac * cyclePerUs * coreNum).
 */
struct MatmulTheoryModel {
    double effMacPerCycle = 0.0;
    double cyclePerUs = 1689.6; // A5: 1.65 cycle/ns
    uint64_t coreNum = 1;

    double TimeUs(uint64_t m, uint64_t n, uint64_t kTotal) const
    {
        if ((effMacPerCycle <= 0.0) || (cyclePerUs <= 0.0) || (coreNum == 0UL) || (m == 0UL) || (n == 0UL) ||
            (kTotal == 0UL)) {
            return 0.0;
        }
        const long double macs = static_cast<long double>(m) * static_cast<long double>(n) *
                                 static_cast<long double>(kTotal);
        const long double denom = static_cast<long double>(effMacPerCycle) * static_cast<long double>(cyclePerUs) *
                                  static_cast<long double>(coreNum);
        return static_cast<double>(macs / denom);
    }
};

/*!
 * \brief Tunables shared by every fit-balance user (values owned by op host headers).
 */
struct FitBalanceConfig {
    bool enabled = false;        // master switch, wired by the op host side
    double minCutTimeUs = 100.0; // skip splitting when total comm+mm time is below this
    double balanceRatio = 1.2;   // comm/calc ratio below this keeps k = 1 (waves already match)
    double roundCostUs = 10.0;   // fixed cost per extra round when deriving the round cap
    double tinyTailRatio = 0.5;  // tail rounds below this ratio of head are merged into the last head round
    bool enableTailMerge = true; // fold tiny tail rounds (needs kernel-side explicit round count)
    uint64_t maxTileCnt = 16UL;  // hard cap of communication rounds
    uint64_t minRounds = 3UL;    // lower bound of the derived round cap
};

/*!
 * \brief One compute wave derived from the matmul tiling result.
 */
struct MmWaveDesc {
    uint64_t waveM = 0;   // M span of one full core pass: ceil(usedCoreNum / nTile) * baseM
    uint64_t waveCnt = 0; // ceil(m / waveM)
};

inline MmWaveDesc CalcMmWave(uint64_t m, uint64_t n, uint64_t baseM, uint64_t baseN, uint64_t usedCoreNum)
{
    MmWaveDesc wave;
    if ((baseM == 0UL) || (baseN == 0UL) || (m == 0UL)) {
        return wave;
    }
    const uint64_t nTile = (n + baseN - 1UL) / baseN;
    const uint64_t blocksPerWave = (nTile == 0UL) ? 1UL : ((usedCoreNum + nTile - 1UL) / nTile);
    wave.waveM = (blocksPerWave == 0UL) ? baseM : (blocksPerWave * baseM);
    wave.waveCnt = (m + wave.waveM - 1UL) / wave.waveM;
    return wave;
}

/*!
 * \brief Cut result aligned with the kernel-side CommTilingData head/tail layout.
 *
 * Invariants kept for kernel safety (see the AllToAll URMA impl):
 *   - headSize is a multiple of baseM, so no matmul block straddles two rounds;
 *   - headCnt + tailCnt == ceil(m / headSize), matching the AIC wait mapping.
 */
struct FitBalanceCut {
    uint64_t headSize = 0; // splitAxisTileSize
    uint64_t headCnt = 0;  // splitAxisTileCnt
    uint64_t tailSize = 0; // splitAxisTailSize
    uint64_t tailCnt = 0;  // splitAxisTailCnt (0 or 1)

    // Diagnostics for logging and offline auditing.
    uint64_t chosenK = 1;
    uint64_t rounds = 1;
    double ratioCalcComm = 1.0;
    double estTimeUs = 0.0;
    double estTimeUsK1 = 0.0;
    bool noCut = false;
    bool mergedTail = false; // tail was folded into the last head round
};

/*!
 * \brief Fully data-driven input of the shared scheduling core.
 *
 * perRowBytes / commFanout fold the op-specific traffic formula into the
 * shared round size:  roundSizeMb = commFanout * mSpan * perRowBytes / 1MB.
 */
struct FitBalanceScheduleInput {
    // Problem shape.
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t kTotal = 0;

    // Matmul tiling result (owned by the quant matmul tiling engine).
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t usedCoreNum = 1;

    // Communication traffic folding (AllToAll PUT: fanout = rankDim - 1).
    uint64_t commFanout = 1;
    double perRowBytes = 0.0;

    // Calibrated performance models.
    CommLinearModel comm;
    MatmulTheoryModel mm;

    FitBalanceConfig cfg;
};

namespace detail {
constexpr double kFitBalanceMbBytes = 1024.0 * 1024.0;
} // namespace detail

inline double CalcRoundSizeMb(const FitBalanceScheduleInput &in, uint64_t mSpan)
{
    return static_cast<double>(in.commFanout * mSpan) * in.perRowBytes / detail::kFitBalanceMbBytes;
}

// Comm/calc ratio, always >= 1 (legacy ratioCalcComm_).
inline double CalcCommCalcRatio(double commUs, double mmUs)
{
    const double hi = std::max(commUs, mmUs);
    const double lo = std::min(commUs, mmUs);
    if (lo <= 0.0) {
        return (hi > 0.0) ? 1e12 : 1.0;
    }
    return hi / lo;
}

// Round cap derived from total communication time and the per-round cost.
inline uint64_t CalcMaxRounds(double totalCommUs, const FitBalanceConfig &cfg)
{
    double rounds = totalCommUs / std::max(cfg.roundCostUs, 1e-9);
    rounds = std::max(rounds, static_cast<double>(cfg.minRounds));
    const uint64_t capped = static_cast<uint64_t>(rounds);
    return std::min(capped, cfg.maxTileCnt);
}

/*!
 * \brief Shared fit-balance decision core.
 *
 * Flow: wave maths -> total time / ratio estimate -> no-cut gates -> k search
 * (powers of two, round-capped) -> head/tail emission. The output maps 1:1 to
 * CommTilingData; k = 1 reproduces the plain CommTilingBase behaviour.
 */
inline FitBalanceCut FitBalanceSchedule(const FitBalanceScheduleInput &in)
{
    FitBalanceCut cut;
    const MmWaveDesc wave = CalcMmWave(in.m, in.n, in.baseM, in.baseN, in.usedCoreNum);
    if (wave.waveM == 0UL) {
        return cut;
    }
    const uint64_t waveM = wave.waveM;

    const double totalCommUs = in.comm.RoundTimeUs(CalcRoundSizeMb(in, in.m));
    const double totalMmUs = in.mm.TimeUs(in.m, in.n, in.kTotal);
    cut.ratioCalcComm = CalcCommCalcRatio(totalCommUs, totalMmUs);

    // Emit a cut from a head size (headCnt full rounds + one tail round).
    auto makeCut = [&in](uint64_t head) {
        FitBalanceCut c;
        c.headSize = head;
        c.headCnt = in.m / head;
        c.tailSize = in.m % head;
        c.tailCnt = (c.tailSize > 0UL) ? 1UL : 0UL;
        c.rounds = c.headCnt + c.tailCnt;
        return c;
    };
    // Single-round cut covering the whole M range. headSize = m keeps the
    // kernel mapping self-consistent for any m: totalTiles = ceil(m / m) = 1
    // matches commTurn = 1. (The legacy headCnt = 0 / tail = m shape is only
    // valid when m <= waveM; emitting it with m > waveM would make the AIC wait
    // on flags the AIV never raises.)
    auto singleCut = [&in]() {
        FitBalanceCut c;
        c.headSize = in.m;
        c.headCnt = 1UL;
        c.tailSize = 0UL;
        c.tailCnt = 0UL;
        c.rounds = 1UL;
        return c;
    };
    // Critical path estimate for a given head size.
    auto estTime = [&in](uint64_t head) -> double {
        if (head >= in.m) {
            // Single round: communication and compute serialise.
            return in.comm.RoundTimeUs(CalcRoundSizeMb(in, in.m)) + in.mm.TimeUs(in.m, in.n, in.kTotal);
        }
        const uint64_t headCnt = in.m / head;
        const uint64_t tail = in.m % head;
        const uint64_t rounds = headCnt + ((tail > 0UL) ? 1UL : 0UL);
        const double commK = in.comm.RoundTimeUs(CalcRoundSizeMb(in, head));
        const double mmK = in.mm.TimeUs(head, in.n, in.kTotal);
        const double lastMm = in.mm.TimeUs((tail > 0UL) ? tail : head, in.n, in.kTotal);
        return commK + static_cast<double>(rounds - 1UL) * std::max(commK, mmK) + lastMm;
    };

    cut.estTimeUsK1 = estTime(waveM);

    // No-cut gates: degenerate wave count or total time below the cut threshold.
    if ((wave.waveCnt < 2UL) || ((totalCommUs + totalMmUs) < in.cfg.minCutTimeUs)) {
        cut = singleCut();
        cut.noCut = true;
        cut.chosenK = 0UL;
        cut.estTimeUs = totalCommUs + totalMmUs;
        return cut;
    }
    // Balanced: rounds naturally match compute waves, keep the k = 1 behaviour.
    if (cut.ratioCalcComm < in.cfg.balanceRatio) {
        cut = makeCut(waveM);
        cut.chosenK = 1UL;
        cut.estTimeUs = cut.estTimeUsK1;
        return cut;
    }

    // k search over powers of two, respecting the derived round cap. k = 1 is
    // the baseline; single-round candidates enter through head >= m.
    const uint64_t maxRounds = CalcMaxRounds(totalCommUs, in.cfg);
    double bestTime = 0.0;
    uint64_t bestHead = 0UL;
    if (wave.waveCnt <= maxRounds) {
        bestTime = cut.estTimeUsK1;
        bestHead = waveM;
    }
    for (uint64_t k = 2UL; k <= wave.waveCnt; k <<= 1UL) {
        const uint64_t head = k * waveM;
        if (head >= in.m) {
            const double t = totalCommUs + totalMmUs;
            if ((bestHead == 0UL) || (t < bestTime)) {
                bestTime = t;
                bestHead = head;
            }
            break; // larger k degenerates to the same single round
        }
        const uint64_t rounds = in.m / head + ((in.m % head > 0UL) ? 1UL : 0UL);
        if (rounds > maxRounds) {
            continue;
        }
        const double t = estTime(head);
        if ((bestHead == 0UL) || (t < bestTime)) {
            bestTime = t;
            bestHead = head;
        }
    }
    if (bestHead == 0UL) {
        // No candidate satisfies the round cap: fall back to a single round.
        cut = singleCut();
        cut.noCut = true;
        cut.estTimeUs = totalCommUs + totalMmUs;
        return cut;
    }
    cut = makeCut(bestHead);
    cut.chosenK = bestHead / waveM;
    cut.estTimeUs = bestTime;

    // Tiny-tail merge: fold a sub-threshold tail round into the last head
    // round (headCnt - 1 uniform rounds + one bigger tail round). Only applied
    // on the searched path; noCut / balanced gates stay bit-identical to the
    // plain CommTilingBase behaviour. estTimeUs keeps the unmerged estimate
    // (conservative upper bound). Kernel contract: the merged round still
    // starts at a multiple of headSize, and the fused urma kernel takes the
    // explicit round count, so the AIC wait mapping stays correct.
    if (in.cfg.enableTailMerge && (cut.tailCnt == 1UL) && (cut.headCnt >= 1UL) &&
        (static_cast<double>(cut.tailSize) <= in.cfg.tinyTailRatio * static_cast<double>(cut.headSize))) {
        cut.headCnt -= 1UL;
        cut.tailSize += cut.headSize; // merged round = head + tail, still prefix-aligned
        cut.rounds = cut.headCnt + 1UL;
        cut.mergedTail = true;
    }
    return cut;
}

} // namespace apace

#endif // __APACE_FIT_BALANCE_TILING_H__
