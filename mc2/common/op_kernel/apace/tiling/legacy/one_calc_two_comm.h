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
 * \file one_calc_two_comm.h
 * \brief 一算两通公式化 tiling 基类（L2 求解层, 公式化分支——EP+TP 双通信域算子）
 *
 * 提取来源（逻辑逐行忠实, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/one_calc_two_comm_tiling.h / .cpp
 *     (OneCalcTwoCommBase 与 OneCalcTwoCommShardHBase 两个基类)
 *
 * 与旧实现的差异:
 *   1. 拟合表/模型参数表经构造函数注入（算子侧标定头提供）;
 *   2. 移除 OP_LOGD 日志;
 *   3. 虚函数钩子（ShardH 的 SetShortTilePositionFlag）保留继承式设计。
 */

#ifndef __APACE_LEGACY_ONE_CALC_TWO_COMM_H__
#define __APACE_LEGACY_ONE_CALC_TWO_COMM_H__

#pragma once

#include <algorithm>

#include "comm_perf_model.h"
#include "form_partition.h"
#include "legacy_tiling_types.h"
#include "matmul_perf_model.h"

namespace apace {
namespace legacy {

// ====== OneCalcTwoComm 参数（来源: one_calc_two_comm_tiling.h） ======
constexpr uint64_t MAX_TILE_CNT_TWO_COMM = 5;
constexpr uint64_t MAX_ALL_TO_ALL_NUM = 32;
constexpr uint64_t MIN_SPLIT_256 = 256;
constexpr uint64_t MIN_SPLIT_EC_LENGTH = 512;
constexpr uint64_t C_ALIGN_LENGTH = 128;
constexpr uint64_t C_MIN_LENGTH = 128;
constexpr double MIN_UNBALANCE_RATIO = 0.2;

class OneCalcTwoCommBase {
public:
    MatmulPerformanceModel bmmPerf;
    HCCLPerformanceModel epCommPerf; // AllToAll
    HCCLPerformanceModel tpCommPerf; // AllGather or ReduceScatter
    FormPartition tilingC;           // Output parameters
    CutResult cutE;                  // Output parameters
    CutResult localCutE;             // Output parameters
    MatmulParameters clusterInfo;
    uint64_t tpDim = ONE;
    uint64_t epDim = ONE;
    bool cutLocal = true;
    uint64_t minProductLocalEC = ONE;
    uint64_t minC = ONE;
    uint64_t minProductEC = ONE;

    // Constructor
    OneCalcTwoCommBase(const TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim, uint64_t batchSize,
                       const CommFitTable &commTable, const MatmulPerfTable &mmTable,
                       SocVersion inputSocVersion = SocVersion::SOC910_93)
        : bmmPerf(args, mmTable, inputSocVersion),
          epCommPerf(static_cast<uint32_t>(inputEpDim), KernelType::ALL_TO_ALL, commTable, inputSocVersion),
          tpCommPerf(static_cast<uint32_t>(inputTpDim), KernelType::ALL_GATHER, commTable, inputSocVersion),
          tilingC(args)
    {
        epDim = inputEpDim;
        tpDim = inputTpDim;
        bmmPerf.SetBatchSize(batchSize);
        clusterInfo = bmmPerf.mmShapeInfo_;

        ECutInit(cutE);
        ECutInit(localCutE);
        tilingC.cutRes.shortTileAtBack = true;
    };

    // TilingMethods
    void ECutInit(CutResult &eCut);
    void EAxisCut(uint64_t tmpCnt, uint64_t eSize, CutResult &eCut) const;
    void MaxTileCntCheckForE();

    // Wrapper function
    void GetTiling();

    virtual ~OneCalcTwoCommBase() {}
};

inline void OneCalcTwoCommBase::ECutInit(CutResult &eCut)
{
    eCut.longTileLen = clusterInfo.batchSize;
    eCut.numLongTile = ONE;
    eCut.shortTileLen = 0;
    eCut.numShortTile = 0;
    eCut.shortTileAtBack = true;
    eCut.totalTileCnt = eCut.numLongTile;
}

inline void OneCalcTwoCommBase::EAxisCut(uint64_t tmpCnt, uint64_t eSize, CutResult &eCut) const
{
    tmpCnt = std::min(eSize, tmpCnt);
    tmpCnt = std::max(ONE, tmpCnt);
    while (tmpCnt > ONE && (eSize % tmpCnt != 0)) { // eSize代表E/Ep。约束E轴均匀切分，tmpCnt整除(E/Ep)
        tmpCnt--;
    }
    eCut.numLongTile = tmpCnt;
    eCut.longTileLen = eSize / tmpCnt;
    eCut.totalTileCnt = eCut.numLongTile;
}

inline void OneCalcTwoCommBase::MaxTileCntCheckForE()
{
    bool obeyMaxCnt = (localCutE.numLongTile + cutE.numLongTile * tilingC.cutRes.totalTileCnt) <= MAX_TILE_CNT_TWO_COMM;
    if (obeyMaxCnt) { // 已经满足约束
        return;
    }

    // 根据MAX_CNT调整E轴切分
    uint64_t tmpCnt = ONE;
    if (epDim <= TWO) { // 切分轮次均分给localCutE和cutE
        // if MAX_CNT not divisible by 2, give remainder cut to cutE
        tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM / TWO, localCutE.numLongTile);
        EAxisCut(tmpCnt, clusterInfo.batchSize, localCutE);
        tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM - tmpCnt, cutE.numLongTile);
        EAxisCut(tmpCnt, clusterInfo.batchSize, cutE);
    } else { // 优先切cutE
        tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM - ONE,
                          cutE.numLongTile); // local本身要占用1个切分轮次
        EAxisCut(tmpCnt, clusterInfo.batchSize, cutE);

        if ((cutE.numLongTile + localCutE.numLongTile) > MAX_TILE_CNT_TWO_COMM) {
            tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM - cutE.numLongTile, localCutE.numLongTile);
            EAxisCut(tmpCnt, clusterInfo.batchSize, localCutE);
        }
    }
}

inline void OneCalcTwoCommBase::GetTiling()
{
    // Get threshold values
    // Local AG/RS: tile_e_local * C * (H/Tp) * Tp * dTypeSize
    // Non-local AG/RS: tile_e * ((Ep - 1) * tile_c) * (H/Tp) * Tp * dTypeSize
    // A2A: Ep * tile_c * (H/Tp) * dTypeSize
    // Non-local BMM: tile_e * ((Ep - 1) * tile_c) * H * (M / Tp), or tile_e *
    // ((Ep - 1) * Tp * tile_c) * H * (M / Tp)
    minProductLocalEC = tpCommPerf.GetLinearThresholdLenCoarse();
    minProductLocalEC = std::max(minProductLocalEC, bmmPerf.GetLinearThresholdLen(ONE));
    uint64_t nonLocalEpSize = epDim - ONE;
    minProductEC = std::max(minProductLocalEC / nonLocalEpSize, MIN_SPLIT_256);
    minC = epCommPerf.GetLinearThresholdLenCoarse(); // A2A cannot skip local

    // Cut local E
    if (cutLocal) {
        uint64_t tmpSize = std::max(minProductLocalEC / tilingC.totalLen,
                                    ONE); // use full C length
        uint64_t tmpCnt = std::max(clusterInfo.batchSize / tmpSize, ONE);
        EAxisCut(tmpCnt, clusterInfo.batchSize, localCutE);
    }

    // Cut E
    uint64_t tmpSize = 0;
    if (tilingC.totalLen < MIN_SPLIT_EC_LENGTH) {
        tmpSize = std::max(minProductEC / tilingC.totalLen, TWO);
    } else {
        tmpSize = std::max(minProductEC / tilingC.totalLen, ONE); // use full C length
    }
    uint64_t tmpCnt = std::max(clusterInfo.batchSize / tmpSize, ONE);
    EAxisCut(tmpCnt, clusterInfo.batchSize, cutE);

    // E max cnt check
    MaxTileCntCheckForE();

    // Cut c
    uint64_t minCutNum = TWO; // if cut c, c totalTileCnt >= 2
    bool underMaxCut = localCutE.totalTileCnt + minCutNum * cutE.totalTileCnt <= MAX_TILE_CNT_TWO_COMM;
    uint64_t maxCutNum = MAX_ALL_TO_ALL_NUM / clusterInfo.batchSize;
    underMaxCut = underMaxCut && (maxCutNum >= minCutNum);
    if (underMaxCut) {
        // Updates C threshold size
        minC = std::max(minC, minProductEC / cutE.longTileLen);
        tilingC.SetMinLenByMax(minC);

        // Try cut C
        if (!tilingC.SetShortTileLen()) {
            tilingC.cutRes.longTileLen = tilingC.cutRes.shortTileLen;
            tilingC.GenerateInitialPartition();

            // C max cnt check
            maxCutNum = std::min(maxCutNum, (MAX_TILE_CNT_TWO_COMM - localCutE.totalTileCnt) / cutE.totalTileCnt);
            tilingC.SetMaxTileCnt(maxCutNum);
            bool kGreaterThanN = clusterInfo.kValue > clusterInfo.nValue;
            tilingC.FitTileLengthContinuous(kGreaterThanN);

            // 长块短块一样长时归一
            if (tilingC.cutRes.shortTileLen == tilingC.cutRes.longTileLen) {
                tilingC.cutRes.shortTileLen = 0;
                tilingC.cutRes.numShortTile = 0;
                tilingC.cutRes.numLongTile++;
            }
        }
    }
}

class OneCalcTwoCommShardHBase {
public:
    MatmulPerformanceModel bmmPerf;
    HCCLPerformanceModel epCommPerf; // AllToAll
    HCCLPerformanceModel tpCommPerf; // AllGather or ReduceScatter
    CutResult cutE;                  // Output parameters
    FormPartition tilingC;           // Output parameters
    CutResult localCutE;             // Output parameters
    MatmulParameters clusterInfo;
    uint64_t tpDim = ONE;
    uint64_t epDim = ONE;
    uint64_t maxLocalCnt = 1;    // Try cut local when maxLocalCut > 1
    uint64_t maxNonLocalCnt = 1; // Try cut non-local when maxNonLocalCut > 1

    // Constructor
    OneCalcTwoCommShardHBase(const TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim, uint64_t batchSize,
                             const CommFitTable &commTable, const MatmulPerfTable &mmTable,
                             SocVersion inputSocVersion = SocVersion::SOC910_93)
        : bmmPerf(args, mmTable, inputSocVersion),
          epCommPerf(static_cast<uint32_t>(inputEpDim), KernelType::ALL_TO_ALL, commTable, inputSocVersion),
          tpCommPerf(static_cast<uint32_t>(inputTpDim), KernelType::ALL_GATHER, commTable, inputSocVersion),
          tilingC(args)
    {
        epDim = inputEpDim;
        epCommPerf.SetFullMeshCommTimeFactor();

        tpDim = inputTpDim;
        bmmPerf.SetBatchSize(batchSize);
        clusterInfo = bmmPerf.mmShapeInfo_;

        InitCutResult(cutE, clusterInfo.batchSize);
        InitCutResult(localCutE, clusterInfo.batchSize);
        tilingC.cutRes.shortTileAtBack = true;
    };

    // TilingMethods
    void InitCutResult(CutResult &tmpCut, uint64_t totalLen) const;
    void CutAxisE(CutResult &cutRes, uint64_t maxCutNum, uint64_t minLen, double unbalanceRatio) const;
    void CutAxisC(uint64_t maxCutNum, uint64_t minLen, double unbalanceRatio);
    void AsignMaxCutNumForBranches(double totalBMMTime, double totalCommTime);
    virtual bool SetShortTilePositionFlag([[maybe_unused]] double totalBMMTime, [[maybe_unused]] double totalCommTime)
    {
        return false;
    };
    void TrimCutResult(CutResult &tmpCut, bool setShortFlag) const;

    // wrapper function
    void GetTiling();

    virtual ~OneCalcTwoCommShardHBase() {}
};

inline void OneCalcTwoCommShardHBase::InitCutResult(CutResult &tmpCut, uint64_t totalLen) const
{
    tmpCut.longTileLen = totalLen;
    tmpCut.numLongTile = ONE;
    tmpCut.shortTileLen = 0;
    tmpCut.numShortTile = 0;
    tmpCut.shortTileAtBack = true;
    tmpCut.totalTileCnt = tmpCut.numLongTile;
}

inline void OneCalcTwoCommShardHBase::AsignMaxCutNumForBranches(double totalBMMTime, double totalCommTime)
{
    uint64_t maxTotalCnt = MAX_TILE_CNT_TWO_COMM;
    if (epDim > TWO) { // Do not cut local branch
        maxNonLocalCnt = maxTotalCnt - ONE;
        return;
    }
    double localMatmulTime = totalBMMTime / static_cast<double>(epDim);
    double nonlocalCommTime = totalCommTime - totalCommTime / static_cast<double>(epDim);
    if (totalCommTime < localMatmulTime) { // Do not cut local branch
        maxLocalCnt = maxTotalCnt - ONE;
    } else if (totalBMMTime < nonlocalCommTime) { // Do not cut non-local branch
        maxNonLocalCnt = maxTotalCnt - ONE;
    } else { // cut both branches
        maxNonLocalCnt = maxTotalCnt / TWO;
        maxLocalCnt = maxTotalCnt - maxNonLocalCnt;
    }
}

inline void OneCalcTwoCommShardHBase::CutAxisE(CutResult &cutRes, uint64_t maxCutNum, uint64_t minLen,
                                               double unbalanceRatio) const
{
    uint64_t totalLen = cutRes.longTileLen;
    minLen = std::max(minLen, ONE);
    if (TWO * minLen > totalLen) { // Total length too short, do not cut
        return;
    }

    if (maxCutNum == TWO) { // Cut by minByMaxRatio
        double targetLong = static_cast<double>(totalLen) / (ONE + unbalanceRatio);
        uint64_t tmpTileLen = std::min(totalLen / TWO, totalLen - static_cast<uint64_t>(targetLong));
        tmpTileLen = std::max(minLen, tmpTileLen);
        cutRes.longTileLen = (tmpTileLen * TWO >= totalLen) ? tmpTileLen : (totalLen - tmpTileLen);
    } else { // Uniform cut using minLen
        uint64_t targetLen = totalLen / maxCutNum;
        targetLen = (totalLen > targetLen * maxCutNum) ? (targetLen + 1) : targetLen;
        cutRes.longTileLen = std::max(targetLen, minLen);
    }

    // Generate final cut
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    if (totalLen > cutRes.numLongTile * cutRes.longTileLen) { // Need short tile
        cutRes.numShortTile = ONE;
        cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    }
    cutRes.totalTileCnt = cutRes.numLongTile + cutRes.numShortTile;
}

inline void OneCalcTwoCommShardHBase::CutAxisC(uint64_t maxCutNum, uint64_t minLen, double unbalanceRatio)
{
    uint64_t totalLen = tilingC.totalLen;
    tilingC.SetMinLenByMax(minLen);
    if (tilingC.SetShortTileLen()) { // Total length too short, do not cut
        return;
    }
    tilingC.SetMaxTileCnt(maxCutNum);

    if (maxCutNum == TWO) { // Cut by minByMaxRatio
        double targetLong = static_cast<double>(totalLen) / (ONE + unbalanceRatio);
        uint64_t tmpTileLen = std::min(totalLen / TWO, totalLen - static_cast<uint64_t>(targetLong));
        tmpTileLen = std::max(minLen, tmpTileLen);
        tmpTileLen = (tmpTileLen + C_ALIGN_LENGTH - 1) / C_ALIGN_LENGTH * C_ALIGN_LENGTH;
        tilingC.cutRes.shortTileLen = (tmpTileLen * TWO < totalLen) ? tmpTileLen : (totalLen - tmpTileLen);
        tilingC.cutRes.longTileLen = totalLen - tilingC.cutRes.shortTileLen;
        tilingC.cutRes.numLongTile = ONE;
    } else { // Uniform cut using minLen
        tilingC.cutRes.longTileLen = tilingC.cutRes.shortTileLen;
        tilingC.GenerateInitialPartition();
        bool kGreaterThanN = clusterInfo.kValue > clusterInfo.nValue;
        tilingC.FitTileLengthContinuous(kGreaterThanN);
    }

    tilingC.cutRes.totalTileCnt = tilingC.cutRes.numLongTile + tilingC.cutRes.numShortTile;
}

inline void OneCalcTwoCommShardHBase::TrimCutResult(CutResult &tmpCut, bool setShortFlag) const
{
    // 长短块一样长时归一
    if (tmpCut.shortTileLen == tmpCut.longTileLen) {
        tmpCut.shortTileLen = 0;
        tmpCut.numShortTile = 0;
        tmpCut.numLongTile++;
    } else if (setShortFlag && tmpCut.numShortTile > 0) {
        tmpCut.shortTileAtBack = false;
    }
}

inline void OneCalcTwoCommShardHBase::GetTiling()
{
    // Local AG/RS: tile_e_local * C * (H/Tp) * Tp *dTypeSize
    // Non-local AG/RS: tile_e * ((Ep - 1) * tile_c) * (H/Tp) * Tp *dTypeSize
    // A2A: Ep * tile_c * (H/Tp) * dTypeSize
    // Non-local BMM: tile_e * ((Ep - 1) * tile_c) * H * (M / Tp)

    // 1. Estimate matmul, communication time
    bmmPerf.FindCubeUtil(clusterInfo.mValue, ONE, false);
    bmmPerf.GetMatmulGradient();
    uint64_t fullM = clusterInfo.batchSize * epDim * clusterInfo.mValue; // (E/ep) * (ep * C)
    double totalBMMTime = bmmPerf.MatmulTime(fullM, ONE);
    double epCommTime = epCommPerf.CommTime(fullM / tpDim);
    double tpCommTime = tpCommPerf.CommTime(fullM / TWO); // Double-ring comm, stepSize = tpDim, rankTile = 2
    double totalCommTime = epCommTime + tpCommTime;
    double unbalanceRatio = std::min(totalBMMTime, totalCommTime) / std::max(totalBMMTime, totalCommTime);
    unbalanceRatio = std::max(MIN_UNBALANCE_RATIO, unbalanceRatio);

    // 2. Get minimum cut length
    // Get threshold values
    uint64_t minProductLocalEC = tpCommPerf.GetLinearThresholdLenCoarse();
    minProductLocalEC = std::max(minProductLocalEC, bmmPerf.GetLinearThresholdLen(ONE));
    uint64_t nonLocalEpSize = epDim - ONE;
    uint64_t minProductEC = std::max(minProductLocalEC / nonLocalEpSize, ONE);
    uint64_t minC = epCommPerf.GetLinearThresholdLenCoarse(); // A2A cannot skip local
    minC = std::max(minC, bmmPerf.GetLinearThresholdLen(ONE)) / nonLocalEpSize;

    // 3. Asign maximum cut number for local and non-local branches
    AsignMaxCutNumForBranches(totalBMMTime, totalCommTime);

    // 4. Cut local branch
    if (maxLocalCnt > ONE) {
        uint64_t minSizeLocalE = std::max(minProductLocalEC / clusterInfo.mValue,
                                          ONE); // use full C length
        CutAxisE(localCutE, maxLocalCnt, minSizeLocalE, unbalanceRatio);
        if (maxLocalCnt > localCutE.totalTileCnt) { // Pass redundant counts to non-local
            maxNonLocalCnt += maxLocalCnt - localCutE.totalTileCnt;
        }
    }

    // 5. Cut non-local branch
    if (maxNonLocalCnt > ONE) {
        // Cut non-local E
        uint64_t minSizeE = std::max(minProductEC / clusterInfo.mValue, ONE); // use full C length
        CutAxisE(cutE, maxNonLocalCnt, minSizeE, unbalanceRatio);

        // Cut non-local C
        uint64_t eTileLen = std::max(cutE.longTileLen,
                                     cutE.shortTileLen); // Try cut c only if eTileLen == 1
        maxNonLocalCnt = maxNonLocalCnt / cutE.totalTileCnt;
        bool flagCutC = (maxNonLocalCnt > ONE) && (eTileLen == ONE);
        if (flagCutC) {
            minC = std::max(minC, minProductEC / cutE.longTileLen);
            CutAxisC(maxNonLocalCnt, minC, unbalanceRatio);
        }
    }

    // 6. Set bool value isShortTileAtBack
    bool flagSetShort = SetShortTilePositionFlag(totalBMMTime, totalCommTime);
    TrimCutResult(localCutE, flagSetShort);
    TrimCutResult(cutE, flagSetShort);
    TrimCutResult(tilingC.cutRes, flagSetShort);
}

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_ONE_CALC_TWO_COMM_H__
