/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 离散切分引擎: "1 短 + N 长"形态的初切/离散化/尾对齐/NoCut
#pragma once

#include "fit_balance_types.h"

constexpr uint64_t GOOD_LINEARITY_NK_BYTES = 32;
constexpr uint64_t SMALL_CUT_COUNT = 2;
constexpr double BACK_TILE_RATIO = 0.75;
constexpr double ALIGN_TOL_CALC_BOUND = 0.5;
constexpr double ALIGN_TOL_COMM_BOUND = 0.75;

// 离散切分与对齐: 总长守恒与对齐粒度的守门人
class CutPlanner {
public:
    CommCutResult cutRes;
    CutParams tileArgs;
    uint64_t totalLen = 1U;
    double backTileRatio = BACK_TILE_RATIO; // 短块占位折算比

    explicit CutPlanner(const FitBalanceTilingArgs& args)
    {
        totalLen = args.mValue;
        tileArgs.mAlignLen = DEFAULT_BASE_M; // 切分长度照顾mm性能，对齐baseM
        tileArgs.minTileLen = tileArgs.mAlignLen;
        tileArgs.maxTileCnt = 1U;
        tileArgs.maxTileLen = totalLen;

        cutRes.shortTileAtBack = false;
        NoCutTiling();
    };

    void SetMinLenByMax(uint64_t newLen);
    uint64_t GetMinLen()
    {
        return tileArgs.minTileLen;
    };
    void SetAlignLength(uint64_t newLen)
    {
        tileArgs.mAlignLen = std::max(newLen, 1UL);
    };
    uint64_t GetAlignLength()
    {
        return tileArgs.mAlignLen;
    };
    void SetMaxTileCnt(uint64_t newSize)
    {
        tileArgs.maxTileCnt = std::max(newSize, 1UL);
    };

    void GenerateInitialPartition();
    void NoCutTiling();
    void MaxTileCntUniform();
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool goodLinearityShape, bool localBlockProtected = false);
    void CalcShortTile();
    void CalcBackTile();
    void RedistributeTile();
};

inline void CutPlanner::SetMinLenByMax(uint64_t newLen)
{
    tileArgs.minTileLen = std::max(tileArgs.minTileLen, newLen);
}

// 对齐短块, 长块吸收差值(长块大时短块不对齐的 tile 浪费比更高)
inline void CutPlanner::AlignTileLarge()
{
    if (cutRes.numLongTile >= tileArgs.mAlignLen) {
        return;
    }
    cutRes.shortTileLen = (cutRes.shortTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + cutRes.numLongTile - 1) / cutRes.numLongTile;
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
}

// 长块小时不动对齐, 只修退化尾块: ①碎尾(每长块借一格凑大) ②长短倒挂(重定长块)
inline void CutPlanner::AlignTileSmall()
{
    bool smallBackTile = (cutRes.shortTileLen < tileArgs.mAlignLen) || (cutRes.shortTileLen < tileArgs.minTileLen);
    smallBackTile = smallBackTile && cutRes.shortTileLen > 0U;
    bool largeFrontTile =
        (cutRes.longTileLen > 2 * tileArgs.mAlignLen) && (cutRes.longTileLen > 2 * tileArgs.minTileLen);
    if (smallBackTile && largeFrontTile) {
        // 碎尾吸收: 每长块让出一格, 尾块增长 numLongTile 格(转移守恒)
        uint64_t demand = tileArgs.mAlignLen * cutRes.numLongTile;
        bool reBalance = (cutRes.shortTileLen + demand) <= cutRes.longTileLen;
        if (reBalance) {
            cutRes.shortTileLen += demand;
            cutRes.longTileLen -= tileArgs.mAlignLen;
        }
    }

    // 长短倒挂校正: 尾块 > 主块时, 按 total/总块数 ceil 对齐重定主块
    bool largeBackTile = (cutRes.shortTileLen > cutRes.longTileLen) &&
                         ((totalLen / cutRes.totalTileCnt) > (tileArgs.mAlignLen * cutRes.totalTileCnt));
    if (largeBackTile) {
        cutRes.longTileLen =
            (totalLen / cutRes.totalTileCnt + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    }
}

// 初切: 夹取长块到 [short, total−short] → 对齐(余数 > 256×tol 才进位:
// 计算重 0.5 中性舍入 / 通信重 0.75 消极进位) → 估算长块个数(短块按 75% 占位折算)
inline void CutPlanner::GenerateInitialPartition()
{
    cutRes.longTileLen = std::max(cutRes.longTileLen, cutRes.shortTileLen);
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen - cutRes.shortTileLen);
    uint64_t newTileLen = cutRes.longTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    double alignRatio = ALIGN_TOL_CALC_BOUND;
    if (!cutRes.shortTileAtBack) {
        alignRatio = ALIGN_TOL_COMM_BOUND;
    }

    bool alignUpFlag = (cutRes.longTileLen - newTileLen) > (tileArgs.mAlignLen * alignRatio);
    if (alignUpFlag) {
        cutRes.longTileLen = newTileLen + tileArgs.mAlignLen;
    } else {
        cutRes.longTileLen = newTileLen;
    }
    cutRes.longTileLen = std::min(cutRes.longTileLen, tileArgs.maxTileLen);
    cutRes.numLongTile = (totalLen - static_cast<uint64_t>(static_cast<double>(cutRes.shortTileLen) * backTileRatio)) /
                         cutRes.longTileLen;
    cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

// 离散化主流程: 初切 → 残余处理(等分/均摊/追加) → 轮次约束 → 尾块收守恒
inline void CutPlanner::FitTileLengthDiscrete(bool goodLinearityShape, bool localBlockProtected)
{
    GenerateInitialPartition();

    uint64_t resLen = totalLen - cutRes.numLongTile * cutRes.longTileLen - cutRes.numShortTile * cutRes.shortTileLen;
    bool largeBackTile =
        (resLen >= cutRes.shortTileLen) && ((SMALL_CUT_COUNT * cutRes.shortTileLen) >= cutRes.longTileLen);
    goodLinearityShape =
        (goodLinearityShape && (cutRes.shortTileLen >= tileArgs.mAlignLen)) || (!cutRes.shortTileAtBack);
    bool smallCut = cutRes.numLongTile <= SMALL_CUT_COUNT;
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    if (balanceCut) { // 重新均匀切分
        cutRes.longTileLen = cutRes.shortTileLen;
        cutRes.numLongTile = (totalLen - cutRes.shortTileLen) / cutRes.longTileLen; // floor
        cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
        cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
        largeBackTile = false;
    }
    if (cutRes.totalTileCnt == 2U && !localBlockProtected) { // 有本地块的形态跳过(保护端部结构)
        RedistributeTile();
        largeBackTile = false;
    }
    bool moreFrontTile = largeBackTile && (cutRes.longTileLen <= cutRes.shortTileLen + resLen);
    if (moreFrontTile) { // 增加长块个数
        cutRes.numLongTile++;
        CalcBackTile();
    }

    // 轮次约束: 超预算且恰好整除对齐时改写纯等长
    tileArgs.maxTileCnt = std::max(tileArgs.maxTileCnt, 1UL);
    if (cutRes.totalTileCnt > tileArgs.maxTileCnt) {
        uint64_t tileLen = totalLen / tileArgs.maxTileCnt;
        bool isDivisible = (tileLen * tileArgs.maxTileCnt == totalLen) && (tileLen % tileArgs.mAlignLen == 0U);
        if (isDivisible) {
            cutRes.numLongTile = tileArgs.maxTileCnt;
            cutRes.longTileLen = tileLen;
            cutRes.numShortTile = 0U;
            cutRes.shortTileLen = 0U;
            cutRes.totalTileCnt = tileArgs.maxTileCnt;
        }
    }
    MaxTileCntUniform();

    // 尾块收守恒 + 对齐收尾(按长块大小分两路)
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    if (cutRes.longTileLen > BYTES_PER_KB) {
        AlignTileLarge();
    } else {
        AlignTileSmall();
    }
}

inline void CutPlanner::NoCutTiling()
{
    cutRes.totalTileCnt = 1U;
    cutRes.numShortTile = 0U;
    cutRes.numLongTile = 1U;
    cutRes.shortTileLen = 0U;
    cutRes.longTileLen = totalLen;
}

// 尾块重算: short = total − numLong × long(纯余数)
inline void CutPlanner::CalcShortTile()
{
    cutRes.numShortTile = 0;
    cutRes.shortTileLen = 0;
    if (totalLen != cutRes.longTileLen * cutRes.numLongTile) {
        cutRes.numShortTile = 1;
        cutRes.shortTileLen = totalLen - cutRes.longTileLen * cutRes.numLongTile;
    }
}

// 尾块重算 + 碎尾合并(重算后尾块 < 半格且长块 > 1 时, 末长块并入尾块, 总块数少一轮)
inline void CutPlanner::CalcBackTile()
{
    CalcShortTile();
    bool tinyBackTile = (cutRes.shortTileLen > 0) && (cutRes.shortTileLen < tileArgs.mAlignLen * 0.5);
    if (tinyBackTile && cutRes.numLongTile > 1) {
        cutRes.numLongTile -= 1;
        cutRes.shortTileLen += cutRes.longTileLen;
    }
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

// 两块均摊: 残余大时均分给长短两块(据经验, 块数多且每块接近临界值时 ETE 劣化)
inline void CutPlanner::RedistributeTile()
{
    uint64_t tmpBackTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    bool redistTileFlag = tmpBackTileLen > 2 * cutRes.shortTileLen;
    if (redistTileFlag) {
        uint64_t redistTileLen = (tmpBackTileLen - cutRes.shortTileLen) / cutRes.totalTileCnt;
        redistTileLen = redistTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen += redistTileLen;
    }
    CalcBackTile();
}

// 轮次超限兜底: ceil 对齐均分(少一块给尾块留位)
inline void CutPlanner::MaxTileCntUniform()
{
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;
    }
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();
        return;
    }

    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();
}
