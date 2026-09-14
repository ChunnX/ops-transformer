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
 * \file form_partition.h
 * \brief 长短块离散/连续切分对齐引擎（L3 切分层, fit-based-balance 与 one-calc-one/two-comm 共用）
 *
 * 提取来源（逻辑逐行忠实, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/hccl_formulaic_tiling.h   (FormPartition 类 + 参数常量)
 *   - mc2/common/op_host/op_tiling/hccl_formulaic_tiling.cpp (方法实现)
 *
 * 与旧实现的差异: 仅移除 OP_LOGW/OP_LOGD 日志, 其余逐行一致。
 */

#ifndef __APACE_LEGACY_FORM_PARTITION_H__
#define __APACE_LEGACY_FORM_PARTITION_H__

#pragma once

#include <algorithm>

#include "legacy_tiling_types.h"

namespace apace {
namespace legacy {

// ====== FormPartition 参数（来源: hccl_formulaic_tiling.h） ======
constexpr uint64_t LARGE_NK_BAR_BASE = 32;
constexpr uint64_t SMALL_TILECNT_PAR = 2;
constexpr uint64_t OPT_TILE_CNT = 8;
constexpr uint64_t OPT_TILE_LEN_RATIO = 2;
constexpr uint64_t OPT_TILE_LEN_RATIO_SOC310P = 1;
constexpr double BACKTILE_ALIGN_RATIO = 0.75;
constexpr uint64_t LARGE_BACKTILE_COMPARE_PAR = 2;
constexpr uint64_t SOC310_BASE_M = 256;
constexpr uint64_t SMALL_SHORTTILE_BAR = 512;
constexpr uint64_t SHORT_ALIGN_DOWN_DIFF = 16;
constexpr double ALIGN_RATIO_CALC_BOUND = 0.5;
constexpr double ALIGN_RATIO_COMM_BOUND = 0.75;
constexpr uint64_t SMALL_SHAPE_BAR = 2048;

// 切分和对齐
class FormPartition {
public:
    CutResult cutRes;                            // 切分结果
    TileArguments tileArgs;                      // 切分参数
    uint64_t totalLen = ONE;                     // 待切分的长度
    double backTileRatio = BACKTILE_ALIGN_RATIO; // 计算长块个数使用的参数

    // Constructor
    explicit FormPartition(const TilingArgs &args)
    {
        totalLen = args.mValue;
        tileArgs.mAlignLen = BASE_BLOCK_M; // 切分长度照顾mm性能，对齐baseM
        tileArgs.minTileLen = tileArgs.mAlignLen;
        tileArgs.maxTileCnt = ONE;
        tileArgs.maxTileLen = totalLen;

        cutRes.shortTileAtBack = false; // 计算Bound时短块后置，通信bound时短块前置
        NoCutTiling();                  // 初始化为不切分
    };

    void SetBackTileRatio(double newRatio)
    {
        backTileRatio = (newRatio > 0) ? newRatio : backTileRatio;
    };
    void SetMinLenByMax(uint64_t newLen);
    void SetMinLenByMin(uint64_t newLen);
    uint64_t GetMinLen()
    {
        return tileArgs.minTileLen;
    };
    void SetAlignLength(uint64_t newLen)
    {
        tileArgs.mAlignLen = std::max(newLen, ONE);
    };
    uint64_t GetAlignLength()
    {
        return tileArgs.mAlignLen;
    };
    void SetMaxTileCnt(uint64_t newSize)
    {
        tileArgs.maxTileCnt = std::max(newSize, ONE);
    };

    // TilingMethods
    void GenerateInitialPartition(bool smallDimAlignUp = false);
    bool SetShortTileLen(bool noCutFlag = false);
    void NoCutTiling();
    void MaxTileCntUniform();

    // 离散切分对齐 (AllGatherMatmul和MatmulReduceScatter算子)
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront = false);
    // 连续切分对齐 (MatmulAllReduce算子)
    void LargeBackTileCheck(bool soc310Flag, bool largeCalcCommRatio);
    void CalcShortTile();
    void CalcBackTile();
    void SmallFrontTileCheck(bool soc310Flag, bool kGreaterThanN);
    void RedistributeTile();
    void CommBoundShortAlign();
    void FitTileLengthContinuous(bool kGreaterThanN, bool largeCalcCommRatio = true, bool soc310Flag = false);
};

inline void FormPartition::SetMinLenByMax(uint64_t newLen)
{
    tileArgs.minTileLen = std::max(tileArgs.minTileLen, newLen);
}

inline void FormPartition::SetMinLenByMin(uint64_t newLen)
{
    tileArgs.minTileLen = std::min(tileArgs.minTileLen, newLen);
}

inline void FormPartition::AlignTileLarge()
{
    if (cutRes.numLongTile >= tileArgs.mAlignLen) {
        return;
    }
    // Align cutRes.shortTileLen
    cutRes.shortTileLen = (cutRes.shortTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + cutRes.numLongTile - 1) / cutRes.numLongTile;
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
}

inline void FormPartition::AlignTileSmall()
{
    // Front tile already aligned, check backTile
    bool smallBackTile = (cutRes.shortTileLen < tileArgs.mAlignLen) || (cutRes.shortTileLen < tileArgs.minTileLen);
    smallBackTile = smallBackTile && cutRes.shortTileLen > 0U;
    bool largeFrontTile =
        (cutRes.longTileLen > 2 * tileArgs.mAlignLen) && (cutRes.longTileLen > 2 * tileArgs.minTileLen);
    if (smallBackTile && largeFrontTile) {
        uint64_t demand = tileArgs.mAlignLen * cutRes.numLongTile;
        bool reBalance = (cutRes.shortTileLen + demand) <= cutRes.longTileLen;
        if (reBalance) {
            cutRes.shortTileLen += demand;
            cutRes.longTileLen -= tileArgs.mAlignLen;
        }
    }

    // 假如对齐后尾块长度大于主块，尝试校正使尾块长度<=主块长度
    // 校正目标：主块长度为totalTileCnt均分MPerRank后向上对齐mAlignLen，尾块长度为MPerRank剩余大小
    // 校正检查：totalTileCnt均分后，所有主块加起来最多向尾块抽取 mAlignLen *
    // numLongTile 大小，尾块剩余大小为 (MPerRank/totalTileCnt) - mAlignLen *
    // numLongTile 校正检查：检查条件为 (MPerRank/totalTileCnt) > mAlignLen *
    // totalTileCnt, 确保尾块至少剩下mAlignLen大小
    bool largeBackTile = (cutRes.shortTileLen > cutRes.longTileLen) &&
                         ((totalLen / cutRes.totalTileCnt) > (tileArgs.mAlignLen * cutRes.totalTileCnt));
    if (largeBackTile) {
        cutRes.longTileLen =
            (totalLen / cutRes.totalTileCnt + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    }
}

inline void FormPartition::GenerateInitialPartition(bool smallDimAlignUp)
{
    cutRes.longTileLen = std::max(cutRes.longTileLen, cutRes.shortTileLen);
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen - cutRes.shortTileLen);
    uint64_t newTileLen = cutRes.longTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    // 0.5 is half of mAlignLen
    double alignRatio = ALIGN_RATIO_CALC_BOUND;
    if (!cutRes.shortTileAtBack) {
        alignRatio = ALIGN_RATIO_COMM_BOUND;
    }

    bool alignUpFlag = ((cutRes.longTileLen - newTileLen) > (tileArgs.mAlignLen * alignRatio)) || smallDimAlignUp;
    if (alignUpFlag) {
        cutRes.longTileLen = newTileLen + tileArgs.mAlignLen;
    } else {
        cutRes.longTileLen = newTileLen;
    }
    // find cutRes.totalTileCnt
    cutRes.longTileLen = std::min(cutRes.longTileLen, tileArgs.maxTileLen);
    cutRes.numLongTile = (totalLen - static_cast<uint64_t>(static_cast<double>(cutRes.shortTileLen) * backTileRatio)) /
                         cutRes.longTileLen;
    cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

inline void FormPartition::FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront)
{
    // 流水配平，找到初始切分长度和个数
    GenerateInitialPartition(smallDimAlignUp);

    // 处理剩余长度
    uint64_t resLen = totalLen - cutRes.numLongTile * cutRes.longTileLen - cutRes.numShortTile * cutRes.shortTileLen;
    bool largeBackTile =
        (resLen >= cutRes.shortTileLen) && ((SMALL_TILECNT_PAR * cutRes.shortTileLen) >= cutRes.longTileLen);
    goodLinearityShape =
        (goodLinearityShape && (cutRes.shortTileLen >= tileArgs.mAlignLen)) || (!cutRes.shortTileAtBack);
    bool smallCut = cutRes.numLongTile <= SMALL_TILECNT_PAR;
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    if (balanceCut) { // 重新均匀切分
        cutRes.longTileLen = cutRes.shortTileLen;
        cutRes.numLongTile = (totalLen - cutRes.shortTileLen) / cutRes.longTileLen; // floor
        cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
        cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
        largeBackTile = false;
    }
    if (cutRes.totalTileCnt == TWO && !localAtFront) {
        // 如果前面没有local特殊处理，就把剩余长度均分给长块和短块
        RedistributeTile();
        largeBackTile = false;
    }
    bool moreFrontTile = largeBackTile && (cutRes.longTileLen <= cutRes.shortTileLen + resLen);
    if (moreFrontTile) { // 增加长块个数
        cutRes.numLongTile++;
        CalcBackTile();
    }

    // 检查切分轮次总数符合约束
    tileArgs.maxTileCnt = std::max(tileArgs.maxTileCnt, ONE); // 确保maxTileCnt > 0
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

    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    if (cutRes.longTileLen > ONE_KBYTE) {
        AlignTileLarge();
    } else {
        AlignTileSmall();
    }
}

inline void FormPartition::NoCutTiling()
{
    cutRes.totalTileCnt = 1U;
    cutRes.numShortTile = 0U;
    cutRes.numLongTile = 1U;
    cutRes.shortTileLen = 0U;
    cutRes.longTileLen = totalLen;
}

inline void FormPartition::LargeBackTileCheck(bool soc310Flag, bool largeCalcCommRatio)
{
    bool largeBackTile = cutRes.shortTileLen >= (cutRes.longTileLen + tileArgs.mAlignLen);
    if (soc310Flag) {
        largeBackTile = (cutRes.shortTileLen >= (cutRes.longTileLen + SOC310_BASE_M));
    }
    if (largeBackTile) {
        cutRes.longTileLen =
            (totalLen / cutRes.totalTileCnt + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen = std::min(totalLen, cutRes.longTileLen);
        cutRes.numLongTile = totalLen / cutRes.longTileLen;
        CalcBackTile();
    }
    if (largeCalcCommRatio) {
        uint64_t moveLen = cutRes.numLongTile * tileArgs.mAlignLen;
        largeBackTile = (cutRes.shortTileLen >= (cutRes.longTileLen + tileArgs.mAlignLen)) &&
                        (cutRes.shortTileLen >= (moveLen + tileArgs.mAlignLen * LARGE_BACKTILE_COMPARE_PAR) &&
                         (cutRes.shortTileLen <= moveLen * LARGE_BACKTILE_COMPARE_PAR));
        if (largeBackTile) {
            cutRes.longTileLen += tileArgs.mAlignLen;
            cutRes.longTileLen =
                (cutRes.longTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
            cutRes.longTileLen = std::min(totalLen, cutRes.longTileLen);
            cutRes.numLongTile = totalLen / cutRes.longTileLen;
            CalcBackTile();
        }
    }
}

inline void FormPartition::CalcShortTile()
{
    cutRes.numShortTile = 0;
    cutRes.shortTileLen = 0;
    if (totalLen != cutRes.longTileLen * cutRes.numLongTile) {
        cutRes.numShortTile = 1;
        cutRes.shortTileLen = totalLen - cutRes.longTileLen * cutRes.numLongTile;
    }
}

inline void FormPartition::CalcBackTile()
{
    CalcShortTile();
    bool tinyBackTile = (cutRes.shortTileLen > 0) && (cutRes.shortTileLen < tileArgs.mAlignLen * 0.5);
    if (tinyBackTile && cutRes.numLongTile > 1) {
        cutRes.numLongTile -= 1;
        cutRes.shortTileLen += cutRes.longTileLen;
    }
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

// 据经验值，切分块数较多且每块长度接近matmul临界值时，ETE性能会劣化
inline void FormPartition::SmallFrontTileCheck(bool soc310Flag, bool kGreaterThanN)
{
    uint64_t optimalTileLen = tileArgs.minTileLen * OPT_TILE_LEN_RATIO;
    if (soc310Flag) {
        optimalTileLen = tileArgs.minTileLen * OPT_TILE_LEN_RATIO_SOC310P;
    }
    bool smallFrontTile = (cutRes.longTileLen < optimalTileLen) && (totalLen > cutRes.longTileLen * OPT_TILE_CNT);
    bool largeFrontFlag = soc310Flag || (cutRes.shortTileAtBack && kGreaterThanN);
    if (smallFrontTile && largeFrontFlag) {
        optimalTileLen = std::max(optimalTileLen, totalLen / OPT_TILE_CNT);
        optimalTileLen = optimalTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen = optimalTileLen;
        cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
        cutRes.numLongTile = totalLen / cutRes.longTileLen;
        CalcBackTile();
    }
}

inline void FormPartition::RedistributeTile()
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

inline void FormPartition::CommBoundShortAlign()
{
    if (cutRes.shortTileAtBack || cutRes.numLongTile > 1) {
        return;
    }
    uint64_t alignDownLen = cutRes.shortTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    uint64_t alignDownDiff = cutRes.shortTileLen - alignDownLen;
    uint64_t alignUpLen = (cutRes.shortTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    bool alignUpShortFlag = (alignUpLen <= SMALL_SHORTTILE_BAR && cutRes.longTileLen > SMALL_SHORTTILE_BAR) ||
                            alignDownDiff <= SHORT_ALIGN_DOWN_DIFF;
    if (alignUpShortFlag) {
        cutRes.shortTileLen = alignUpLen;
    }
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + cutRes.numLongTile - 1) / cutRes.numLongTile;
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
}

inline void FormPartition::FitTileLengthContinuous(bool kGreaterThanN, bool largeCalcCommRatio, bool soc310Flag)
{
    cutRes.numLongTile = (totalLen - static_cast<uint64_t>(static_cast<double>(cutRes.shortTileLen) * backTileRatio)) /
                         cutRes.longTileLen;
    cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
    RedistributeTile();
    SmallFrontTileCheck(soc310Flag, kGreaterThanN);
    // check totalTileCnt is valid.
    MaxTileCntUniform();

    LargeBackTileCheck(soc310Flag, largeCalcCommRatio);
    CommBoundShortAlign();
}

inline bool FormPartition::SetShortTileLen(bool noCutFlag)
{
    cutRes.shortTileLen = tileArgs.minTileLen;
    cutRes.numShortTile = 1U;
    // do not cut if mValue is too small
    bool smallMPerRank = (totalLen < cutRes.shortTileLen * 2);
    smallMPerRank = smallMPerRank || noCutFlag;
    if (smallMPerRank) {
        NoCutTiling();
    }
    return smallMPerRank;
}

inline void FormPartition::MaxTileCntUniform()
{
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;
    }
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();
        return;
    }

    // 尾块应该小于主块
    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();
}

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_FORM_PARTITION_H__
