/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 求解层: 调度上下文 + 公共逻辑片段
#pragma once

#include "fit_balance_types.h"
#include "comm_model.h"
#include "matmul_model.h"
#include "cut_planner.h"

// 调度上下文: 三件套(模型×2 + 切分器) + 键控维度 + 派生倍数 + 过程量
struct FitBasedBalanceCtx {
    MatmulPerfModel matmulPerf;
    HCCLCommModel commPerf;
    CutPlanner tilingM;
    LocalMMPosition localMMPosition;
    CommComputeOrder order;
    TilingCommEngine commEngine;
    uint64_t commRankCntForMm = 1; // 派生: 稳态工作集数据份数(FRONT=rankDim−1, 其余=volumeMul)
    uint64_t mmRankMul = 1;        // 派生: mm 总行数倍数(fullVolume ? rankDim : 1)
    uint64_t commVolumeMul = 1;    // 派生: 单次通信物理字节倍数(⑥ 消费)
    double frontMMTime = 0;        // FRONT: 本地首块计算耗时
    double backMMTime = 0;         // BACK: 本地尾块计算耗时
};

// 长短块等长时退化为纯等分
inline void MergeEqualTiles(CutPlanner& tilingM)
{
    if (tilingM.cutRes.shortTileLen == tilingM.cutRes.longTileLen) {
        tilingM.cutRes.shortTileLen = 0U;
        tilingM.cutRes.numShortTile = 0U;
        tilingM.cutRes.numLongTile++;
    }
}

// 线性度好的 shape(k×n 较大, 通信拟合更可信)
inline bool IsGoodLinearityShape(const MatmulShapeInfo& mmInfo)
{
    return mmInfo.kValue * mmInfo.nValue >= GOOD_LINEARITY_NK_BYTES * BYTES_PER_MB;
}

// 双向配平长块(NONE 形态): 短块与相邻长块构成"通信 ∥ 计算"边界配对,
// 长块按配平等长反解。配对方向由通算顺序决定:
//   计算在前: 短块在后 → comm(长)=mm(短); 短块在前 → mm(长)=comm(短)
//   通信在前: 短块在后 → mm(长)=comm(短); 短块在前 → comm(长)=mm(短)
inline void BalanceLongTileLen(FitBasedBalanceCtx& ctx)
{
    const bool commSideCovers = (ctx.order == CommComputeOrder::COMPUTE_FIRST) == ctx.tilingM.cutRes.shortTileAtBack;
    if (commSideCovers) {
        const double targetTime = ctx.matmulPerf.MatmulTime(ctx.tilingM.cutRes.shortTileLen, ctx.mmRankMul);
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(targetTime);
    } else {
        const double targetTime = ctx.commPerf.CommTime(ctx.tilingM.cutRes.shortTileLen);
        ctx.tilingM.cutRes.longTileLen = ctx.matmulPerf.InverseMatmulTime(targetTime, ctx.mmRankMul);
    }
}

// 总量估计: 不切口径(轮数=1)的通信与计算总耗时, 用于短块位置判定与 ⑤ 的串行基准
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalCommTimeNoCut = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(FitBasedBalanceCtx& ctx)
{
    ctx.matmulPerf.FindCubeUtil(ctx.commRankCntForMm);
    ctx.matmulPerf.GetMatmulGradient();
    EstimateTimesResult r;
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.mmRankMul);
    r.totalCommTimeNoCut = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    return r;
}

// 流水线总时估计(切 N 轮的近似): 不可重叠段在流水线两端, 中段取瓶颈侧。
// 单块(含 NoCut 候选)返回 serialTotalTime。
// totalMmTime/serialTotalTime 由调用方一次算好传入(搜索循环内不变)。
inline double EstimateCutTotalTime(FitBasedBalanceCtx& ctx, const CommCutResult& cut, double totalMmTime,
                                   double serialTotalTime)
{
    if (cut.totalTileCnt <= 1U) {
        return serialTotalTime;
    }
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.mValue;
    const double totalCommTime = ctx.commPerf.TotalCommTime(mValue, cut.totalTileCnt);
    const uint64_t firstLen = (cut.numShortTile > 0U && !cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    const uint64_t lastLen = (cut.numShortTile > 0U && cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    if (ctx.localMMPosition == LocalMMPosition::FRONT) {
        // 通信在前 + 本地首块: 首块通信不可重叠, 末段计算收尾
        const double commFirst = ctx.commPerf.CommTime(firstLen);
        const double mmLast = ctx.matmulPerf.MatmulTime(lastLen, ctx.mmRankMul);
        return commFirst + std::max(totalCommTime - commFirst, totalMmTime - mmLast) + mmLast;
    }
    if (ctx.localMMPosition == LocalMMPosition::BACK) {
        // 计算在前 + 本地尾块: 首块计算不可重叠, 末段本地窗口与末块通信并行
        const double mmFirst = ctx.matmulPerf.MatmulTime(firstLen, ctx.mmRankMul);
        const double commLast = ctx.commPerf.CommTime(lastLen);
        const double mmRest = totalMmTime - mmFirst - ctx.backMMTime;
        return mmFirst + std::max(mmRest, totalCommTime - commLast) + std::max(commLast, ctx.backMMTime);
    }
    if (ctx.order == CommComputeOrder::COMM_FIRST) {
        // 无本地块 + 通信在前: 首块通信不可重叠, 末块计算收尾
        const double commFirst = ctx.commPerf.CommTime(firstLen);
        const double mmLast = ctx.matmulPerf.MatmulTime(lastLen, ctx.mmRankMul);
        return commFirst + std::max(totalCommTime - commFirst, totalMmTime - mmLast) + mmLast;
    }
    // 无本地块 + 计算在前: 首块计算不可重叠, 末块通信收尾
    const double mmFirst = ctx.matmulPerf.MatmulTime(firstLen, ctx.mmRankMul);
    const double commLast = ctx.commPerf.CommTime(lastLen);
    return mmFirst + std::max(totalMmTime - mmFirst, totalCommTime - commLast) + commLast;
}

// 单轮通信量上限整形(CCU 专用): HCCL 单次调用 256MB 上限, 超限块按候选表
// {2048,1024,512,256} 等长重切(大块优先=少轮次), 尾块吸收余数;
// 轮次预算 = MAX_COMM_ROUNDS−1(1 轮保留给 scale 通道)。
// 两约束(256MB/轮次预算)冲突时放弃整形。
inline void ApplySingleRoundCommLimit(CutPlanner& tilingM, const HCCLCommModel& commPerf, uint64_t volumeMul,
                                      uint64_t mValue)
{
    constexpr uint64_t MAX_ROUND_COMM_BYTES = 268435456;               // HCCL 单次调用通信量上限 (256MB)
    constexpr uint64_t TILE_LEN_CANDIDATES[] = {2048, 1024, 512, 256}; // 等长重切候选(大值优先=少轮次)
    constexpr uint64_t CANDIDATE_ALIGN = 256;                          // 候选全超限时的兜底对齐粒度
    // 单行通信字节数(须含份数 volumeMul, 与 k 的标定口径无关)
    const uint64_t rowBytes = commPerf.commTypeInfo_.commMatrixLen * commPerf.commTypeInfo_.commDtypeSize * volumeMul /
                              commPerf.commTypeInfo_.commDtypeSizeExpansionFraction;
    if (rowBytes == 0) {
        return;
    }
    if (tilingM.cutRes.longTileLen * rowBytes <= MAX_ROUND_COMM_BYTES) {
        return; // 未超限: 保持长短配平结构
    }
    const uint64_t maxTileLen = MAX_ROUND_COMM_BYTES / rowBytes;
    if (maxTileLen < CANDIDATE_ALIGN) {
        return; // 单行通信量过大, 任何对齐块都超限——放弃整形
    }
    // 优选候选(降序第一个 ≤ maxTileLen); 全部超限则 floor 对齐兜底
    uint64_t tileLen = 0;
    for (const uint64_t cand : TILE_LEN_CANDIDATES) {
        if (cand <= maxTileLen) {
            tileLen = cand;
            break;
        }
    }
    if (tileLen == 0) {
        tileLen = maxTileLen / CANDIDATE_ALIGN * CANDIDATE_ALIGN;
    }
    // 轮次预算: 候选块太小块数超预算时, 反向取满足预算的最小块
    const uint64_t roundsBudget = MAX_COMM_ROUNDS - 1;
    if ((mValue + tileLen - 1) / tileLen > roundsBudget) {
        const uint64_t minLen =
            ((mValue + roundsBudget - 1) / roundsBudget + CANDIDATE_ALIGN - 1) / CANDIDATE_ALIGN * CANDIDATE_ALIGN;
        uint64_t pick = minLen;
        for (const uint64_t cand : TILE_LEN_CANDIDATES) {
            if (cand >= minLen) {
                pick = cand;
                break;
            }
        }
        if (pick > maxTileLen) {
            return; // 256MB 上限与轮次预算冲突——放弃整形
        }
        tileLen = pick;
    }
    // 等长重切: 主块等长 + 尾块余数, 短块位置保持
    tilingM.cutRes.longTileLen = tileLen;
    tilingM.cutRes.numLongTile = mValue / tileLen;
    tilingM.cutRes.shortTileLen = mValue - tilingM.cutRes.numLongTile * tileLen;
    tilingM.cutRes.numShortTile = (tilingM.cutRes.shortTileLen > 0) ? 1 : 0;
    tilingM.cutRes.totalTileCnt = tilingM.cutRes.numLongTile + tilingM.cutRes.numShortTile;
}
