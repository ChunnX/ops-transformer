/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 求解层:
// 调度上下文 + InitCtx 上下文构造 + 公共逻辑片段 + Schedule 键控流水
// (①Estimate→②SetShort→③SetLong 配平→④离散化→⑤轮数搜索/门控→⑥CCU 整形)。
#pragma once

#include "fit_balance_types.h"
#include "comm_model.h"
#include "matmul_model.h"
#include "cut_planner.h"

// ============================================================================
// 求解层: 调度上下文 + 公共逻辑片段(求解器键控分支复用)
// ============================================================================

// 调度上下文: 引擎三件套 + 键控维度 + 过程量(刻意不含静态标定数据, 在 FitBalanceCalibration)
struct FitBasedBalanceCtx {
    MatmulPerfModel matmulPerf;
    HCCLCommModel commPerf;
    CutPlanner tilingM;
    LocalMMPosition localMMPosition; // 键控维度: 本地 mm 块位置(,
                                     // 取代原通信原语; 原语语义全部终结在标定层 CommProfile)
    CommComputeOrder order;          // 键控维度: 通算顺序
    TilingCommEngine commEngine;     // 键控维度: 通信引擎(CCU 专属的单轮通信量上限整形)
    uint64_t commRankCntForMm = 1;   // 派生: CommProfile.mmWorksetMul 换算(mm 工作集 M 跨度倍数)
    uint64_t mmRankMul = 1;          // 派生: CommProfile.mmTotalMul 换算(mm 总行数倍数)
    uint64_t commVolumeMul = 1; // 派生: CommProfile.commVolumeMul 换算(单次通信调用物理字节倍数, ⑥ 消费)
    double ratioCalcComm = 1.0; // 过程量: Estimate 骨干设置的通算比(键控分支消费)
    double frontMMTime = 0; // 过程量(FRONT 键): 本地首块计算耗时(流水线首端, 启动端被本地块占据)
    double backMMTime = 0; // 过程量(BACK 键): 本地尾块计算耗时(流水线尾端, 与末块通信重叠)
    // 诊断量: 切分完成后按实际轮数精算的总通信时, 与 NoCut 候单轮
    // 之差 = (N−1)·b 启动税。仅供 op 侧日志消费, 不参与引擎决策
    double commTimeActualUs = 0; // 实际切分(N 轮)的总通信时
    double commTimeNoCutUs = 0;  // 不切候选(1 轮)的总通信时
};

// 公共逻辑片段(可独立调用的计算单元, 键控分支按需复用)
// ============================================================================

// 短长块等长归一(等长时"长短"结构退化为纯等分, 去掉多余的块类型标记)
inline void MergeEqualTiles(CutPlanner &tilingM)
{
    if (tilingM.cutRes.shortTileLen == tilingM.cutRes.longTileLen) {
        tilingM.cutRes.shortTileLen = 0U;
        tilingM.cutRes.numShortTile = 0U;
        tilingM.cutRes.numLongTile++;
    }
}

// 线性度好的 shape（k×n 较大, 通信拟合更可信）
inline bool IsGoodLinearityShape(const MatmulShapeInfo &mmInfo)
{
    return mmInfo.kValue * mmInfo.nValue >= GOOD_LINEARITY_NK_BYTES * BYTES_PER_MB;
}

// 双向配平长块(无本地块形态): 短块是流水线垫片, 长块按它反解——
//   短块后置(计算 bound) → 长块 = 反解通信时长(掩盖短块计算);
//   短块前置(通信 bound) → 长块 = 反解计算时长(掩盖短块通信)。
inline void BalanceLongTileLen(FitBasedBalanceCtx &ctx)
{
    if (ctx.tilingM.cutRes.shortTileAtBack) {
        const double targetTime = ctx.matmulPerf.MatmulTime(ctx.tilingM.cutRes.shortTileLen, ctx.mmRankMul);
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(targetTime);
    } else {
        const double targetTime = ctx.commPerf.CommTime(ctx.tilingM.cutRes.shortTileLen);
        ctx.tilingM.cutRes.longTileLen = ctx.matmulPerf.InverseMatmulTime(targetTime, ctx.mmRankMul);
    }
}

// ---- 以下片段按需调用(调=走公共逻辑, 不调=走各自键控分支), 不设引擎级
//      bool 开关——片段与键控分支逻辑交错, 开关无法表达交错顺序 ----

// Estimate 骨干: FindCubeUtil + GetMatmulGradient + 总量估计 + 通算比;
// shortTileAtBack 判定不在此(各键控分支语义不同, 见 Schedule)。
// mmRankMul 由 InitCtx 从标定 CommProfile 换算。
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalTpTime = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(FitBasedBalanceCtx &ctx)
{
    ctx.matmulPerf.FindCubeUtil(ctx.commRankCntForMm);
    ctx.matmulPerf.GetMatmulGradient();
    EstimateTimesResult r;
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.mmRankMul);
    // N=1: Estimate 发生在切分前、轮数未知, 按不切保守估通信总量;
    // 切分完成后由 Schedule 尾部按实际轮数精算(ctx.commTimeActualUs)供日志/观察
    r.totalTpTime = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    ctx.ratioCalcComm = (std::max(r.totalTpTime, r.totalMatmulTime) / std::min(r.totalTpTime, r.totalMatmulTime));
    return r;
}

// L2 占用判定: A+B 权重工作集超 950 L2 容量(L2_CACHE_BYTES)时切分收益变化
inline bool ExceedsL2Cache(const MatmulShapeInfo &mmInfo)
{
    const uint64_t l2UseSize = mmInfo.mValue * mmInfo.kValue * mmInfo.inMatrixADtypeSize +
                               mmInfo.kValue * mmInfo.nValue * mmInfo.inMatrixBDtypeSize;
    return l2UseSize > L2_CACHE_BYTES;
}

// L2 超限鼓励大块: minLen×2——超限的工作集切成小块会令 L2 反复换入换出, 抬下限让块变大
inline void ApplyL2MinLenBoost(CutPlanner &tilingM, bool exceedsL2)
{
    if (exceedsL2) {
        tilingM.SetMinLenByMax(tilingM.GetMinLen() * 2U);
    }
}

// 单轮通信量上限整形:
// HCCL 单次通信调用有 256MB 上限——单块通信量(块长×单行通信字节)超过时按候选表
// {2048,1024,512,256}(大块优先=少轮次)等长重切, 尾块吸收余数, 完全覆盖长短结构
// (超限属极端大 shape, 约束满足优先); 轮次预算 = MAX_COMM_ROUNDS−1(1 轮保留给 scale
// 通道通信)。
// 量级安全: 块长×单行字节 ≤ 1e6×5e5 = 5e11 << uint64 上限, 无需分段溢出回检;
// 两约束(256MB/轮次预算)冲突时放弃整形, 保持原切分。
// 流水线总时估计: 切 N 轮的近似总时——不可重叠段在
// 流水线两端(与通算顺序对应), 中段取通信/计算瓶颈侧:
//   通信在前(AG):       T_cut ≈ T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
//   通信在后(RS/AR/A2A): T_cut ≈ T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)
// 单块(含 NoCut 候选)返回 serialTotalTime(全串行基准)。totalMmTime/serialTotalTime 由调用方
// 一次算好传入(搜索循环内不变, 免重复求值)。
inline double EstimateCutTotalTime(FitBasedBalanceCtx &ctx, const CommCutResult &cut, double totalMmTime,
                                   double serialTotalTime)
{
    if (cut.totalTileCnt <= 1U) {
        return serialTotalTime; // 单块即 NoCut 候选
    }
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.mValue;
    const double totalCommTime = ctx.commPerf.TotalCommTime(mValue, cut.totalTileCnt);
    const uint64_t firstLen = (cut.numShortTile > 0U && !cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    const uint64_t lastLen = (cut.numShortTile > 0U && cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    if (ctx.localMMPosition == LocalMMPosition::FRONT) {
        // 通信在前 + 本地首块: 首块通信不可重叠(远程数据未到), 末段计算收尾
        const double commFirst = ctx.commPerf.CommTime(firstLen);
        const double mmLast = ctx.matmulPerf.MatmulTime(lastLen, ctx.mmRankMul);
        return commFirst + std::max(totalCommTime - commFirst, totalMmTime - mmLast) + mmLast;
    }
    if (ctx.localMMPosition == LocalMMPosition::BACK) {
        // 计算在前 + 本地尾块: 首块计算不可重叠, 末段本地窗口与末块通信并行(取 max)——
        // 本地窗口时长自总计算中抽出归末段(待 MMA2A 迁移实测校准)
        const double mmFirst = ctx.matmulPerf.MatmulTime(firstLen, ctx.mmRankMul);
        const double commLast = ctx.commPerf.CommTime(lastLen);
        const double mmRest = totalMmTime - mmFirst - ctx.backMMTime;
        return mmFirst + std::max(mmRest, totalCommTime - commLast) + std::max(commLast, ctx.backMMTime);
    }
    const double mmFirst = ctx.matmulPerf.MatmulTime(firstLen, ctx.mmRankMul);
    const double commLast = ctx.commPerf.CommTime(lastLen);
    return mmFirst + std::max(totalMmTime - mmFirst, totalCommTime - commLast) + commLast;
}

inline void ApplySingleRoundCommLimit(CutPlanner &tilingM, const HCCLCommModel &commPerf, uint64_t volumeMul,
                                      uint64_t mValue)
{
    constexpr uint64_t MAX_ROUND_COMM_BYTES = 268435456;               // HCCL 单次调用通信量上限 (256MB)
    constexpr uint64_t TILE_LEN_CANDIDATES[] = {2048, 1024, 512, 256}; // 等长重切候选(大值优先=少轮次)
    constexpr uint64_t CANDIDATE_ALIGN = 256;                          // 候选全超限时的兜底对齐粒度
    // 单行通信字节数(物理量, 256MB 是 HCCL 单次调用的真实传输上限——须含份数, 与 k 的标定口径
    // 融合无关): 通信矩阵另一维 × 每元素字节 / 子字节折算 × 单次调用物理倍数(CommProfile.commVolumeMul)
    const uint64_t rowBytes = commPerf.commTypeInfo_.commMatrixLen * commPerf.commTypeInfo_.commDtypeSize * volumeMul /
                              commPerf.commTypeInfo_.commDtypeSizeExpansionFraction;
    if (rowBytes == 0) {
        return; // 防御: 单行为零无法约束
    }
    if (tilingM.cutRes.longTileLen * rowBytes <= MAX_ROUND_COMM_BYTES) {
        return; // 未超限: 保持长短配平结构
    }
    const uint64_t maxTileLen = MAX_ROUND_COMM_BYTES / rowBytes; // 满足上限的最大块长
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
    // 轮次预算: 候选块太小块数超预算时, 反向取满足预算的最小块(仍优先候选表, 降序第一个 ≥)
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
    // 等长重切(mValue > maxTileLen ≥ tileLen 保证主块数 ≥ 1): 主块等长 + 尾块余数, 短块位置保持
    tilingM.cutRes.longTileLen = tileLen;
    tilingM.cutRes.numLongTile = mValue / tileLen;
    tilingM.cutRes.shortTileLen = mValue - tilingM.cutRes.numLongTile * tileLen;
    tilingM.cutRes.numShortTile = (tilingM.cutRes.shortTileLen > 0) ? 1 : 0;
    tilingM.cutRes.totalTileCnt = tilingM.cutRes.numLongTile + tilingM.cutRes.numShortTile;
}
