/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 求解层(solver, §4.39 拆分——原三/四区):
// 调度上下文 + InitCtx 上下文构造 + 公共逻辑片段 + Schedule 键控流水
// (①Estimate→②SetShort→③SetLong 配平→④离散化→⑤轮数搜索/门控→⑥CCU 整形)。
#pragma once

#include "fit_balance_types.h"
#include "comm_model.h"
#include "matmul_model.h"
#include "cut_planner.h"
#include "calibration.h"

// ============================================================================
// 三、求解层（两分支求解器）
// ============================================================================
// ---- 3.1 自由长度求解器（ctx + 键控分支, HCCL AICPU/CCU; 钩子机制已于 §4.24 废除）----
// ============================================================================

// 调度上下文: 引擎三件套 + 键控维度 + 过程量(刻意不含静态标定数据, 在 FitBalanceCalibration)
struct FitBasedBalanceCtx {
    MatmulPerfModel matmulPerf;
    HCCLCommModel commPerf;
    CutPlanner tilingM;
    LocalMMPosition localMMPosition; // 键控维度: 本地 mm 块位置(§4.38——引擎唯一的结构键控,
                                     // 取代原通信原语; 原语语义全部终结在标定层 CommProfile)
    CommComputeOrder order;          // 键控维度: 通算顺序
    TilingCommEngine commEngine;     // 键控维度: 通信引擎(CCU 专属的单轮通信量上限整形)
    uint64_t commRankCntForMm = 1;   // 派生: CommProfile.mmWorksetMul 换算(mm 工作集 M 跨度倍数)
    uint64_t mmRankMul = 1;          // 派生: CommProfile.mmTotalMul 换算(mm 总行数倍数)
    uint64_t commVolumeMul = 1; // 派生: CommProfile.commVolumeMul 换算(单次通信调用物理字节倍数, ⑥ 消费)
    double ratioCalcComm = 1.0; // 过程量: Estimate 骨干设置的通算比(键控分支消费)
    double frontMMTime = 0; // 过程量(FRONT 键): 本地首块计算耗时(流水线首端, 启动端被本地块占据)
    double backMMTime = 0; // 过程量(BACK 键): 本地尾块计算耗时(流水线尾端, 与末块通信重叠)
    // 诊断量(§4.37, Schedule 尾部写入): 切分完成后按实际轮数精算的总通信时, 与 NoCut 候单轮
    // 之差 = (N−1)·b 启动税。仅供 op 侧日志消费, 不参与引擎决策(机制与日志分离, §4.24 裁定)
    double commTimeActualUs = 0; // 实际切分(N 轮)的总通信时
    double commTimeNoCutUs = 0;  // 不切候选(1 轮)的总通信时
};

// 四、公共逻辑片段（求解器键控分支与后续算子迁移复用的公共逻辑, P2 迁移时优先对齐;
//     2026-09-15 五算子逐行对比提取, 覆盖率注明各行; §4.24 后由键控分支/P2 迁移消费）
// ============================================================================

// 短长块等长归一（5/5 算子 Adjust 尾部逐字一致）
inline void MergeEqualTiles(CutPlanner &tilingM)
{
    if (tilingM.cutRes.shortTileLen == tilingM.cutRes.longTileLen) {
        tilingM.cutRes.shortTileLen = 0U;
        tilingM.cutRes.numShortTile = 0U;
        tilingM.cutRes.numLongTile++;
    }
}

// 线性度好的 shape（k×n 较大, 通信拟合更可信; 4/5 算子一致, agmm 传字面 true,
// allto_all_matmul 自有离散切分不用）
inline bool IsGoodLinearityShape(const MatmulShapeInfo &mmInfo)
{
    return mmInfo.shape.kValue * mmInfo.shape.nValue >= GOOD_LINEARITY_NK_BYTES * BYTES_PER_MB;
}

// 双向配平长块（3/5 算子一致: mmrs / all_reduce / matmul_allto_all）:
//   短块后置(计算 bound) → 长块 = 反解通信时长(掩盖短块计算);
//   短块前置(通信 bound) → 长块 = 反解计算时长(掩盖短块通信)。
// agmm 不用(其长块掩盖的是 frontMMTime 本地首轮计算); allto_all_matmul 自有模型。
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

// ---- 以下为多算子共有(非全算子)片段: 求解器键控分支与 P2 迁移算子按需调用
//      (调=走公共逻辑, 不调=走各自键控分支), 不设引擎级 bool 开关——片段与键控
//      分支逻辑交错, 引擎级开关无法表达交错顺序且会重演配置汤) ----

// Estimate 骨干（4/5: agmm/mmrs/all_reduce/matmul_allto_all 一致）:
// FindCubeUtil + GetMatmulGradient + 总量估计 + 通算比; shortTileAtBack 判定不在此
// (各算子语义相反: mmrs 系 mm>=comm→后置, allto_all_matmul 是 comm>mm→后置, agmm 恒后置)。
// 总量倍数 ctx.mmRankMul 由 InitCtx 按通信原语单次派生(见 MmRankMul 注释)。
// [已废弃 2026-09-17] mmTimeScale/KC_QUANT×1.43 补偿不再保留——与 A2AMM 独立拟合
// 同根因(老切分尾块引发 matmul 膨胀, 理论失准), 新架构不引入该膨胀, 无需补偿;
// MMA2A 的 E14(80μs 不切门)迁移时同样不带 ×1.43。
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalTpTime = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(FitBasedBalanceCtx &ctx)
{
    ctx.matmulPerf.FindCubeUtil(ctx.commRankCntForMm);
    ctx.matmulPerf.GetMatmulGradient();
    EstimateTimesResult r;
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.shape.mValue, ctx.mmRankMul);
    // N=1(NoCut 候选口径, §4.37): Estimate 发生在切分前、轮数未知, 按不切保守估通信总量;
    // 切分完成后由 Schedule 尾部按实际轮数精算(ctx.commTimeActualUs)供日志/观察
    r.totalTpTime = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.shape.mValue, 1);
    ctx.ratioCalcComm = (std::max(r.totalTpTime, r.totalMatmulTime) / std::min(r.totalTpTime, r.totalMatmulTime));
    return r;
}

// L2 占用判定（3/3 逐字一致: mmrs/all_reduce/matmul_allto_all, 含 128MB 同值常量）
inline bool ExceedsL2Cache(const MatmulShapeInfo &mmInfo)
{
    constexpr uint64_t L2_CACHE_SIZE = 128UL * BYTES_PER_MB;
    const uint64_t l2UseSize = mmInfo.shape.mValue * mmInfo.shape.kValue * mmInfo.dtypeSizes.inMatrixA +
                               mmInfo.shape.kValue * mmInfo.shape.nValue * mmInfo.dtypeSizes.inMatrixB;
    return l2UseSize > L2_CACHE_SIZE;
}

// L2 超限鼓励大块: minLen×2（3/5; mmrs 附加 m>minLen×4 守卫 → mGuardMul=4,
// all_reduce/matmul_allto_all 无守卫 → mGuardMul=0; agmm/allto_all_matmul 不用）
inline void ApplyL2MinLenBoost(CutPlanner &tilingM, const MatmulShapeInfo &mmInfo, bool exceedsL2,
                               uint64_t mGuardMul = 0UL)
{
    const bool guardOk = (mGuardMul == 0UL) || (mmInfo.shape.mValue > tilingM.GetMinLen() * mGuardMul);
    if (exceedsL2 && guardOk) {
        tilingM.SetMinLenByMax(tilingM.GetMinLen() * 2U);
    }
}

// 小 shape 整 M 单块装下: minLen 提到 m（2/2 逐字一致: mmrs/matmul_allto_all;
// 判据 m×n ≤ 2×coreNum×baseM×baseN; GetBaseM/N 为非 const 成员故 ctx 取非常量引用）
inline void ApplySmallShapeMinLen(CutPlanner &tilingM, FitBasedBalanceCtx &ctx)
{
    if (ctx.matmulPerf.mmShapeInfo_.shape.mValue * ctx.matmulPerf.mmShapeInfo_.shape.nValue <=
        ctx.matmulPerf.mmShapeInfo_.coreNum * ctx.matmulPerf.GetBaseM() * ctx.matmulPerf.GetBaseN() * 2U) {
        tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.shape.mValue);
    }
}

// 单轮通信量上限整形(CCU 键, §4.37: 自 AGMM 调用点 AdjustCutResultForCCU 四函数链迁入合并):
// HCCL 单次通信调用有 256MB 上限——单块通信量(块长×单行通信字节)超过时按候选表
// {2048,1024,512,256}(大块优先=少轮次)等长重切, 尾块吸收余数, 完全覆盖长短结构
// (超限属极端大 shape, 约束满足优先); 轮次预算 = MAX_COMM_ROUNDS−1(1 轮保留给 scale
// 通道通信, AGMM quant 语义, 与老 HCCL_MAX_TOTAL_TILES=62 一致)。
// 量级安全: 块长×单行字节 ≤ 1e6×5e5 = 5e11 << uint64 上限, 免老代码三段溢出回检;
// 两约束(256MB/轮次预算)冲突时放弃整形保持原切分(老代码 UNSUPPORTED 路径)。
// 流水线总时估计(§4.37.1, 轮数搜索/门控的统一估值器): 切 N 轮的近似总时——不可重叠段在
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
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.shape.mValue;
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
