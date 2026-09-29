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

// 统一入口: 算子侧只传大结构体标定数据 + 键控枚举(原语/顺序/引擎; soc 恒 950)
inline FitBasedBalanceCtx InitCtx(const FitBalanceTilingArgs &args, LocalMMPosition localMMPosition,
                                  CommComputeOrder order, TilingCommEngine commEngine,
                                  const FitBalanceCalibration &calib,
                                  CommTopology topoType = CommTopology::STANDARD_CARD)
{
    // ---- 标定选行(所有匹配键此刻已知: rankDim 与 dtype 均来自 args, 就地选定传纯参数) ----
    // 通信行: 按引擎分区(aicpu/ccu) + rankDim 匹配首个命中行; miss=零值行(§4.29 防御 NoCut)
    const auto &commRows = ((commEngine == TilingCommEngine::CCU) ? calib.ccu : calib.aicpu).rows;
    CommFitParams commFitParams = {};
    for (size_t i = 0; i < FitBalanceCalibration::MAX_COMM_FIT_ROWS; ++i) {
        if (commRows[i].rankDim == 0) { // 哨兵截断
            break;
        }
        if (commRows[i].rankDim == args.rankDim) {
            commFitParams = {commRows[i].slopeUsPerMb, commRows[i].roundOverheadUs, commRows[i].fixedOverheadUs};
            break;
        }
    }
    // matmul 行: 按"计算形态"匹配算力行——形态键 = (量化类别, fp4 打包, A/B 字节数) 四维
    // (C 侧字节数不进键, §4.37 根治裁定: glue cType=aType 使其恒等于 aBytes, 零信息量),
    // 键推导与模型 SetCalcType 同源
    const auto isQuant = (args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
                         (args.aType == matmul_tiling::DataType::DT_INT8) ||
                         (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const MatmulDtypeKind calcType = isQuant ? MatmulDtypeKind::QUANT : MatmulDtypeKind::FP16;
    const bool isFp4 = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const auto matchesCalcProfile = [&](const auto &row) {
        return (row.calcType == calcType) && (row.isFp4 == isFp4) && (row.aBytes == args.inputDtypeSize) &&
               (row.bBytes == args.inputDtypeSize); // C 侧字节数不进键(§4.37 根治裁定, 见 CubeCalc 注释)
    };
    double computesPerCycle = calib.mmFit.defaultComputesPerCycle;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_CUBE_CALC_ROWS; ++i) {
        if (calib.mmFit.cubeCalc[i].computesPerCycle == 0.0) { // 哨兵截断(未填行; 勿用 aBytes——
            break;                                             // glue 真实口径 fp4 行 aBytes=0 合法, 见
        } // AGMM_QUANT_CPC_KEY_DEFECT_AND_FIX.md)
        if (matchesCalcProfile(calib.mmFit.cubeCalc[i])) {
            computesPerCycle = calib.mmFit.cubeCalc[i].computesPerCycle;
            break;
        }
    }
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, computesPerCycle, calib.mmFit.baseBlock),
                           HCCLCommModel(args.rankDim, commFitParams, topoType),
                           CutPlanner(args),
                           localMMPosition,
                           order,
                           commEngine,
                           1,
                           1,
                           1,
                           1.0};
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS); // 通信 API FFTS 队列轮次上限(2026-09-17: 63)
    // §4.38 结构参数入口换算(标定层 CommProfile 声明模式, 引擎一处集中换算)
    ctx.commRankCntForMm = RankMulValue(calib.profile.mmWorksetMul, ctx.commPerf.rankDim());
    ctx.mmRankMul = RankMulValue(calib.profile.mmTotalMul, ctx.commPerf.rankDim());
    ctx.commVolumeMul = RankMulValue(calib.profile.commVolumeMul, ctx.commPerf.rankDim());

    // ---- 通算胶水(键控, 原散在各算子构造函数, §4.24 裁定随引擎下沉) ----
    // D4 通信矩阵另一维(§4.38: 轴向由标定层 CommProfile.axis 声明——A 侧承载沿 K 轴, C 侧沿 N 轴)
    const bool commOnKAxis = (calib.profile.axis == CommAxis::K);
    ctx.commPerf.SetCommShapeLen(commOnKAxis ? ctx.matmulPerf.mmShapeInfo_.kValue : ctx.matmulPerf.mmShapeInfo_.nValue);
    // D5+D6 通信承载 dtype 与 fp4 子字节建模(两件独立的事, 原耦合在 AG 构造函数单分支内, §4.33 解耦):
    // ① 通信 dtype 按 CommProfile.axis 承载侧: K(A 侧输入)取 A 字节, N(C 侧输出)取 C 字节——
    //    fp4 只可能是输入 A 的 dtype, C 侧(计算输出)永远不是 fp4, 故子字节建模仅 K 轴承载时需要;
    // ② fp4(0.5B/元素)无法用 uint64 表达: 通信侧按"1 字节打包 2 元素"建模(dtypeSize=1+
    //    fraction=2 → GetRealDtypeSizes()=0.5, 复用 HCCL 子字节机制)
    const bool isFp4Input = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    if (commOnKAxis) {
        if (isFp4Input) {
            ctx.commPerf.SetCommDTypeSize(1);
            ctx.commPerf.SetCommDtypeSizeExpansionFraction(FP4_PACK_RATIO);
        } else {
            ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize);
        }
    } else {
        ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.outMatrixCDtypeSize);
    }
    // ③ mm 侧 fp4 口径(与通信承载哪侧无关, 故独立于上分支): L2 工作集估算按 1B/元素保守取值
    //    (真实 0.5B 无法表达, 2x 高估防 memUsage 退化); 算力查表已由 matchesCalcProfile 的
    //    isFp4 维度接管(§4.17), 不在此处理。op 侧对 fp4 传 inputDtypeSize=1 时此两行为防御冗余
    if (isFp4Input) {
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
    }
    // §4.29 分派后引擎不设回退模型: 标定未命中(零值行 slope=0)时置 minLen=mValue——
    // NoCut 门控即输出单块不切(防御行为, 正常分派下 op 层已引走 COF, 不可达;
    // 同时避免 InverseCommTime 除零斜率的 float→uint UB)
    if (ctx.commPerf.commEstimatePar_.slopeUsPerMb == 0.0) {
        ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
    }
    // D7 离散对齐粒度初始化: baseM
    ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.GetBaseM());
    ctx.tilingM.SetAlignLength(ctx.matmulPerf.GetBaseM());
    return ctx;
}

// ============================================================================
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
    return mmInfo.kValue * mmInfo.nValue >= GOOD_LINEARITY_NK_BYTES * BYTES_PER_MB;
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
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.mmRankMul);
    // N=1(NoCut 候选口径, §4.37): Estimate 发生在切分前、轮数未知, 按不切保守估通信总量;
    // 切分完成后由 Schedule 尾部按实际轮数精算(ctx.commTimeActualUs)供日志/观察
    r.totalTpTime = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    ctx.ratioCalcComm = (std::max(r.totalTpTime, r.totalMatmulTime) / std::min(r.totalTpTime, r.totalMatmulTime));
    return r;
}

// L2 占用判定（3/3 逐字一致: mmrs/all_reduce/matmul_allto_all, 含 128MB 同值常量）
inline bool ExceedsL2Cache(const MatmulShapeInfo &mmInfo)
{
    constexpr uint64_t L2_CACHE_SIZE = 128UL * BYTES_PER_MB;
    const uint64_t l2UseSize = mmInfo.mValue * mmInfo.kValue * mmInfo.inMatrixADtypeSize +
                               mmInfo.kValue * mmInfo.nValue * mmInfo.inMatrixBDtypeSize;
    return l2UseSize > L2_CACHE_SIZE;
}

// L2 超限鼓励大块: minLen×2（3/5; mmrs 附加 m>minLen×4 守卫 → mGuardMul=4,
// all_reduce/matmul_allto_all 无守卫 → mGuardMul=0; agmm/allto_all_matmul 不用）
inline void ApplyL2MinLenBoost(CutPlanner &tilingM, const MatmulShapeInfo &mmInfo, bool exceedsL2,
                               uint64_t mGuardMul = 0UL)
{
    const bool guardOk = (mGuardMul == 0UL) || (mmInfo.mValue > tilingM.GetMinLen() * mGuardMul);
    if (exceedsL2 && guardOk) {
        tilingM.SetMinLenByMax(tilingM.GetMinLen() * 2U);
    }
}

// 小 shape 整 M 单块装下: minLen 提到 m（2/2 逐字一致: mmrs/matmul_allto_all;
// 判据 m×n ≤ 2×coreNum×baseM×baseN; GetBaseM/N 为非 const 成员故 ctx 取非常量引用）
inline void ApplySmallShapeMinLen(CutPlanner &tilingM, FitBasedBalanceCtx &ctx)
{
    if (ctx.matmulPerf.mmShapeInfo_.mValue * ctx.matmulPerf.mmShapeInfo_.nValue <=
        ctx.matmulPerf.mmShapeInfo_.coreNum * ctx.matmulPerf.GetBaseM() * ctx.matmulPerf.GetBaseN() * 2U) {
        tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
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

// 键控维度: LocalMMPosition(§4.38——取代通信原语的结构键控)。
// FRONT: 本地 mm 块在流水线首端(本卡数据已在手, 不等通信)——启动端被本地块免费占据,
//        短块退到尾部专职快速收尾(恒后置), 长块通信掩盖本地首块计算(E5/E19, 原 AG 分支);
// BACK:  本地 mm 块在流水线尾端(本地保留份无需发出、无通信 deadline)——收尾端被本地块
//        占据(与末块通信重叠), 短块前置专职启动通信管道, 长块通信掩盖本地尾块计算(镜像 E5/E19);
// NONE:  无本地块——短块作流水线垫片, 按 bound 判定放首/尾(E4 通用规律)。
inline CommCutResult Schedule(FitBasedBalanceCtx &ctx)
{
    const bool isLocalMMFront = (ctx.localMMPosition == LocalMMPosition::FRONT);
    const bool isLocalMMBack = (ctx.localMMPosition == LocalMMPosition::BACK);

    // ---- ① Estimate: 公共骨干(E1) + E2 总量倍数(ctx.mmRankMul, InitCtx 已派生) ----
    const auto estimate = EstimateTotalTimes(ctx);
    if (isLocalMMFront || isLocalMMBack) {
        // 本地块独立计算的 L2 利用率与稳态不同(单份数据、流水未满, 0.6 指数口径;
        // FRONT=首端 E5, BACK=尾端镜像——同一物理量: 本地窗口的计算时长)
        const double localMMUtil = ctx.matmulPerf.FindCubeUtil(ctx.matmulPerf.mmShapeInfo_.mValue, 1,
                                                               MatmulPerfConst::LOCAL_MM_UTIL_EXPONENT_COEFFICIENT);
        const double localMMTime = (std::abs(localMMUtil) > 1e-10) ?
                                       ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1) / localMMUtil :
                                       ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        if (isLocalMMFront) {
            ctx.frontMMTime = localMMTime;
            ctx.tilingM.cutRes.shortTileAtBack = true; // 本地块占启动端, 短块专职收尾(恒后置)
        } else {
            ctx.backMMTime = localMMTime;
            ctx.tilingM.cutRes.shortTileAtBack = false; // 本地块占收尾端, 短块专职启动(恒前置)
        }
    } else if (ctx.order == CommComputeOrder::COMPUTE_FIRST) {
        // 先行段为瓶颈 → 短块后置(短块放另一端, 由长块掩盖)
        ctx.tilingM.cutRes.shortTileAtBack = (estimate.totalMatmulTime >= estimate.totalTpTime);
    } else {
        ctx.tilingM.cutRes.shortTileAtBack = (estimate.totalTpTime > estimate.totalMatmulTime);
    }

    // ---- ② SetShort: 短块落位(minTileLen 为短块长度种子; NoCut 几何门在落位后) ----
    // [P2 接线] E11-E13(L2/小 shape)随各算子迁移接入; §4.37 废弃条目(E9/E10/E14-E16/E20/E26/E29)不再接线
    ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
    ctx.tilingM.cutRes.numShortTile = 1U;

    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
        return ctx.tilingM.cutRes;
    }

    // ---- ③ SetLong: 本地窗口配平(FRONT/BACK) / E18 公共双向配平(NONE) ----
    if (isLocalMMFront) {
        // 首长块的通信恰好掩盖本地首块计算(E19)
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.frontMMTime);
    } else if (isLocalMMBack) {
        // 末长块的通信恰好掩盖本地尾块计算(镜像 E19: 本地保留份在收尾端与末块通信重叠)
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.backMMTime);
    } else {
        BalanceLongTileLen(ctx); // 乘数 ctx.mmRankMul(老代码 A2A 曾误用管线系数 rankTileNum)
    }
    // ⑤ 轮数搜索的恢复点: ③ 后形态(短块落位 + 长块反解值)。反解与轮数无关——
    // 每档仅在收紧的轮次预算下重跑 ④ 离散化, 无需重算配平
    const CommCutResult afterLong = ctx.tilingM.cutRes;

    // ---- ④ Adjust: E24(goodLinearity: AG 恒 true) + 离散化 + E25 等长归一 ----
    // 有本地块的形态恒信线性段(行为基线, 原 AG 恒 true); 离散化保护含本地块的端部结构
    const bool goodLinearity =
        (isLocalMMFront || isLocalMMBack) ? true : IsGoodLinearityShape(ctx.matmulPerf.mmShapeInfo_);
    ctx.tilingM.FitTileLengthDiscrete(goodLinearity, isLocalMMFront || isLocalMMBack);
    MergeEqualTiles(ctx.tilingM);

    // ---- ⑤ 轮数搜索 + 切分收益门控(§4.37.1): 把每轮启动开销纳入切分决策 ----
    // 切分不改变总通信数据量(k·X 不变), 只多付启动税 (N−1)·b, 买到的是通信-计算重叠。
    // 流水线总时近似(不可重叠段在流水线两端, 按 LocalMMPosition/顺序三形态, 见 EstimateCutTotalTime):
    //   FRONT(本地首块): T_cut ≈ T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
    //   BACK(本地尾块):  T_cut ≈ T_mm(首块) + max(T_mm(其余), T_comm(除末轮)) + max(T_comm(末轮), 本地窗口)
    //   NONE(通信在后):  T_cut ≈ T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)
    // 不切 T_serial = T_comm(1 轮) + T_mm(全串行)。
    // 搜索动机(2026-09-21 用户指出): 配平产出的轮数 N₀ 是"配平最优"而非"全局最优"——
    // 通信重时 T(N) 随轮数单调增(启动税吃收益)、计算重时单调降(至首块通信≈b 触底),
    // 最优轮数常在 [1, N₀] 中间, 二值判定(N₀ vs 1)会错过中间解。
    // 搜索实现: budget 线性枚举 [2, N₀)(每档恢复 ③ 后形态 + 收紧轮次预算 + 重跑 ④,
    // 长短结构随预算重新整形), T_cut 择优; 全部候选 ≥ T_serial → NoCut。
    // 成本: N₀ ≤ MAX_COMM_ROUNDS(63), 每档纯整数重切 O(块数)——host 侧微秒级。
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.mValue;
    const double serialTotalTime = ctx.commPerf.TotalCommTime(mValue, 1U) + estimate.totalMatmulTime;
    CommCutResult bestCut = ctx.tilingM.cutRes; // 初次切分(N₀ 轮, 配平产出)即候选
    double bestTime = EstimateCutTotalTime(ctx, bestCut, estimate.totalMatmulTime, serialTotalTime);
    for (uint64_t budget = 2U; budget < bestCut.totalTileCnt; ++budget) {
        ctx.tilingM.cutRes = afterLong;    // 恢复 ③ 后形态(反解值复用)
        ctx.tilingM.SetMaxTileCnt(budget); // 收紧轮次预算
        ctx.tilingM.FitTileLengthDiscrete(goodLinearity, isLocalMMFront || isLocalMMBack);
        MergeEqualTiles(ctx.tilingM);
        const double cutTime = EstimateCutTotalTime(ctx, ctx.tilingM.cutRes, estimate.totalMatmulTime, serialTotalTime);
        if (cutTime < bestTime) {
            bestTime = cutTime;
            bestCut = ctx.tilingM.cutRes;
        }
    }
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS); // 恢复默认轮次预算
    if (bestTime >= serialTotalTime) {
        ctx.tilingM.NoCutTiling(); // 全部候选不敌串行——启动税买不回重叠收益
    } else {
        ctx.tilingM.cutRes = bestCut;
    }

    // ---- ⑥ 单轮通信量上限整形(CCU 键, §4.37): HCCL 单次调用 256MB 上限, 超限块等长重切 ----
    if (ctx.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(ctx.tilingM, ctx.commPerf, ctx.commVolumeMul, ctx.matmulPerf.mmShapeInfo_.mValue);
    }

    // §4.37: 按最终切分的实际轮数精算总通信时(N 轮 vs NoCut 候选 1 轮), 差值即切分
    // 多付的启动税 (N−1)·b——诊断输出供 op 侧日志
    ctx.commTimeActualUs =
        ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.tilingM.cutRes.totalTileCnt);
    ctx.commTimeNoCutUs = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    return ctx.tilingM.cutRes;
}
