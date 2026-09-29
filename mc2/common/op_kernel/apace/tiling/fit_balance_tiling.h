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
 * \brief fit-balance tiling 公共引擎——对外总接口(伞形头)
 *
 * 自由长度求解分支: HCCL AICPU/CCU 引擎, 每轮启动开销显著 → 反解最优轮长并压轮数。
 * 消费者为通算融合算子的 fit-balance 路径(950 标卡 4P/8P; 引擎仅支持 950)。
 *
 * 架构: 引擎不区分通信原语——原语语义全部终结在标定层(通信份数口径融进标定斜率 k,
 * 轴向与 mm 倍数在 CommProfile, 流水线结构由 LocalMMPosition 表达); 键控维度仅
 * (LocalMMPosition 本地 mm 块位置, CommComputeOrder 通算顺序, TilingCommEngine
 * 通信引擎, dtype)。算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用 FitBalanceCut。
 *
 * 统一引擎分派(三引擎接入结构一致): CCU/AICPU 走自由长度分支(各引擎拟合表);
 * AIV_URMA 枚举值保留供未来方案接入; 每引擎一组标定常量 + 一张表 + 一个分派点。
 */
// ============================================================================
//  文件拆分: 引擎按功能分层(fit_balance_tiling/ 目录), 消费方
// include 本伞形头即可(路径与拆分前完全一致):
//   fit_balance_types.h  零: 公共类型/枚举/常量
//   comm_model.h         一.1: 通信模型(y=k·x+N·b+c 三口径, 零份数乘子)
//   matmul_model.h       一.2: 计算模型(利用率双口径/梯度/正反解)
//   cut_planner.h        二: 离散切分引擎(CutPlanner 七步)
//   calibration.h        三.1: 客户标定数据总集(CommProfile + k/b/c + 算力档)
//   solver.h             三.2/四: 求解层(ctx/InitCtx/公共片段/Schedule)
// ============================================================================

#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include "fit_balance_tiling/fit_balance_types.h"
#include "fit_balance_tiling/comm_model.h"
#include "fit_balance_tiling/matmul_model.h"
#include "fit_balance_tiling/cut_planner.h"
#include "fit_balance_tiling/calibration.h"
#include "fit_balance_tiling/solver.h"

// 统一入口: 算子侧只传大结构体标定数据 + 键控枚举(原语/顺序/引擎; soc 恒 950)
inline FitBasedBalanceCtx InitCtx(const FitBalanceTilingArgs &args, LocalMMPosition localMMPosition,
                                  CommComputeOrder order, TilingCommEngine commEngine,
                                  const FitBalanceCalibration &calib)
{
    // ---- 标定选行(所有匹配键此刻已知: rankDim 与 dtype 均来自 args, 就地选定传纯参数) ----
    // 通信行: 按引擎分区(aicpu/ccu) + rankDim 匹配首个命中行; miss=零值行
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
    // ,
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
               (row.bBytes == args.inputDtypeSize); // C 侧字节数不进键
    };
    double computesPerCycle = calib.mmFit.defaultComputesPerCycle;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_CUBE_CALC_ROWS; ++i) {
        if (calib.mmFit.cubeCalc[i].computesPerCycle == 0.0) { // 哨兵截断(未填行; 勿用 aBytes——
            break;                                             // glue 真实口径 fp4 行 aBytes=0 合法, 见
        } //
        if (matchesCalcProfile(calib.mmFit.cubeCalc[i])) {
            computesPerCycle = calib.mmFit.cubeCalc[i].computesPerCycle;
            break;
        }
    }
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, computesPerCycle, calib.mmFit.baseBlock),
                           HCCLCommModel(args.rankDim, commFitParams),
                           CutPlanner(args),
                           localMMPosition,
                           order,
                           commEngine,
                           1,
                           1,
                           1,
                           1.0};
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS); // 通信 API FFTS 队列轮次上限
    //  结构参数入口换算(标定层 CommProfile 声明模式, 引擎一处集中换算)
    ctx.commRankCntForMm = RankMulValue(calib.profile.mmWorksetMul, ctx.commPerf.rankDim());
    ctx.mmRankMul = RankMulValue(calib.profile.mmTotalMul, ctx.commPerf.rankDim());
    ctx.commVolumeMul = RankMulValue(calib.profile.commVolumeMul, ctx.commPerf.rankDim());

    // ---- 通算胶水 ----
    // D4 通信矩阵另一维
    const bool commOnKAxis = (calib.profile.axis == CommAxis::K);
    ctx.commPerf.SetCommShapeLen(commOnKAxis ? ctx.matmulPerf.mmShapeInfo_.kValue : ctx.matmulPerf.mmShapeInfo_.nValue);
    // D5+D6 通信承载 dtype 与 fp4 子字节建模:
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
    //    isFp4 维度接管, 不在此处理。op 侧对 fp4 传 inputDtypeSize=1 时此两行为防御冗余
    if (isFp4Input) {
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
    }
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

// 键控维度: LocalMMPosition。
// FRONT: 本地 mm 块在流水线首端(本卡数据已在手, 不等通信)——启动端被本地块免费占据,
//        短块退到尾部专职快速收尾(恒后置), 长块通信掩盖本地首块计算;
// BACK:  本地 mm 块在流水线尾端(本地保留份无需发出、无通信 deadline)——收尾端被本地块
//        占据(与末块通信重叠), 短块前置专职启动通信管道, 长块通信掩盖本地尾块计算;
// NONE:  无本地块——短块作流水线垫片, 按 bound 判定放首/尾。
inline CommCutResult Schedule(FitBasedBalanceCtx &ctx)
{
    const bool isLocalMMFront = (ctx.localMMPosition == LocalMMPosition::FRONT);
    const bool isLocalMMBack = (ctx.localMMPosition == LocalMMPosition::BACK);

    // ---- ① Estimate: 总量估计(通算比/bound 判定; mmRankMul 由 InitCtx 派生) ----
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
    ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
    ctx.tilingM.cutRes.numShortTile = 1U;

    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
        return ctx.tilingM.cutRes;
    }

    // ---- ③ SetLong: 本地窗口配平(FRONT/BACK) / 双向配平(NONE) ----
    if (isLocalMMFront) {
        // 首长块的通信恰好掩盖本地首块计算
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.frontMMTime);
    } else if (isLocalMMBack) {
        // 末长块的通信恰好掩盖本地尾块计算(本地保留份在收尾端与末块通信重叠)
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.backMMTime);
    } else {
        BalanceLongTileLen(ctx);
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

    // ---- ⑤ 轮数搜索 + 切分收益门控: 把每轮启动开销纳入切分决策 ----
    // 切分不改变总通信数据量(k·X 不变), 只多付启动税 (N−1)·b, 买到的是通信-计算重叠。
    // 流水线总时近似(不可重叠段在流水线两端, 按 LocalMMPosition/顺序三形态, 见 EstimateCutTotalTime):
    //   FRONT(本地首块): T_cut ≈ T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
    //   BACK(本地尾块):  T_cut ≈ T_mm(首块) + max(T_mm(其余), T_comm(除末轮)) + max(T_comm(末轮), 本地窗口)
    //   NONE(通信在后):  T_cut ≈ T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)
    // 不切 T_serial = T_comm(1 轮) + T_mm(全串行)。
    // 搜索动机: 配平产出的轮数 N₀ 是"配平最优"而非"全局最优"——
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

    // ---- ⑥ 单轮通信量上限整形: HCCL 单次调用 256MB 上限, 超限块等长重切 ----
    if (ctx.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(ctx.tilingM, ctx.commPerf, ctx.commVolumeMul, ctx.matmulPerf.mmShapeInfo_.mValue);
    }

    // : 按最终切分的实际轮数精算总通信时(N 轮 vs NoCut 候选 1 轮), 差值即切分
    // 多付的启动税 (N−1)·b——诊断输出供 op 侧日志
    ctx.commTimeActualUs =
        ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, ctx.tilingM.cutRes.totalTileCnt);
    ctx.commTimeNoCutUs = ctx.commPerf.TotalCommTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
    return ctx.tilingM.cutRes;
}

// ============================================================================
// 诊断打印(日志级别为 debug 即 ASCEND_GLOBAL_LOG_LEVEL=0 时输出, 纯 printf 零日志库依赖):
// 输入 shape / 最终切分 / 启动税(N 轮 vs 不切的总通信时之差 = (N−1)·b)。
inline void PrintTilingData(const FitBalanceTilingArgs &args, const CommCutResult &cut, double commTimeActualUs,
                            double commTimeNoCutUs)
{
    const char *level = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (level == nullptr || level[0] != '0') {
        return;
    }
    printf("[FitBalance Input]\n");
    printf("  mValue             : %lu\n", args.mValue);
    printf("  nValue             : %lu\n", args.nValue);
    printf("  kValue             : %lu\n", args.kValue);
    printf("  rankDim            : %u\n", args.rankDim);
    printf("[FitBalance Cut]\n");
    printf("  totalTileCnt       : %lu\n", cut.totalTileCnt);
    printf("  longTileLen        : %lu\n", cut.longTileLen);
    printf("  numLongTile        : %lu\n", cut.numLongTile);
    printf("  shortTileLen       : %lu\n", cut.shortTileLen);
    printf("  numShortTile       : %lu\n", cut.numShortTile);
    printf("  shortTileAtBack    : %d\n", cut.shortTileAtBack);
    printf("[FitBalance CommTime]\n");
    printf("  actual(%lu rounds) : %.3f us\n", cut.totalTileCnt, commTimeActualUs);
    printf("  no-cut(1 round)    : %.3f us\n", commTimeNoCutUs);
    printf("  round tax          : %.3f us\n", commTimeActualUs - commTimeNoCutUs);
}

// 对外唯一入口: 算子侧一行调用——上下文构造 + 键控求解的串联。
// 诊断打印内置于本函数(debug 级), op 侧无需再写日志。
inline CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, LocalMMPosition localMMPosition,
                                   CommComputeOrder order, TilingCommEngine commEngine,
                                   const FitBalanceCalibration &calib)
{
    FitBasedBalanceCtx ctx = InitCtx(args, localMMPosition, order, commEngine, calib);
    CommCutResult cut = Schedule(ctx);
    PrintTilingData(args, cut, ctx.commTimeActualUs, ctx.commTimeNoCutUs);
    return cut;
}

#endif // __APACE_FIT_BALANCE_TILING_H__
