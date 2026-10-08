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
 * \brief fit-balance tiling 公共引擎——对外总接口头文件
 *
 * 自由长度求解分支: HCCL AICPU/CCU 引擎, 每轮启动开销显著 → 反解最优轮长并压轮数。
 * 消费者为通算融合算子的 fit-balance 路径(950 标卡 4P/8P; 引擎仅支持 950)。
 *
 * 架构: 引擎不区分通信原语——原语语义全部终结在标定层(通信份数口径融进标定斜率 k,
 * 轴向与 mm 倍数在 CommProfile, 流水线结构由 LocalMMPosition 表达); 键控维度仅
 * (LocalMMPosition 本地 mm 块位置, CommComputeOrder 通算顺序, TilingCommEngine
 * 通信引擎, dtype)。算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用 FitBalanceCut。
 *
 * 引擎接入: 每引擎一组标定行(新增引擎加行即可)——标定命中即可走本引擎(AIV_URMA 同构,
 * 标定后视情况接入); miss 的处理由算子侧自选, 本引擎不预设兜底策略。
 */
// ============================================================================
//  文件拆分: 引擎按功能分层(fit_balance_tiling/ 目录), 消费方
// include 本头文件即可(路径与拆分前完全一致):
//   fit_balance_types.h  零: 公共类型/枚举/常量 + FitBalanceCalibration 标定总集
//   comm_model.h         一.1: 通信模型(y=k·x+N·b+c 三口径, 零份数乘子)
//   matmul_model.h       一.2: 计算模型(利用率双口径/梯度/正反解)
//   cut_planner.h        二: 离散切分引擎(CutPlanner 七步)
//   solver.h             三: 求解层(ctx/InitCtx/公共片段/Schedule)
// ============================================================================

#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include "fit_balance_tiling_utils/fit_balance_types.h"
#include "fit_balance_tiling_utils/comm_model.h"
#include "fit_balance_tiling_utils/matmul_model.h"
#include "fit_balance_tiling_utils/cut_planner.h"
#include "fit_balance_tiling_utils/solver.h"
#include "../tiling_data_struct_define.h" // CommTilingData(CutResultFromCof 的入参类型, 核利用率优先结果)

// 统一入口: 算子侧只传大结构体标定数据 + 键控枚举(原语/顺序/引擎; soc 恒 950)
// 通信标定行查找: 按 (通信引擎, rankDim) 二维匹配首个命中行; miss 返回 nullptr。
// rankDim=0 为未填行哨兵(截断); 零斜率行同样按未填处理(物理上斜率不可能为 0,
// 占位/换算默认值一律不构成标定命中——未标定形态 miss 走其他切分策略)。
inline const FitBalanceCalibration::CommFitRow* FindCommFitRow(const FitBalanceCalibration& calib,
                                                               TilingCommEngine commEngine, uint32_t rankDim)
{
    for (size_t i = 0; i < FitBalanceCalibration::MAX_COMM_FIT_ROWS; ++i) {
        const auto& row = calib.commRows[i];
        if (row.rankDim == 0) {
            break;
        }
        if (row.commEngine == commEngine && row.rankDim == rankDim && row.slopeUsPerMb != 0.0) {
            return &row;
        }
    }
    return nullptr;
}

// 标定命中判定(算子查询): 命中即可走本引擎; miss 的处理由算子侧自选
inline bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration& calib)
{
    return FindCommFitRow(calib, commEngine, rankDim) != nullptr;
}

// 数据流形态推导(标定栏 dataflow 声明, 引擎内部换算——算子侧不再传结构枚举):
//   commOnKAxis:  通信在前(COMM_FIRST)=传输入 A 侧 → K 轴; 在后 = 传输出 C 侧 → N 轴
//   rowMul / commVolumeMul: 全量形态(切分域=本卡份, 实算 mm 行数与单次通信字节均为
//                          切分域×rankDim)=rankDim; 单份形态=1
//   worksetShares: FRONT(本地块前置——本地份剔出稳态、单独按单份建模)=rankDim−1; 其余随形态
//   注: A2A 系老口径 workset=rankDim 与单份形态推导值(1)不同——A2A 系无现存消费者,
//       BACK 形态实测校准时定稿
inline FitBasedBalanceCtx InitCtx(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    const CommComputeOrder order = calib.dataflow.order;
    const LocalMMPosition localMMPosition = calib.dataflow.localMMPosition;
    // ---- 标定选行(所有匹配键此刻已知: rankDim 与 dtype 均来自 args, 就地选定传纯参数) ----
    const auto* commRow = FindCommFitRow(calib, args.commEngine, args.rankDim);
    CommFitParams commFitParams = (commRow != nullptr) ? CommFitParams{commRow->slopeUsPerMb, commRow->roundOverheadUs,
                                                                       commRow->fixedOverheadUs} :
                                                         CommFitParams{}; // miss=零值行(slope=0 → NoCut 防御)
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
    const auto matchesCalcProfile = [&](const auto& row) {
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
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, computesPerCycle, calib.mmFit.baseBlock, calib.mmFit.cyclePerMicroSec,
                                           calib.mmFit.l2CacheBytes),
                           HCCLCommModel(args.rankDim, commFitParams),
                           CutPlanner(args),
                           localMMPosition,
                           order,
                           args.commEngine,
                           1,
                           1,
                           1,
                           1.0};
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS); // 通信 API FFTS 队列轮次上限
    // 数据流形态推导出的三个倍数(规则见函数头注释)
    const uint64_t volumeMul = calib.dataflow.fullVolume ? ctx.commPerf.rankDim() : 1U;
    ctx.commRankCntForMm = (localMMPosition == LocalMMPosition::FRONT) ? (ctx.commPerf.rankDim() - 1U) : volumeMul;
    ctx.mmRankMul = volumeMul;
    ctx.commVolumeMul = volumeMul;

    // ---- 通算胶水 ----
    // D4 通信矩阵另一维(承载侧由通算顺序推导: 通信在前=传输入 A 侧, 在后=传输出 C 侧)
    const bool commOnKAxis = (order == CommComputeOrder::COMM_FIRST);
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
    // NoCut 门控即输出单块不切(防御行为, 正常接入下 op 层已按自选策略处理 miss, 不可达;
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
inline CommCutResult Schedule(FitBasedBalanceCtx& ctx)
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

    // ---- ② SetShort: 短块下限闸门 + 落位(最短可切校验在落位后: 总长装不下两块下限则整块不切) ----
    // L2 维度: 工作集超 950 L2 容量 → minLen×2 鼓励大块
    ApplyL2MinLenBoost(ctx.tilingM, ExceedsL2Cache(ctx.matmulPerf.mmShapeInfo_));
    // 启动开销维度: 短块独立占一轮通信, 其通信时长须 ≥ 2b(一轮自身开销 + 至少等量的
    // 有效传输)才值得成轮——由 2b 反解行数下限, 小 k 场景短块被抬大或经 NoCut 门不切
    ctx.tilingM.SetMinLenByMax(ctx.commPerf.InverseCommTime(2.0 * ctx.commPerf.commEstimatePar_.roundOverheadUs));
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
    const uint64_t initialTileCnt = bestCut.totalTileCnt; // 搜索上界 N₀ 固定: bestCut 在循环内会被
                                                          // 更优候选替换(轮数可降), 不能充当循环界
    for (uint64_t budget = 2U; budget < initialTileCnt; ++budget) {
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
inline void PrintTilingData(const FitBalanceTilingArgs& args, const CommCutResult& cut, double commTimeActualUs,
                            double commTimeNoCutUs)
{
    const char* level = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
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

// 核利用率优先切分结果(head/tail 形态) → 引擎切分结果(1 短 + N 长)转换:
// head 为主体块、tail 为尾块 → 长块/短块; 尾块天然后置。
// 供同时使用两种切分策略的算子统一输出形态(是否混用由算子开发者决定)。
inline CommCutResult CutResultFromCof(const CommTilingData& data)
{
    CommCutResult r;
    r.shortTileAtBack = true;
    r.longTileLen = data.splitAxisTileSize;
    r.numLongTile = data.splitAxisTileCnt;
    r.shortTileLen = data.splitAxisTailSize;
    r.numShortTile = data.splitAxisTailCnt;
    r.totalTileCnt = data.splitAxisTileCnt + data.splitAxisTailCnt;
    return r;
}

// 旧入参转换(存量算子过渡用): 从旧全局 mc2tiling::TilingArgs 挑字段构造引擎入参。
// commEngine 旧结构无对应字段(算子图属性 comm_mode), 由调用方解析后经 commEngine 参数补入。
// 新算子直接构造 FitBalanceTilingArgs, 无需本函数。
template <typename LegacyArgs>
inline FitBalanceTilingArgs MakeFitBalanceArgs(const LegacyArgs& a, TilingCommEngine commEngine)
{
    FitBalanceTilingArgs e = {};
    e.mValue = a.mValue;
    e.nValue = a.nValue;
    e.kValue = a.kValue;
    e.rankDim = a.rankDim;
    e.aicCoreNum = a.aicCoreNum;
    e.aType = a.aType;
    e.geAType = a.geAType;
    e.inputDtypeSize = a.inputDtypeSize;
    e.outputDtypeSize = a.outputDtypeSize;
    e.commEngine = commEngine;
    return e;
}

// 对外唯一入口: 算子侧一行调用——上下文构造 + 键控求解的串联。
// 诊断打印内置于本函数(debug 级), op 侧无需再写日志。
inline CommCutResult FitBalanceCut(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    FitBasedBalanceCtx ctx = InitCtx(args, calib);
    CommCutResult cut = Schedule(ctx);
    PrintTilingData(args, cut, ctx.commTimeActualUs, ctx.commTimeNoCutUs);
    return cut;
}

#endif // __APACE_FIT_BALANCE_TILING_H__
