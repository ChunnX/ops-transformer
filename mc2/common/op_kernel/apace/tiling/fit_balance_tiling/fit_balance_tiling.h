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
 * 通信-计算流水线配平切分引擎。引擎不区分通信原语，键控维度为
 * (LocalMMPosition, CommComputeOrder, TilingCommEngine, dtype)；
 * 算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用 FitBalanceCut。
 *
 * 分文件布局(消费方只 include 本头文件):
 *   fit_balance_tiling_utils/fit_balance_types.h  公共类型/枚举/常量/标定结构
 *   fit_balance_tiling_utils/comm_model.h         通信模型 y = k·x + 轮数·b + c
 *   fit_balance_tiling_utils/matmul_model.h       计算模型(利用率/梯度)
 *   fit_balance_tiling_utils/cut_planner.h        离散切分 CutPlanner
 *   fit_balance_tiling_utils/solver.h             求解层(ctx/InitCtx/Schedule)
 */

#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include "fit_balance_tiling_utils/fit_balance_types.h"
#include "fit_balance_tiling_utils/comm_model.h"
#include "fit_balance_tiling_utils/matmul_model.h"
#include "fit_balance_tiling_utils/cut_planner.h"
#include "fit_balance_tiling_utils/solver.h"
#include "../tiling_data_struct_define.h" // CommTilingData, CutResultFromCof 的入参类型

// 通信标定行查找: 按 (通信引擎, rankDim) 匹配首个命中行, miss 返回 nullptr。
// rankDim=0 为未填行哨兵; 零斜率行同样按未填处理。
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

// 标定命中判定(算子分派): 命中 → 本引擎; miss → 核利用率优先切分兜底
// (core_occupancy_first_tiling.h; 算子有更合适策略时可替换)
inline bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration& calib)
{
    return FindCommFitRow(calib, commEngine, rankDim) != nullptr;
}

// 上下文构造: 标定选行(通信行 + 算力档) + 数据流形态推导(轴/倍数) + 通信口径设置。
inline FitBasedBalanceCtx InitCtx(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    const CommComputeOrder order = calib.dataflow.order;
    const LocalMMPosition localMMPosition = calib.dataflow.localMMPosition;
    const auto* commRow = FindCommFitRow(calib, args.commEngine, args.rankDim);
    // miss = 零值行(slope=0), 后续 NoCut 防御兜底
    CommFitParams commFitParams = (commRow != nullptr) ? CommFitParams{commRow->slopeUsPerMb, commRow->roundOverheadUs,
                                                                       commRow->fixedOverheadUs} :
                                                         CommFitParams{};
    // 算力档按 (量化类别, isFp4, A/B 字节数) 四维匹配, 键推导与 SetCalcType 同源
    const auto isQuant = (args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
                         (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
                         (args.aType == matmul_tiling::DataType::DT_INT8) ||
                         (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const MatmulDtypeKind calcType = isQuant ? MatmulDtypeKind::QUANT : MatmulDtypeKind::FP16;
    const bool isFp4 = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    const auto matchesCalcProfile = [&](const auto& row) {
        return (row.calcType == calcType) && (row.isFp4 == isFp4) && (row.aBytes == args.inputDtypeSize) &&
               (row.bBytes == args.inputDtypeSize);
    };
    double computesPerCycle = calib.mmFit.defaultComputesPerCycle;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_CUBE_CALC_ROWS; ++i) {
        // 哨兵判 computesPerCycle==0: fp4 行 aBytes=0 合法, 不能用 aBytes 判空
        if (calib.mmFit.cubeCalc[i].computesPerCycle == 0.0) {
            break;
        }
        if (matchesCalcProfile(calib.mmFit.cubeCalc[i])) {
            computesPerCycle = calib.mmFit.cubeCalc[i].computesPerCycle;
            break;
        }
    }
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, computesPerCycle, calib.mmFit.baseBlock, calib.mmFit.cyclePerMicroSec),
                           HCCLCommModel(args.rankDim, commFitParams),
                           CutPlanner(args),
                           localMMPosition,
                           order,
                           args.commEngine,
                           1,
                           1,
                           1};
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS);
    // 数据流形态推导三个倍数:
    //   fullVolume=true(mValue 为每卡份) → ×rankDim; false(mValue 已是全量) → ×1
    //   FRONT 的稳态工作集剔出本地份(单独按单份建模) → rankDim−1
    const uint64_t volumeMul = calib.dataflow.fullVolume ? ctx.commPerf.rankDim() : 1U;
    ctx.commRankCntForMm = (localMMPosition == LocalMMPosition::FRONT) ? (ctx.commPerf.rankDim() - 1U) : volumeMul;
    ctx.mmRankMul = volumeMul;
    ctx.commVolumeMul = volumeMul;

    // 通信承载侧由通算顺序推导: 通信在前=传输入 A 侧(K 轴), 在后=传输出 C 侧(N 轴)
    const bool commOnKAxis = (order == CommComputeOrder::COMM_FIRST);
    ctx.commPerf.SetCommShapeLen(commOnKAxis ? ctx.matmulPerf.mmShapeInfo_.kValue : ctx.matmulPerf.mmShapeInfo_.nValue);
    // fp4(0.5B/元素)无法用整数字节表达: 按 1 字节打包 2 元素建模(dtypeSize=1 + fraction=2)
    const bool isFp4Input = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    if (commOnKAxis) {
        if (isFp4Input) {
            ctx.commPerf.SetCommDTypeSize(1);
            ctx.commPerf.SetCommDtypeSizeExpansionFraction(FP4_PACK_RATIO);
        } else {
            ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize);
        }
    } else {
        // C 侧(计算输出)永远不是 fp4
        ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.outMatrixCDtypeSize);
    }
    // mm 侧 fp4 口径(独立于通信承载侧): L2 估算按 1B/元素保守取值
    if (isFp4Input) {
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
    }
    // 标定 miss 防御: slope=0 时抬 minLen 到全量, 后续最短可切校验兜底 NoCut;
    // 同时避免 InverseCommTime 除零
    if (ctx.commPerf.commEstimatePar_.slopeUsPerMb == 0.0) {
        ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
    }
    ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.GetBaseM());
    ctx.tilingM.SetAlignLength(ctx.matmulPerf.GetBaseM());
    return ctx;
}

// ============================================================================
// 求解主流程 Schedule: 六段流水
//   ① Estimate: 总量估计, 决定短块位置
//   ② SetShort: 短块下限闸门 + 落位
//   ③ SetLong:  长块配平反解
//   ④ Adjust:   离散化整形
//   ⑤ 轮数搜索: 切分收益判定, 全败则 NoCut
//   ⑥ CCU 整形: 单轮 256MB 上限
// ============================================================================

// 键控维度: LocalMMPosition——本地 mm 块在流水线中的位置(不经通信的数据块)。
//   FRONT: 本地块在流水线首端(本卡数据已在手), 短块退到尾部收尾;
//   BACK:  本地块在流水线尾端(本地保留份无需发出), 短块前置启动通信;
//   NONE:  无本地块, 短块按 bound 判定放首/尾。
inline CommCutResult Schedule(FitBasedBalanceCtx& ctx)
{
    const bool isLocalMMFront = (ctx.localMMPosition == LocalMMPosition::FRONT);
    const bool isLocalMMBack = (ctx.localMMPosition == LocalMMPosition::BACK);

    // ---- ① Estimate: 总量估计 + 短块位置判定 ----
    const auto estimate = EstimateTotalTimes(ctx);
    if (isLocalMMFront || isLocalMMBack) {
        // 本地窗口计算时长(单份数据/流水未满, 0.6 指数比稳态 0.7 更保守)
        const double localMMUtil = ctx.matmulPerf.FindCubeUtil(ctx.matmulPerf.mmShapeInfo_.mValue, 1,
                                                               MatmulPerfConst::LOCAL_MM_UTIL_EXPONENT_COEFFICIENT);
        // 除零/NaN 防护: 退化 shape(某维为 0)下 util 可为 0 或 NaN, 退回梯度口径估时
        const double localMMTime = (std::abs(localMMUtil) > 1e-10) ?
                                       ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1) / localMMUtil :
                                       ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        if (isLocalMMFront) {
            ctx.frontMMTime = localMMTime;
            ctx.tilingM.cutRes.shortTileAtBack = true;
        } else {
            ctx.backMMTime = localMMTime;
            ctx.tilingM.cutRes.shortTileAtBack = false;
        }
    } else if (ctx.order == CommComputeOrder::COMPUTE_FIRST) {
        // 先行段为瓶颈 → 短块放另一端
        ctx.tilingM.cutRes.shortTileAtBack = (estimate.totalMatmulTime >= estimate.totalCommTimeNoCut);
    } else {
        ctx.tilingM.cutRes.shortTileAtBack = (estimate.totalCommTimeNoCut > estimate.totalMatmulTime);
    }

    // ---- ② SetShort: 短块下限闸门(由 2b 反解——短块独立占一轮通信,
    //      其通信时长须 ≥ 2b 才值得成轮) + 落位 ----
    ctx.tilingM.SetMinLenByMax(ctx.commPerf.InverseCommTime(2.0 * ctx.commPerf.commEstimatePar_.roundOverheadUs));
    ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
    ctx.tilingM.cutRes.numShortTile = 1U;

    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
        return ctx.tilingM.cutRes;
    }

    // ---- ③ SetLong: 长块配平 ----
    if (isLocalMMFront) {
        // 首长块的通信恰好掩盖本地首块计算
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.frontMMTime);
    } else if (isLocalMMBack) {
        // 末长块的通信恰好掩盖本地尾块计算
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.backMMTime);
    } else {
        BalanceLongTileLen(ctx);
    }
    // ⑤ 轮数搜索的恢复点(反解与轮数无关, 每档仅重跑 ④)
    const CommCutResult afterLong = ctx.tilingM.cutRes;

    // ---- ④ Adjust: 离散化 + 等长归一 ----
    // 有本地块的形态恒信线性段; 离散化保护含本地块的端部结构
    const bool goodLinearity =
        (isLocalMMFront || isLocalMMBack) ? true : IsGoodLinearityShape(ctx.matmulPerf.mmShapeInfo_);
    ctx.tilingM.FitTileLengthDiscrete(goodLinearity, isLocalMMFront || isLocalMMBack);
    MergeEqualTiles(ctx.tilingM);

    // ---- ⑤ 轮数搜索: 配平产出的轮数 N₀ 未必全局最优, 线性枚举 [2, N₀) 择优;
    //      全部候选不敌全串行 → NoCut ----
    const uint64_t mValue = ctx.matmulPerf.mmShapeInfo_.mValue;
    const double serialTotalTime = ctx.commPerf.TotalCommTime(mValue, 1U) + estimate.totalMatmulTime;
    CommCutResult bestCut = ctx.tilingM.cutRes;
    double bestTime = EstimateCutTotalTime(ctx, bestCut, estimate.totalMatmulTime, serialTotalTime);
    // 循环界须固定: bestCut 在循环内会被更优候选替换(轮数可降), 不能充当循环界
    const uint64_t initialTileCnt = bestCut.totalTileCnt;
    for (uint64_t budget = 2U; budget < initialTileCnt; ++budget) {
        ctx.tilingM.cutRes = afterLong;
        ctx.tilingM.SetMaxTileCnt(budget);
        ctx.tilingM.FitTileLengthDiscrete(goodLinearity, isLocalMMFront || isLocalMMBack);
        MergeEqualTiles(ctx.tilingM);
        const double cutTime = EstimateCutTotalTime(ctx, ctx.tilingM.cutRes, estimate.totalMatmulTime, serialTotalTime);
        if (cutTime < bestTime) {
            bestTime = cutTime;
            bestCut = ctx.tilingM.cutRes;
        }
    }
    ctx.tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS);
    if (bestTime >= serialTotalTime) {
        ctx.tilingM.NoCutTiling();
    } else {
        ctx.tilingM.cutRes = bestCut;
    }

    // ---- ⑥ CCU 整形: HCCL 单次调用 256MB 上限 ----
    if (ctx.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(ctx.tilingM, ctx.commPerf, ctx.commVolumeMul, ctx.matmulPerf.mmShapeInfo_.mValue);
    }

    return ctx.tilingM.cutRes;
}

// 核利用率优先切分结果(head/tail 形态) → 本引擎切分结果(1 短 + N 长)转换。
// 标定 miss 走核利用率优先切分兜底时, 用它统一输出形态。
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

// 旧入参转换(存量算子过渡用): 从旧全局 mc2tiling::TilingArgs 挑字段构造引擎入参;
// commEngine 由调用方解析后补入。新算子直接构造 FitBalanceTilingArgs。
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

// 对外唯一入口: 上下文构造 + 求解的串联。
inline CommCutResult FitBalanceCut(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    FitBasedBalanceCtx ctx = InitCtx(args, calib);
    return Schedule(ctx);
}

#endif // __APACE_FIT_BALANCE_TILING_H__
