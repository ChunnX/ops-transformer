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
 * \file all_gather_matmul_v2_fit_balance_tiling.h
 * \brief AllGatherMatmulV2 fit-balance 算子侧唯一维护点
 *
 * 内容分区（迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md §4.9 统一接口模式）:
 *   一、客户标定数据 —— 唯一实例 CALIBRATION(单一 FitBalanceCalibration 大结构体,
 *      2026-09-15 评审修订), 重新标定后只改其中数值;
 *   二、算子特有钩子（原派生类 override 的逐行拷贝; 含少量算子逻辑参数）;
 *   三、组装与新旧切换: GetFitBalanceCut(args) 一行调用; kUseNewEngine 置 false 即回老类
 *      （老类 all_gather_fit_balance_tiling.h 原样保留）。
 *
 * 公共逻辑（模型/切分引擎/调度流）在 apace/tiling/fit_balance_tiling.h 三、3.1（自由长度求解器）,
 * 本文件不重复任何公共实现。
 */

#ifndef __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__

#pragma once

#include "all_gather_fit_balance_tiling.h" // 老类(回退分支保留, 迁移收尾后随 P4 删除)
#include "apace/tiling/fit_balance_tiling.h"
#include "mc2_log.h"
#include "op_host/op_tiling/formulaic_tiling_datatype.h" // 旧 CutResult/KernelType/TopoType(对外接口沿用旧类型)
#include "op_host/op_tiling/matmul_formulaic_tiling.h"   // mc2tiling::TilingArgs

namespace agmm_v2_fit_balance {

namespace fb = mc2tiling::fit_balance; // fit-balance 家族公共层

// ============================================================================
// 一、客户标定数据（唯一实例; 结构与字段含义见公共文件 FitBalanceCalibration 定义。
//     当前值提取自旧公共层全量表的本算子可达行, 2026-09-14）
// ============================================================================

static const fb::FitBalanceCalibration CALIBRATION = {
    /* aicpu（commMode != "ccu"）*/
    {
        /* rows: {卡数, 斜率(us/MB), 截距(us)} */
        {
            {4, 5.841842, 8.590666}, // 950 AllGather 标卡 4P: R²=1.0000
            // 8P 未标定(留空, miss 走平台默认行 + rankDim 折算缩放); 标定后追加 {8, 斜率, 截距}
        },
        // platformDefault: 平台默认回退行, 用公共层默认值(勿动)
    },
    /* ccu（commMode == "ccu"）*/
    {
        /* rows */
        {
            {4, 5.841842, 8.590666}, // TODO(CCU 重标): 当前沿用历史共用值(旧表无引擎维度)
        },
    },
    /* matmul */
    {
        /* cubeCalc: {计算类型, isFp4, A/B/C 字节数, 有效算力(MACs/cycle)} */
        {
            {fb::MatmulCalcType::QUANT, false, 1, 1, 2, 8192}, // fp8 量化(1B/1B/2B)
            {fb::MatmulCalcType::QUANT, true, 1, 1, 2, 16384}, // fp4 量化(打包 1B/1B/2B, 算力约为 fp8 的 2×)
        },
        /* l2Param: {计算类型, isFp4, A/B/C 字节数, L2 档位参数} */
        {
            {fb::MatmulCalcType::FP16, false, 2, 2, 2, {80, 128, 0.95, 0.85, 0.75}},  // 半精度
            {fb::MatmulCalcType::QUANT, false, 1, 1, 2, {80, 128, 0.95, 0.85, 0.75}}, // fp8 量化
            {fb::MatmulCalcType::QUANT, true, 1, 1, 2, {80, 128, 0.95, 0.85, 0.75}},  // fp4 量化(共用 fp8 档位)
        },
        /* baseBlock: matmul 最优基本块(单值; 未填用默认 {128,256,64}) */
        {256, 256, 128},
    },
};

// ============================================================================
// 二、算子特有钩子（= 老 all_gather_fit_balance_tiling.cpp 逐行拷贝, 状态存本结构体;
//     以下 4 个常量为钩子逻辑参数(非标定数据): 减半鼓励门与 fp4 建模折算）
// ============================================================================

constexpr double MM_EXPANSION_TIME = 40;    // 短块减半仅当半块计算耗时 > 此值(us)
constexpr double COMM_EXPANSION_TIME = 40;  // 且半块通信耗时 > 此值(us)
constexpr uint64_t FP4_EXPANSION_RATIO = 2; // fp4 通信按"1字节打包2元素"建模的分母

struct AgMmV2Hooks {
    double frontMMTime = 0;

    void EstimateMMCommTime(mc2tiling::fit_balance::FitBasedBalanceCtx &ctx)
    {
        // Estimate 骨干(4/5 算子一致的公共片段): 模型刷新 + 总量 + 通算比
        // AG 语义: 通信聚合 rankDim 份后 mm 按 rankDim 倍计(与老代码 MatmulTime(m, rankDim_) 一致)
        const auto est = mc2tiling::fit_balance::EstimateTotalTimes(ctx, /*mmRankMul=*/ctx.commPerf.rankDim());
        const double totalMatmulTime = est.totalMatmulTime;
        const double totalTpTime = est.totalTpTime;
        double frontUtil = ctx.matmulPerf.FindCubeUtilByL2Usage(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        if (std::abs(frontUtil) > 1e-10) {
            frontMMTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1) / frontUtil;
        } else {
            frontMMTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        }
        ctx.tilingM.cutRes.shortTileAtBack = true; // AG 特有: 恒后置(其余算子按 bound 判定)
        // ratioCalcComm 已由 EstimateTotalTimes 设置

        uint64_t sizeOfComm = ctx.matmulPerf.mmShapeInfo_.mValue * ctx.matmulPerf.mmShapeInfo_.kValue *
                              (ctx.commPerf.rankDim() - 1) / mc2tiling::fit_balance::ONE_MBYTE *
                              ctx.commPerf.GetCommDTypeSize();
        OPS_LOG_D("AllGatherMatmul",
                  "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
                  "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
                  "rankDim_ %lu, rankTile %lu",
                  ctx.matmulPerf.mmShapeInfo_.mValue, ctx.matmulPerf.mmShapeInfo_.nValue,
                  ctx.matmulPerf.mmShapeInfo_.kValue, ctx.matmulPerf.cubeUtil_, sizeOfComm, totalMatmulTime,
                  totalTpTime, ctx.tilingM.GetMinLen(), ctx.tilingM.GetAlignLength(), ctx.commPerf.commTimeFactor_,
                  ctx.commPerf.rankDim(), ctx.rankTileNum);
    }

    void SetLongTileLen(mc2tiling::fit_balance::FitBasedBalanceCtx &ctx)
    {
        // Long tile should overlap the cost time of local tile
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(frontMMTime);
    }

    void SetShortTileLen(mc2tiling::fit_balance::FitBasedBalanceCtx &ctx)
    {
        // Encourage split more if the comm and calc is balanced and the cost of cutLen is sufficiently high
        bool isCalcCommBalance = ctx.ratioCalcComm < mc2tiling::fit_balance::CALC_COMM_RATIO;
        uint64_t cutLen = ctx.tilingM.GetAlignLength() / mc2tiling::fit_balance::TWO;
        double mmCost = ctx.matmulPerf.MatmulTime(cutLen, ctx.commPerf.rankDim());
        double commCost = ctx.commPerf.CommTime(cutLen);
        if (isCalcCommBalance && (mmCost > MM_EXPANSION_TIME) && (commCost > COMM_EXPANSION_TIME)) {
            ctx.tilingM.SetAlignLength(cutLen);
            ctx.tilingM.SetMinLenByMin(cutLen);
        }

        ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
        ctx.tilingM.cutRes.numShortTile = 1U;
    }

    void AdjustLongShortTileLen(mc2tiling::fit_balance::FitBasedBalanceCtx &ctx)
    {
        ctx.tilingM.FitTileLengthDiscrete(false, true, true);
        mc2tiling::fit_balance::MergeEqualTiles(ctx.tilingM); // 公共钩子片段(五算子一致的等长归一尾)
    }
};

// ============================================================================
// 三、边界转换 + 组装与新旧切换
// ============================================================================

// 统一引擎路径: 上下文构造 + 算子后置初始化(= 老类构造函数体逐行对应) + 统一调度
// baseMOut(可空): 回传 matmul 基本块(调用点 AdjustCutResultForAICPU 需要)
inline CutResult ScheduleFitBalance(const mc2tiling::TilingArgs &args, fb::CommEngine commEngine,
                                    uint32_t *baseMOut = nullptr)
{
    // 引擎分路在 InitCtx 内完成(按 CommEngine 选 CALIBRATION 的 aicpu/ccu 分区);
    // 本算子 kernel 仅 HCCL 两引擎, AIV_URMA 不可达(枚举值保留供三引擎分派框架)
    auto ctx = fb::InitCtx(args, fb::KernelType::ALL_GATHER, commEngine, CALIBRATION, fb::TopoType::STANDARD_CARD);
    ctx.commPerf.SetCommShapeLen(ctx.matmulPerf.mmShapeInfo_.kValue);
    // fp4(DT_FLOAT4_E2M1) 通信侧建模: 每元素 0.5 字节, 按"1 字节打包 2 个 fp4"建模
    // (SetCommDTypeSize(1) + ExpansionFraction(2), 复用 HCCL 子字节机制);
    // matmul 侧 fp4 路由已由引擎正确处理(SetCalcType 经 geAType 识别, 查 CALIBRATION 的
    // fp4 行取算力), 无需手动覆盖。inMatrixADtypeSize 置 1(L2 占用按 2x 高估, 避免 0 退化)。
    if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
        ctx.commPerf.SetCommDTypeSize(1);
        ctx.commPerf.SetCommDtypeSizeExpansionFraction(FP4_EXPANSION_RATIO);
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
    } else {
        ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize);
    }
    ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.GetBaseM());
    ctx.tilingM.SetAlignLength(ctx.matmulPerf.GetBaseM());

    AgMmV2Hooks hooks;
    CutResult cut = fb::Schedule(ctx, hooks);
    if (baseMOut != nullptr) {
        *baseMOut = ctx.matmulPerf.GetBaseM();
    }
    return cut;
}

// 新旧切换开关: 置 false 并重新编译即整体回退老类路径（出问题时的回退手段）
constexpr bool kUseNewEngine = true;

// 对外唯一入口：两个 GetTilingResult 调用点共用（commEngine 决定通信拟合选表; 三引擎
// 统一分派模式——调用点用 mc2tiling::fit_balance::ParseCommEngine(commMode_) 解析后传入）
inline CutResult GetFitBalanceCut(const mc2tiling::TilingArgs &args, fb::CommEngine commEngine,
                                  uint32_t *baseMOut = nullptr)
{
    if (kUseNewEngine) {
        return ScheduleFitBalance(args, commEngine, baseMOut);
    }
    AllGatherMMFitBalanceTiling tileFormulate(args, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    if (baseMOut != nullptr) {
        *baseMOut = tileFormulate.matmulPerf_.GetBaseM();
    }
    return tileFormulate.GetTiling();
}

} // namespace agmm_v2_fit_balance

#endif // __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
