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
 * \file allto_all_matmul_v2_fit_balance_params.h
 * \brief AlltoAllMatmulV2 fit-balance tiling: 拟合数据与阈值常量的唯一维护点
 *
 * 架构约定:
 *   - 公共算法在 apace/tiling/fit_balance_tiling.h (共享调度核心) 与
 *     apace/tiling/all_to_all_quant_matmul/all_to_all_fit_balance_tiling.h (算子适配层);
 *   - 本头文件只维护本算子的拟合数据与公共接口所需常量入参;
 *   - op tiling 侧通过 BuildAllToAllFitBalanceInput() 把常量打包成结构体,
 *     调用 apace 统一接口拿到切分结果。
 *
 * 数据来源(标定完成后回填, 占位值 <= 0 时自动回退现有 CommTilingBase 路径):
 *   COMM_FIT_ROWS / ROUND_COST_ROWS <-- 标定计划表A / 表C (按 rankDim)
 *   EFF_MAC_PER_CYCLE_FP8 / _FP4    <-- 标定计划表B (按 dtype)
 *   FB_* 阈值                        <-- 标定计划表E
 * 标定计划文档: tiling-plan/ALLTO_ALL_MATMUL_V2_FIT_BALANCE_CALIBRATION_PLAN.md
 */

#ifndef ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_PARAMS_H
#define ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_PARAMS_H

#include <cstdlib>
#include "apace/tiling/all_to_all_quant_matmul/all_to_all_fit_balance_tiling.h"

namespace MC2Tiling {
namespace fit_balance_params {

// ====== 通信拟合数据: y(us) = slope * x(MB) + intercept, x 为单轮单 rank 总发送量 ======
struct CommFitRow {
    uint32_t rankDim;
    double slopeUsPerMb;
    double interceptUs;
};
static const CommFitRow COMM_FIT_ROWS[] = {
    {4U, 0.0, 0.0}, // TODO(标定表A-4P): 回填 slope/intercept
    {8U, 0.0, 0.0}, // TODO(标定表A-8P): 回填 slope/intercept
};

// ====== 每轮通信固定开销(用于轮次上限推导) ======
struct RoundCostRow {
    uint32_t rankDim;
    double roundCostUs;
};
static const RoundCostRow ROUND_COST_ROWS[] = {
    {4U, 10.0}, // TODO(标定表C-4P): 回填实测值
    {8U, 10.0}, // TODO(标定表C-8P): 回填实测值
};

// ====== 计算有效算力(MACs/cycle, 含 cube 利用率) ======
static const double EFF_MAC_PER_CYCLE_FP8 = 0.0; // TODO(标定表B): fp8 实测值
static const double EFF_MAC_PER_CYCLE_FP4 = 0.0; // TODO(标定表B): fp4 实测值

// ====== 模型与阈值常量 ======
static const double A5_CYCLE_PER_US = 1689.6;   // A5 频率: 1.65 cycle/ns
static const double FB_MIN_CUT_TIME_US = 100.0; // 总时延下限, 低于则不切
static const double FB_BALANCE_RATIO = 1.2;     // 通算比低于该值不干预(k=1)
static const double FB_ROUND_COST_DEFAULT_US = 10.0; // roundCost 缺省值
static const double FB_TINY_TAIL_RATIO = 0.5;   // 尾轮低于 head×该比例时并入最后一个头轮
static const bool FB_ENABLE_TAIL_MERGE = true;  // 尾轮合并(需 kernel 侧显式轮数, urma impl 已适配)
static const uint64_t FB_MAX_TILE_CNT = 16UL;   // 通信轮次硬上限
static const uint64_t FB_MIN_ROUNDS = 3UL;      // 轮次下限

// 环境开关: MC2_ALLTO_ALL_MATMUL_V2_FIT_BALANCE=1 启用(默认关闭, 行为与现状一致)
inline bool IsFitBalanceEnvEnabled()
{
    const char *env = getenv("MC2_ALLTO_ALL_MATMUL_V2_FIT_BALANCE");
    return (env != nullptr) && (env[0] == '1');
}

inline const CommFitRow *FindCommFit(uint32_t rankDim)
{
    for (const CommFitRow &row : COMM_FIT_ROWS) {
        if (row.rankDim == rankDim) {
            return &row;
        }
    }
    return nullptr;
}

inline double FindRoundCost(uint32_t rankDim)
{
    for (const RoundCostRow &row : ROUND_COST_ROWS) {
        if (row.rankDim == rankDim) {
            return row.roundCostUs;
        }
    }
    return FB_ROUND_COST_DEFAULT_US;
}

// 拟合数据是否就绪(未标定时禁用 fit-balance, 回退现状路径)
inline bool FitBalanceDataReady(uint32_t rankDim, bool isFp4)
{
    const CommFitRow *row = FindCommFit(rankDim);
    if (row == nullptr || row->slopeUsPerMb <= 0.0) {
        return false;
    }
    const double effMac = isFp4 ? EFF_MAC_PER_CYCLE_FP4 : EFF_MAC_PER_CYCLE_FP8;
    return effMac > 0.0;
}

/*!
 * \brief 打包常量与运行时 shape, 生成 apace 统一接口的输入
 * \return false 表示拟合数据缺失, 调用方应回退现有 CommTilingBase 路径
 */
inline bool BuildAllToAllFitBalanceInput(uint64_t m, uint64_t n, uint64_t ka, uint32_t rankDim, bool isFp4,
                                         uint64_t baseM, uint64_t baseN, uint64_t usedCoreNum,
                                         apace::all_to_all_quant_matmul::AllToAllFitBalanceInput &out)
{
    if (!FitBalanceDataReady(rankDim, isFp4)) {
        return false;
    }
    const CommFitRow *row = FindCommFit(rankDim);
    const double effMac = isFp4 ? EFF_MAC_PER_CYCLE_FP4 : EFF_MAC_PER_CYCLE_FP8;
    out = apace::all_to_all_quant_matmul::AllToAllFitBalanceInput();
    out.m = m;
    out.n = n;
    out.ka = ka;
    out.rankDim = rankDim;
    out.isFp4 = isFp4;
    out.baseM = baseM;
    out.baseN = baseN;
    out.usedCoreNum = usedCoreNum;
    out.comm.slopeUsPerMb = row->slopeUsPerMb;
    out.comm.interceptUs = row->interceptUs;
    out.mm.effMacPerCycle = effMac;
    out.mm.cyclePerUs = A5_CYCLE_PER_US;
    out.mm.coreNum = usedCoreNum;
    out.cfg.enabled = true;
    out.cfg.minCutTimeUs = FB_MIN_CUT_TIME_US;
    out.cfg.balanceRatio = FB_BALANCE_RATIO;
    out.cfg.roundCostUs = FindRoundCost(rankDim);
    out.cfg.tinyTailRatio = FB_TINY_TAIL_RATIO;
    out.cfg.enableTailMerge = FB_ENABLE_TAIL_MERGE;
    out.cfg.maxTileCnt = FB_MAX_TILE_CNT;
    out.cfg.minRounds = FB_MIN_ROUNDS;
    return true;
}

} // namespace fit_balance_params
} // namespace MC2Tiling

#endif // ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_PARAMS_H
