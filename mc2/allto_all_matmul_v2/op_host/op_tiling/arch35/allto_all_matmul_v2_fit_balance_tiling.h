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
 * \file allto_all_matmul_v2_fit_balance_tiling.h
 * \brief AlltoAllMatmulV2 fit-balance 通信切分：算子特有逻辑与拟合数据的唯一维护点
 *
 * 架构约定:
 *   - 与算子无关的公共决策算法（时延模型/wave 推导/NoCut 与平衡门控/k 搜索/
 *     尾轮合并）在 apace/tiling/fit_balance_tiling.h，对所有算子复用;
 *   - 本文件只保留本算子特有的内容:
 *       公式: 每轮每行通信字节数(MX data+scale, fp8/fp4)、AllToAll fanout、
 *             总 K 折算、双通道 CommTilingData 组装;
 *       数据: 通信/计算拟合系数与阈值常量（标定完成后回填）。
 *   - op tiling 侧把常量与运行时 shape 打包成 AllToAllFitBalanceInput，
 *     调用本文件的 Schedule() 拿到切分结果，再经 FillCommTilingData() 写回 tilingData。
 *
 * 数据来源(标定完成后回填, 占位值 <= 0 时自动回退 CoreOccupancyFirstTiling 路径):
 *   COMM_FIT_ROWS / ROUND_COST_ROWS <-- 标定计划表A / 表C (按 rankDim)
 *   EFF_MAC_PER_CYCLE_*             <-- 标定计划表B (按 dtype × precisionMode 分档:
 *                                       mode 0/1 同档(REMOTE/DEFERRED_SYNC, 总 MACs 相同),
 *                                       mode 2 单独一档(GM 累加: local 前置 + atomiadd + 双写 GM 开销))
 *   FB_* 阈值                        <-- 标定计划表E
 * 标定计划文档: tiling-plan/ALLTO_ALL_MATMUL_V2_FIT_BALANCE_CALIBRATION_PLAN.md
 */

#ifndef ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H
#define ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H

#include <cstdlib>
#include "apace/tiling/fit_balance_tiling.h"
#include "apace/tiling/tiling_data_struct_define.h"
#include "apace/utils/apace_constant.h"

namespace MC2Tiling {
namespace fit_balance {

// ============================================================================
// 拟合数据与阈值常量（标定占位，唯一维护点）
// ============================================================================

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
// 按 dtype × precisionMode 分档: mode 0/1 同档(REMOTE / DEFERRED_SYNC, 总 MACs 与流水结构
// 相同, mode 1 的 local 段提前只让实际重叠更好, 模型保守可用); mode 2 单独一档
// (GM 累加: RunLocalMatmul 前置 + remote 段原子累加 + bias 去除, 存在 MACs 之外的
// atomiadd/双写 GM 开销, 实测耗时系统性高于纯 MMAD 模型)。
// precisionMode 含义: 0 = REMOTE 融合(默认), 1 = DEFERRED_SYNC(L0c 累加), 2 = GM 累加。
struct EffMacRow {
    bool isFp4;
    uint32_t precisionMode; // 0/1 一档(填 0), 2 一档(填 2)
    double effMacPerCycle;
};
static const EffMacRow EFF_MAC_ROWS[] = {
    {false, 0U, 0.0}, // TODO(标定表B-01): fp8 × mode0/1 实测值
    {false, 2U, 0.0}, // TODO(标定表B-2):  fp8 × mode2 实测值
    {true, 0U, 0.0},  // TODO(标定表B-01): fp4 × mode0/1 实测值
    {true, 2U, 0.0},  // TODO(标定表B-2):  fp4 × mode2 实测值
};

inline double FindEffMac(bool isFp4, uint32_t precisionMode)
{
    // mode 0 与 1 共用一档(表 B-01); 归一化后查表, 未命中返回 0 表示缺数据。
    const uint32_t modeClass = (precisionMode == 2U) ? 2U : 0U;
    for (const EffMacRow &row : EFF_MAC_ROWS) {
        if (row.isFp4 == isFp4 && row.precisionMode == modeClass) {
            return row.effMacPerCycle;
        }
    }
    return 0.0;
}

// ====== 模型与阈值常量 ======
static const double A5_CYCLE_PER_US = 1689.6;        // A5 频率: 1.65 cycle/ns
static const double FB_MIN_CUT_TIME_US = 100.0;      // 总时延下限, 低于则不切
static const double FB_BALANCE_RATIO = 1.2;          // 通算比低于该值不干预(k=1)
static const double FB_ROUND_COST_DEFAULT_US = 10.0; // roundCost 缺省值
static const double FB_TINY_TAIL_RATIO = 0.5;        // 尾轮低于 head×该比例时并入最后一个头轮
static const bool FB_ENABLE_TAIL_MERGE = true;       // 尾轮合并(需 kernel 侧显式轮数, urma impl 已适配)
static const uint64_t FB_MAX_TILE_CNT = 16UL;        // 通信轮次硬上限
static const uint64_t FB_MIN_ROUNDS = 3UL;           // 轮次下限

// 环境开关: MC2_ALLTO_ALL_MATMUL_V2_FIT_BALANCE=1 启用(默认关闭, 行为与现状一致)
inline bool IsEnabled()
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
inline bool DataReady(uint32_t rankDim, bool isFp4, uint32_t precisionMode)
{
    const CommFitRow *row = FindCommFit(rankDim);
    if (row == nullptr || row->slopeUsPerMb <= 0.0) {
        return false;
    }
    return FindEffMac(isFp4, precisionMode) > 0.0;
}

// ============================================================================
// 算子特有公式与调度
// ============================================================================

/*!
 * \brief Per-row bytes shipped by one AllToAll round (data + scale channels merged).
 *
 * MX quantised layouts (ka is per-rank K, ka % 64 == 0 is enforced by the op):
 *   data : fp8 -> ka * 1B ; fp4 -> ka / 2B (two elements packed per byte)
 *   scale: ceil(ka / 64) groups * 2B
 */
inline double CalcPerRowBytes(uint64_t ka, bool isFp4)
{
    const double dataBytes = isFp4 ? (static_cast<double>(ka) / 2.0) : static_cast<double>(ka);
    const double scaleBytes = static_cast<double>((ka + 63UL) / 64UL) * 2.0;
    return dataBytes + scaleBytes;
}

/*!
 * \brief Op-facing input: SWAT matmul tiling result + calibrated models.
 */
struct AllToAllFitBalanceInput {
    // Per-rank problem shape (m is per-rank M, ka is per-rank K).
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t ka = 0;
    uint32_t rankDim = 1;
    bool isFp4 = false;
    uint32_t precisionMode = 0; // 0 REMOTE / 1 DEFERRED_SYNC / 2 GM 累加, 决定 effMac 分档

    // Matmul tiling result from QuantMatmulTilingSwat.
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t usedCoreNum = 1;

    // Calibrated models chosen by the op host (see the fitting-data section above).
    apace::CommLinearModel comm;
    apace::MatmulTheoryModel mm;
    apace::FitBalanceConfig cfg;
};

/*!
 * \brief Scheduling result: the shared cut plus per-channel non-split sizes.
 */
struct AllToAllFitBalanceResult {
    apace::FitBalanceCut cut;
    uint64_t dataNonSplitAxis = 0;  // bytes per row on the data channel (fp4 packs 2 elements per byte)
    uint64_t scaleNonSplitAxis = 0; // bytes per row on the scale channel
};

/*!
 * \brief 打包常量与运行时 shape, 生成调度入参
 * \return false 表示拟合数据缺失(含该 precisionMode 档位未标定), 调用方应回退 CoreOccupancyFirstTiling 路径
 */
inline bool BuildScheduleInput(uint64_t m, uint64_t n, uint64_t ka, uint32_t rankDim, bool isFp4,
                               uint32_t precisionMode, uint64_t baseM, uint64_t baseN, uint64_t usedCoreNum,
                               AllToAllFitBalanceInput &out)
{
    if (!DataReady(rankDim, isFp4, precisionMode)) {
        return false;
    }
    const CommFitRow *row = FindCommFit(rankDim);
    const double effMac = FindEffMac(isFp4, precisionMode);
    out = AllToAllFitBalanceInput();
    out.m = m;
    out.n = n;
    out.ka = ka;
    out.rankDim = rankDim;
    out.isFp4 = isFp4;
    out.precisionMode = precisionMode;
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

/*!
 * \brief 运行共享 fit-balance 核心决策（AllToAll 量化 Matmul 融合语义）。
 */
inline AllToAllFitBalanceResult Schedule(const AllToAllFitBalanceInput &in)
{
    AllToAllFitBalanceResult res;
    apace::FitBalanceScheduleInput core;
    core.m = in.m;
    core.n = in.n;
    core.kTotal = static_cast<uint64_t>(in.rankDim) * in.ka;
    core.baseM = in.baseM;
    core.baseN = in.baseN;
    core.usedCoreNum = in.usedCoreNum;
    core.commFanout = (in.rankDim > 1U) ? (static_cast<uint64_t>(in.rankDim) - 1UL) : 1UL;
    core.perRowBytes = CalcPerRowBytes(in.ka, in.isFp4);
    core.comm = in.comm;
    core.mm = in.mm;
    core.mm.coreNum = in.usedCoreNum; // compute runs on the cores the matmul tiling picked
    core.cfg = in.cfg;
    res.cut = apace::FitBalanceSchedule(core);
    res.dataNonSplitAxis = in.isFp4 ? (in.ka / 2UL) : in.ka;
    res.scaleNonSplitAxis = in.ka / MX_GROUP_SIZE;
    return res;
}

/*!
 * \brief Write the scheduling result into the kernel-side dual-channel CommTilingData.
 */
inline void FillCommTilingData(const AllToAllFitBalanceResult &res, CommTilingData &dataTiling,
                               CommTilingData &scaleTiling)
{
    dataTiling.splitAxisTileSize = res.cut.headSize;
    dataTiling.splitAxisTileCnt = res.cut.headCnt;
    dataTiling.splitAxisTailSize = res.cut.tailSize;
    dataTiling.splitAxisTailCnt = res.cut.tailCnt;
    dataTiling.nonSplitAxisSize = res.dataNonSplitAxis;
    scaleTiling = dataTiling;
    scaleTiling.nonSplitAxisSize = res.scaleNonSplitAxis;
}

} // namespace fit_balance
} // namespace MC2Tiling

#endif // ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H
