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
 * \brief AlltoAllMatmulV2 fit-balance 算子侧唯一维护点（与其他算子同构格式, 迁移方案 §4.9）
 *
 * 内容分区:
 *   一、客户标定常量区 —— 重新标定后只改本区的 constexpr 值;
 *   二、标定表组装 —— 仅包装上方常量, 无魔数;
 *   三、开关与数据就绪 —— 环境变量开关(保留已发布语义) + 标定数据就绪检查;
 *   四、算子流量公式与组装 —— BuildScheduleInput/Schedule/FillCommTilingData。
 *
 * 公共引擎在 apace/tiling/fit_balance_tiling.h 三、3.2（wave-packing 求解器, URMA 分支）。
 * 本算子即 aiv_urma 实现（kernel 仅 URMA）, tiling 无条件走 fit-balance 路径。
 */

#ifndef ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H
#define ALL_TO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H

#include <cstdlib>
#include "apace/tiling/fit_balance_tiling.h"
#include "apace/tiling/tiling_data_struct_define.h"
#include "apace/utils/apace_constant.h"

namespace MC2Tiling {
namespace fit_balance {

namespace fb = apace::fit_balance; // fit-balance 家族公共层

// ============================================================================
// 一、客户标定常量区 —— 重新标定后只改本区，各常量的物理含义/单位/来源见行尾注释
//     （当前值 2026-09-09~10 标定, 标定环境与口径见
//      tiling-plan/ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_PLAN.md 表0/表A/表B/表C）
// ============================================================================

// ---- 1. 通信拟合 · URMA 引擎（4P 已标定; URMA 无 AICPU/CCU 之分, 不按模式拆分）----
// 标定产出: 单轮通信时延 y(us) = 斜率·x + 截距, x = 单轮单 rank 总发送量(MB)
//          = (rankDim-1) × mSpan × 每行字节数(MX data+scale 合并)
constexpr uint32_t A2A_V2_COMM_4P_RANK_DIM = 4;
constexpr double A2A_V2_COMM_4P_SLOPE_US_PER_MB = 7.724; // 拟合斜率 (us/MB), R²=1.0000
constexpr double A2A_V2_COMM_4P_INTERCEPT_US = 8.91;     // 拟合截距 (us), 含每轮同步固定开销
constexpr uint32_t A2A_V2_COMM_8P_RANK_DIM = 8;          // 8P 占位(未标定): 行已接线, slope<=0 时
constexpr double A2A_V2_COMM_8P_SLOPE_US_PER_MB = 0;     // DataReady=false 自动回退 CoreOccupancyFirst
constexpr double A2A_V2_COMM_8P_INTERCEPT_US = 0;        // TODO(8P标定): 填实测斜率/截距即生效

// ---- 2. 每轮通信固定开销（轮次上限推导用: maxRounds = totalComm/roundCost）----
constexpr double A2A_V2_ROUND_COST_4P_US = 4.45;      // 标定表C: R=2/4/8/16 严格线性回归
constexpr double A2A_V2_ROUND_COST_8P_US = 10.0;      // 未标定缺省值
constexpr double A2A_V2_ROUND_COST_DEFAULT_US = 10.0; // 查表未命中回退值

// ---- 3. matmul 有效算力（MACs/cycle, 含 cube 利用率; 按 dtype × precisionMode 分档）----
// mode 0/1 同档(REMOTE/DEFERRED_SYNC 总 MACs 与流水同构); mode 2 单独一档(GM 累加额外开销)
constexpr double A2A_V2_EFFMAC_FP8_MODE01 = 7490;  // 标定表B-01
constexpr double A2A_V2_EFFMAC_FP8_MODE2 = 7213;   // 标定表B-2
constexpr double A2A_V2_EFFMAC_FP4_MODE01 = 14064; // 约为 FP8 的 1.88×
constexpr double A2A_V2_EFFMAC_FP4_MODE2 = 14055;

// ---- 4. 决策阈值 ----
constexpr double A2A_V2_CYCLE_PER_US = 1689.6;    // A5 频率: 1.65 cycle/ns
constexpr double A2A_V2_BALANCE_RATIO = 1.2;      // 通算比低于此值不干预(k=1, 轮次天然贴合计算波)
constexpr double A2A_V2_TINY_TAIL_RATIO = 0.5;    // 尾轮低于 head×此比例时并入最后一个头轮
constexpr double A2A_V2_TAIL_MERGE_MARGIN = 0.02; // 尾轮合并定价容差: mergedEst ≤ est×(1+此值)才放行
constexpr bool A2A_V2_ENABLE_TAIL_MERGE = true;   // 尾轮合并(需 kernel 侧显式轮数, urma impl 已适配)
constexpr uint64_t A2A_V2_MAX_TILE_CNT = 16;      // 通信轮次硬上限(FFTS 队列限制)
constexpr uint64_t A2A_V2_MIN_ROUNDS = 3;         // 轮次下限

// ============================================================================
// 二、标定表组装（勿动: 仅包装上方常量; 新增标定形态时按同 schema 加行）
// ============================================================================

static const fb::UrmaCommFitRow COMM_FIT_ROWS[] = {
    {A2A_V2_COMM_4P_RANK_DIM, A2A_V2_COMM_4P_SLOPE_US_PER_MB, A2A_V2_COMM_4P_INTERCEPT_US},
    {A2A_V2_COMM_8P_RANK_DIM, A2A_V2_COMM_8P_SLOPE_US_PER_MB, A2A_V2_COMM_8P_INTERCEPT_US},
};

static const fb::RoundCostRow ROUND_COST_ROWS[] = {
    {A2A_V2_COMM_4P_RANK_DIM, A2A_V2_ROUND_COST_4P_US},
    {A2A_V2_COMM_8P_RANK_DIM, A2A_V2_ROUND_COST_8P_US},
};

static const fb::EffMacRow EFF_MAC_ROWS[] = {
    {false, 0U, A2A_V2_EFFMAC_FP8_MODE01},
    {false, 2U, A2A_V2_EFFMAC_FP8_MODE2},
    {true, 0U, A2A_V2_EFFMAC_FP4_MODE01},
    {true, 2U, A2A_V2_EFFMAC_FP4_MODE2},
};

// ============================================================================
// 三、开关与数据就绪
// ============================================================================

// 环境开关: MC2_ALLTO_ALL_MATMUL_V2_FIT_BALANCE=0 强制关闭(一键回退); 缺省按标定数据
// 就绪情况自动启用。标定辅助开关 MC2_A2A_V2_CALIB_R1=1 隐含关闭(优先级最高, 保证 R=1 标定口径)。
inline bool IsEnabled()
{
    const char *r1Env = getenv("MC2_A2A_V2_CALIB_R1");
    if ((r1Env != nullptr) && (r1Env[0] == '1')) {
        return false;
    }
    const char *env = getenv("MC2_ALLTO_ALL_MATMUL_V2_FIT_BALANCE");
    return !((env != nullptr) && (env[0] == '0'));
}

// 拟合数据是否就绪(未标定档位自动回退现状路径)
inline bool DataReady(uint32_t rankDim, bool isFp4, uint32_t precisionMode)
{
    const fb::UrmaCommFitRow *row =
        fb::FindUrmaCommFit(COMM_FIT_ROWS, sizeof(COMM_FIT_ROWS) / sizeof(COMM_FIT_ROWS[0]), rankDim);
    if (row == nullptr || row->slopeUsPerMb <= 0.0) {
        return false;
    }
    return fb::FindEffMac(EFF_MAC_ROWS, sizeof(EFF_MAC_ROWS) / sizeof(EFF_MAC_ROWS[0]), isFp4, precisionMode) > 0.0;
}

// ============================================================================
// 四、算子流量公式与组装
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
    fb::CommLinearModel comm;
    fb::MatmulTheoryModel mm;
    fb::FitBalanceConfig cfg;
};

/*!
 * \brief Scheduling result: the shared cut plus per-channel non-split sizes.
 */
struct AllToAllFitBalanceResult {
    fb::FitBalanceCut cut;
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
    const fb::UrmaCommFitRow *row =
        fb::FindUrmaCommFit(COMM_FIT_ROWS, sizeof(COMM_FIT_ROWS) / sizeof(COMM_FIT_ROWS[0]), rankDim);
    if (row == nullptr || row->slopeUsPerMb <= 0.0) {
        return false;
    }
    const double effMac =
        fb::FindEffMac(EFF_MAC_ROWS, sizeof(EFF_MAC_ROWS) / sizeof(EFF_MAC_ROWS[0]), isFp4, precisionMode);
    if (effMac <= 0.0) {
        return false;
    }
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
    out.mm.cyclePerUs = A2A_V2_CYCLE_PER_US;
    out.mm.coreNum = usedCoreNum;
    out.cfg.enabled = true;
    out.cfg.balanceRatio = A2A_V2_BALANCE_RATIO;
    out.cfg.roundCostUs = fb::FindRoundCost(ROUND_COST_ROWS, sizeof(ROUND_COST_ROWS) / sizeof(ROUND_COST_ROWS[0]),
                                            rankDim, A2A_V2_ROUND_COST_DEFAULT_US);
    out.cfg.tinyTailRatio = A2A_V2_TINY_TAIL_RATIO;
    out.cfg.enableTailMerge = A2A_V2_ENABLE_TAIL_MERGE;
    out.cfg.tailMergeMarginRatio = A2A_V2_TAIL_MERGE_MARGIN;
    out.cfg.maxTileCnt = A2A_V2_MAX_TILE_CNT;
    out.cfg.minRounds = A2A_V2_MIN_ROUNDS;
    return true;
}

/*!
 * \brief 运行共享 fit-balance 核心决策（AllToAll 量化 Matmul 融合语义）。
 */
inline AllToAllFitBalanceResult Schedule(const AllToAllFitBalanceInput &in)
{
    AllToAllFitBalanceResult res;
    fb::FitBalanceScheduleInput core;
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
    res.cut = fb::FitBalanceSchedule(core);
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
