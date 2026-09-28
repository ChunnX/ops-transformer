/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_v2_fit_balance_tiling.h
 * \brief AllGatherMatmulV2 fit-balance 算子侧零特有代码形态（2026-09-17 §4.24 收官裁定）
 *
 * 命名空间 optiling(与本目录其他文件一致); 引擎符号为 global(apace/tiling 约定)。
 * 本文件只剩三样东西, 无任何算子特有逻辑（钩子机制已废除, 结构差异由 LocalMMPosition 键控）:
 *   一、客户标定数据 CALIBRATION（唯一编辑面, 重新标定只改数值; §4.38 后含 CommProfile
 *       结构参数——本算子: 通信沿 K 轴传 A 侧数据 / mm 工作集倍数 rankDim−1(本地份不经
 *       通信) / mm 总行数倍数 rankDim(聚合后算全量)）;
 *   二、一行组装: InitCtx(LocalMMPosition::FRONT, COMM_FIRST, commEngine, CALIBRATION) + Schedule;
 *   三、kUseNewEngine 新旧切换（置 false 重编译即回老类, 老类原样保留）。
 *
 * 公共逻辑与 FRONT 键控分支（本地首块重叠建模/长块反解/短块收尾 + D4/D5/D6 胶水)
 * 均在 apace/tiling/fit_balance_tiling.h(global)。
 */

#ifndef __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__

#pragma once

#include "all_gather_fit_balance_tiling.h"            // 老类(回退分支保留, 迁移收尾后随 P4 删除)
#include "apace/tiling/core_occupancy_first_tiling.h" // GetCommTilingFromV3/CommTilingData(miss→COF 分派)
#include "apace/tiling/fit_balance_tiling.h"
#include "mc2_log.h"
#include "op_host/op_tiling/formulaic_tiling_datatype.h" // 旧 CutResult/KernelType/TopoType(对外接口沿用旧类型)
#include "op_host/op_tiling/matmul_formulaic_tiling.h"   // mc2tiling::TilingArgs

namespace optiling { // 与本目录其他 op_tiling 文件一致(2026-09-17 裁定)

// ============================================================================
// 一、客户标定数据（唯一实例; 结构与字段含义见公共文件 FitBalanceCalibration 定义。
//     当前值提取自旧公共层全量表的本算子可达行, 2026-09-14）
// ============================================================================

static const FitBalanceCalibration CALIBRATION = {
    /* profile: 通信结构参数(§4.38——原语语义终结在标定层, 引擎不区分通信原语) */
    {
        CommAxis::K,                 // 通信承载 A 侧输入(gather 沿 K 轴聚散, 传 A 矩阵行)
        RankMul::RANK_DIM_MINUS_ONE, // mm 工作集倍数: 远程 rankDim−1 份经通信参与计算(本地份不经通信)
        RankMul::RANK_DIM,           // mm 总行数倍数: 聚合后每卡算全量 m×rankDim 行
        RankMul::RANK_DIM, // 单次 gather 调用物理字节倍数: 单次搬 rankDim 份(⑥ 的 256MB 判定)
    },
    /* aicpu（commMode != "ccu"）*/
    {
        /* rows: {卡数, k 带宽斜率(us/MB), b 每轮启动开销(us), c 一次性固定开销(us)} */
        {
            // 2026-09-23 CommBench 实测标定(runner/calib_kbc.py, AG/aicpu/int8/rank4):
            //   k: turn=1 尺寸扫描 8-96MB/rank 线性拟合 26.5894us/MB(R2=1.00000)。
            //      §4.38 口径消融: 引擎 x = m×K×字节数(单卡份 MB), k 直接用工具原始值——
            //      原 commRankCnt(=rankDim) 乘子已融进 k(旧值 6.647350=26.5894/4 为含乘子口径)
            //   b: 固定 chunk 扫轮数 T(N)=N(k*chunk+b)+c, chunk 1/4MB 一致(b=32.27/33.17,
            //      取中位 33.17); 16MB 大块带宽略亚线性致其拟合值偏离(b=66,c=-146),弃用
            //   c: 同拟合截距, chunk 1/4MB 值 34.42/20.58, 取中位 20.58
            {4, 26.5894, 33.17, 20.58}, // 950 AllGather AICPU 标卡 4P 实测(CommBench, 工具口径 k)
                                        // 8P 未标定(留空, miss 走 COF 分派); 标定后追加 {8, k, b, c}
        },
    },
    /* ccu（commMode == "ccu"）*/
    {
        /* rows */
        {
            // §4.38 口径消融等价换算: 旧值(含 commRankCnt=4 乘子口径)×4 = 引擎新口径(x=单卡份)
            {4, 23.367368, 8.590666, 0.0}, // TODO(CCU 重标): 历史共用值换算, 旧表无引擎维度
        },
    },
    /* mmFit: matmul 性能拟合(算力档位/基本块) */
    {
        /* cubeCalc: {计算类型, isFp4, A/B 字节数, 有效算力(MACs/cycle)}——键不含 C 侧字节数
         * (glue 恒 cType=aType → cBytes≡aBytes 零信息量; C dtype 只影响写出带宽不影响
         * MACs/cycle, §4.37 缺陷根治裁定, 详见引擎 CubeCalc 注释) */
        {
            {MatmulDtypeKind::QUANT, false, 1, 1,
             8192}, // fp8 量化(glue 真实/假设口径键天然合并)
                    // fp4 glue 真实口径: GetDataTypeSize(DT_FLOAT4_E2M1) 无条目返回 0 → 键 {0,0}
                    // [上游演化同步点] 若 GetDataTypeSize 补 fp4 条目(glue 将发 1), 此处需补
            {MatmulDtypeKind::QUANT, true, 0, 0, 16384},
        },
        /* baseBlock: matmul 最优基本块(单值; 未填用默认 {128,256,64}) */
        {256, 256, 128},
    },
};

// ============================================================================
// 二、组装与新旧切换
// ============================================================================

// 边界转换(引擎自包含, 2026-09-17 裁定): 旧全局类型 ↔ 引擎自有类型整字段拷贝
inline FitBalanceTilingArgs ToEngineArgs(const mc2tiling::TilingArgs &a)
{
    return {static_cast<AicpuComType>(a.cmdType),
            a.rankDim,
            a.usedCoreNum,
            a.orgMValue,
            a.orgNValue,
            a.orgKValue,
            a.batchValue,
            a.mValue,
            a.kValue,
            a.nValue,
            a.baseMLimit,
            a.inputDtypeSize,
            a.outputDtypeSize,
            a.aicCoreNum,
            a.commTurn,
            a.rankTileNum,
            a.commAlg,
            a.isATrans,
            a.isBTrans,
            a.enablePad,
            a.enableSplitK,
            a.isBias,
            a.isStorageGather,
            a.isLocal,
            a.geAType,
            a.geBType,
            a.geCType,
            a.geBiasType,
            a.antiquantscaleDType,
            a.aType,
            a.bType,
            a.cType,
            a.biasType};
}

inline CutResult ToLegacyCut(const CommCutResult &c)
{
    return {c.shortTileAtBack, c.totalTileCnt, c.numLongTile, c.numShortTile, c.longTileLen, c.shortTileLen};
}

// 统一引擎路径: 键控枚举 + 标定 → InitCtx(含通算胶水) → Schedule(含 AG 键控分支)。
// baseMOut(可空): 回传 matmul 基本块(回退老类路径的调用点后处理 AdjustCutResultForAICPU 需要;
// 新引擎路径不消费——§4.37 切分收益门控已接替, 保留出参仅为回退路径完整)
inline CutResult ScheduleFitBalance(const mc2tiling::TilingArgs &args, TilingCommEngine commEngine,
                                    uint32_t *baseMOut = nullptr)
{
    // 引擎分路在 InitCtx 内完成(按 CommEngine 选 CALIBRATION 的 aicpu/ccu 分区);
    // 本算子 kernel 仅 HCCL 两引擎, AIV_URMA 不可达(枚举值保留供三引擎分派框架)
    auto ctx = InitCtx(ToEngineArgs(args), LocalMMPosition::FRONT, CommComputeOrder::COMM_FIRST, commEngine,
                       CALIBRATION, CommTopology::STANDARD_CARD);
    CutResult cut = ToLegacyCut(Schedule(ctx));
    // INFO 级分派标识(与 miss 路径的 "COF cut:" 对仗): 默认日志级别即可正向确认命中标定并
    // 走进 fit-balance 新引擎(kUseNewEngine=false 回退老类不打本行); 详细决策与启动税诊断
    // 仍为 debug 级(下方两行)
    OPS_LOG_I("AllGatherMatmul", "fit-balance cut: engine=%d, %lu rounds: %lux%lu+%lux%lu atBack=%d",
              static_cast<int32_t>(commEngine), cut.totalTileCnt, cut.numLongTile, cut.longTileLen, cut.numShortTile,
              cut.shortTileLen, cut.shortTileAtBack);
    OPS_LOG_D("AllGatherMatmul",
              "Input shape {M, N, K} = {%lu, %lu, %lu}, Final cut: shortTileAtBack %d, longTileLen %lu, "
              "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
              ctx.matmulPerf.mmShapeInfo_.mValue, ctx.matmulPerf.mmShapeInfo_.nValue,
              ctx.matmulPerf.mmShapeInfo_.kValue, cut.shortTileAtBack, cut.longTileLen, cut.numLongTile,
              cut.shortTileLen, cut.numShortTile);
    // §4.37 诊断: 实际切分 N 轮 vs 不切的总通信时(含每轮启动开销 b 与一次性 c), 差值=切分启动税
    OPS_LOG_D("AllGatherMatmul", "Comm total time: %lu rounds actual %.3fus vs no-cut %.3fus (round tax %.3fus)",
              cut.totalTileCnt, ctx.commTimeActualUs, ctx.commTimeNoCutUs, ctx.commTimeActualUs - ctx.commTimeNoCutUs);
    if (baseMOut != nullptr) {
        *baseMOut = ctx.matmulPerf.GetBaseM();
    }
    return cut;
}

// §4.29 全家族分派: 通信标定表命中 → fit-balance; miss(未标定 rank) → 核利用率优先(op 侧自备
// 全量 mm tiling 后调 GetCommTilingFromV3)。本函数只判断标定命中, COF 路径在各 tiling 类实现。
inline bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine)
{
    const auto &rows = (commEngine == TilingCommEngine::CCU) ? CALIBRATION.ccu.rows : CALIBRATION.aicpu.rows;
    for (const auto &row : rows) {
        if (row.rankDim == 0) { // 哨兵截断(与 InitCtx 同约定)
            break;
        }
        if (row.rankDim == rankDim) {
            return true;
        }
    }
    return false;
}

// COF 结果(head 在前/tail 在后) → CutResult(long/short) 映射; 短块(尾块)后置
inline CutResult CutResultFromCof(const CommTilingData &data)
{
    CutResult r;
    r.shortTileAtBack = true;
    r.longTileLen = data.splitAxisTileSize;
    r.numLongTile = data.splitAxisTileCnt;
    r.shortTileLen = data.splitAxisTailSize;
    r.numShortTile = data.splitAxisTailCnt;
    r.totalTileCnt = data.splitAxisTileCnt + data.splitAxisTailCnt;
    return r;
}

// 新旧切换开关: 置 false 并重新编译即整体回退老类路径（出问题时的回退手段）
constexpr bool kUseNewEngine = true;

// 对外唯一入口：两个 GetTilingResult 调用点共用（commEngine 决定通信拟合选表; 三引擎
// 统一分派模式——调用点用 ParseCommEngine(commMode_) 解析后传入(global 符号)）
inline CutResult GetFitBalanceCut(const mc2tiling::TilingArgs &args, TilingCommEngine commEngine,
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

} // namespace optiling

#endif // __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
