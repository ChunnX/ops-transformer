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
 * 本文件只剩三样东西, 无任何算子特有逻辑（钩子机制已废除, AG 特有逻辑已键控内化引擎）:
 *   一、客户标定数据 CALIBRATION（唯一编辑面, 重新标定只改数值）;
 *   二、一行组装: InitCtx(ALL_GATHER, COMM_FIRST, commEngine, CALIBRATION) + Schedule;
 *   三、kUseNewEngine 新旧切换（置 false 重编译即回老类, 老类原样保留）。
 *
 * 公共逻辑与 AG 键控分支（E4 恒后置/E5 本地首轮重叠/E9 减半鼓励/E19 长块反解/E24
 * localAtFront + D4/D5/D6 胶水)均在 apace/tiling/fit_balance_tiling.h(global)。
 */

#ifndef __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__

#pragma once

#include "all_gather_fit_balance_tiling.h" // 老类(回退分支保留, 迁移收尾后随 P4 删除)
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
            {MatmulDtypeKind::QUANT, false, 1, 1, 2, 8192}, // fp8 量化(1B/1B/2B)
            {MatmulDtypeKind::QUANT, true, 1, 1, 2, 16384}, // fp4 量化(打包 1B/1B/2B, 算力约为 fp8 的 2×)
        },
        /* l2Param: {计算类型, isFp4, A/B/C 字节数, L2 档位参数} */
        {
            {MatmulDtypeKind::FP16, false, 2, 2, 2, {80, 128, 0.95, 0.85, 0.75}},  // 半精度
            {MatmulDtypeKind::QUANT, false, 1, 1, 2, {80, 128, 0.95, 0.85, 0.75}}, // fp8 量化
            {MatmulDtypeKind::QUANT, true, 1, 1, 2, {80, 128, 0.95, 0.85, 0.75}},  // fp4 量化(共用 fp8 档位)
        },
        /* baseBlock: matmul 最优基本块(单值; 未填用默认 {128,256,64}) */
        {256, 256, 128},
    },
    // tunables: 键控分支阈值(E9 减半鼓励门 40/40 等), 缺省=现网值, 通常勿动
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
// baseMOut(可空): 回传 matmul 基本块(调用点 AdjustCutResultForAICPU 需要)
inline CutResult ScheduleFitBalance(const mc2tiling::TilingArgs &args, TilingCommEngine commEngine,
                                    uint32_t *baseMOut = nullptr)
{
    // 引擎分路在 InitCtx 内完成(按 CommEngine 选 CALIBRATION 的 aicpu/ccu 分区);
    // 本算子 kernel 仅 HCCL 两引擎, AIV_URMA 不可达(枚举值保留供三引擎分派框架)
    auto ctx = InitCtx(ToEngineArgs(args), CommPrimitive::ALL_GATHER, CommComputeOrder::COMM_FIRST, commEngine,
                       CALIBRATION, CommTopology::STANDARD_CARD);
    CutResult cut = ToLegacyCut(Schedule(ctx));
    OPS_LOG_D("AllGatherMatmul",
              "Input shape {M, N, K} = {%lu, %lu, %lu}, Final cut: shortTileAtBack %d, longTileLen %lu, "
              "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
              ctx.matmulPerf.mmShapeInfo_.mValue, ctx.matmulPerf.mmShapeInfo_.nValue,
              ctx.matmulPerf.mmShapeInfo_.kValue, cut.shortTileAtBack, cut.longTileLen, cut.numLongTile,
              cut.shortTileLen, cut.numShortTile);
    if (baseMOut != nullptr) {
        *baseMOut = ctx.matmulPerf.GetBaseM();
    }
    return cut;
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
