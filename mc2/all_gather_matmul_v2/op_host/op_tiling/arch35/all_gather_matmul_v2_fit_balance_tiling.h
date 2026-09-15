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
 *   一、客户标定常量区 —— **重新标定后只改本区的 constexpr 值**，其余分区勿动;
 *   二、标定表组装 —— 仅包装上方常量，无任何魔数;
 *   三、算子特有钩子（原派生类 override 的逐行拷贝）;
 *   四、组装与新旧切换: GetFitBalanceCut(args) 一行调用; kUseApace 置 false 即回老类
 *      （老类 all_gather_fit_balance_tiling.h 原样保留）。
 *
 * 公共逻辑（模型/切分引擎/调度流）在 apace/tiling/fit_based_balance_tiling.h,
 * 本文件不重复任何公共实现。
 */

#ifndef __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__

#pragma once

#include "all_gather_fit_balance_tiling.h" // 老类(回退分支保留, 迁移收尾后随 P4 删除)
#include "apace/tiling/fit_based_balance_tiling.h"
#include "mc2_log.h"
#include "op_host/op_tiling/formulaic_tiling_datatype.h" // 旧 CutResult/KernelType/TopoType(对外接口沿用旧类型)
#include "op_host/op_tiling/matmul_formulaic_tiling.h"   // mc2tiling::TilingArgs

namespace agmm_v2_fit_balance {

// ============================================================================
// 一、客户标定常量区 —— 重新标定后只改本区，各常量的物理含义/单位/来源见行尾注释
//     （当前值提取自旧公共层全量表的本算子可达行, 2026-09-14）
// ============================================================================

// ---- 1. 通信拟合（已标定形态: 950 标卡 4P）----
// 标定产出: 单轮通信时延 y(us) = 斜率·x + 截距, x = 单轮单 rank 发送量(MB)
// 注: 本组数据 AICPU/CCU 两种通信模式共用(旧 ARCH35 表 key 无通信引擎维度),
//     模式差异由调用点的 AdjustCutResultForAICPU/CCU 后置修正吸收
//     (AICPU: 轮数压至 4 减少每轮固定开销; CCU: 按 HCCL buffer 容量上限约束单轮数据量)。
constexpr const char *AGMM_COMM_4P_KEY =
    "4_1_0_4"; // 查表 key: soc(4=950)_kernelType(1=ALL_GATHER)_topo(0=标卡)_rankDim(4)
constexpr double AGMM_COMM_4P_SLOPE_US_PER_MB = 5.841842; // 拟合斜率 (us/MB)
constexpr double AGMM_COMM_4P_INTERCEPT_US = 8.590666;    // 拟合截距 (us), 含每轮同步固定开销

// ---- 1b. 8P 占位（未标定: 常量已定义但表区行未接线, 8P 请求仍走 DEFAULT 回退行+折算缩放。
//      标定后填入实测值并取消表区该行注释即生效, 不碰任何功能代码。
//      key 的 topo 段: 本算子调用点 topo 硬编码 STANDARD_CARD(0) 故为 "4_1_0_8";
//      若将来调用点改为按实际拓扑传 EIGHT_P(1), key 需同步改为 "4_1_1_8"）----
constexpr const char *AGMM_COMM_8P_KEY = "4_1_0_8"; // soc(4)_kernelType(1)_topo(0)_rankDim(8)
constexpr double AGMM_COMM_8P_SLOPE_US_PER_MB = 0;  // TODO(8P标定): 填实测拟合斜率 (us/MB)
constexpr double AGMM_COMM_8P_INTERCEPT_US = 0;     // TODO(8P标定): 填实测拟合截距 (us)

// ---- 2. 通信拟合（未标定形态回退行: rankDim=2/8 等走此行 + rankDim 折算因子缩放, 平台级数据, 通常勿动）----
constexpr double AGMM_COMM_DEFAULT_SMALL_BOUNDARY_MB = 0.032; // 小包段上界 (MB)
constexpr double AGMM_COMM_DEFAULT_LARGE_BOUNDARY_MB = 1.371; // 线性段下界 (MB)
constexpr double AGMM_COMM_DEFAULT_SLOPE_US_PER_MB = 2.59568; // 线性段斜率 (us/MB)
constexpr double AGMM_COMM_DEFAULT_INTERCEPT_US = 5.8887;     // 线性段截距 (us)
constexpr double AGMM_COMM_DEFAULT_PARABOLIC_B = 2.86702; // 小包段 y = a·x²+b·x+c 之 b (本行 a=0, 退化为线性)
constexpr double AGMM_COMM_DEFAULT_PARABOLIC_C = 5.608;           // 小包段常数项 c (us)
constexpr double AGMM_COMM_DEFAULT_INVERSE_BOUNDARY1_US = 5.6997; // 反解(t→x)分段下界 (us)
constexpr double AGMM_COMM_DEFAULT_INVERSE_BOUNDARY2_US = 9.448;  // 反解(t→x)分段上界 (us)

// ---- 3. 切分决策阈值（通信侧鼓励细分的门控, 表E 口径）----
constexpr double AGMM_MM_EXPANSION_TIME_US = 40;   // 短块减半仅当半块计算耗时 > 此值
constexpr double AGMM_COMM_EXPANSION_TIME_US = 40; // 且半块通信耗时 > 此值

// ---- 4. matmul 有效算力（标定产出, MACs/cycle, 含 cube 利用率）----
constexpr const char *AGMM_MM_KEY_QUANT =
    "4_1_1_1_2"; // 查表 key: soc(4)_calcType(1=QUANT)_dtype 1B/1B/2B(fp8/fp4 量化)
constexpr const char *AGMM_MM_KEY_FP16 = "4_0_2_2_2";   // 查表 key: soc(4)_calcType(0=FP16)_dtype 2B/2B/2B(半精度)
constexpr double AGMM_MM_QUANT_MACS_PER_CYCLE = 8192;   // 量化路径有效算力
constexpr double AGMM_MM_DEFAULT_MACS_PER_CYCLE = 4096; // 未命中 key 时的回退算力

// ---- 5. fp4 建模参数（fp4 公共层不识别, 需手工折算）----
constexpr uint64_t AGMM_FP4_EXPANSION_RATIO = 2; // fp4 通信按"1 字节打包 2 元素"建模的分母
constexpr double AGMM_FP4_MACS_RATIO = 4;        // fp4 有效算力 = 此倍数 × 基础算力(4096)

// ---- 6. matmul L2 档位利用率（数据量 ≤PART/≤FULL/超出 三档 cube 利用率）----
constexpr uint64_t AGMM_L2_QUANT_PART_MB = 80;     // 量化路径部分命中阈值 (MB)
constexpr uint64_t AGMM_L2_QUANT_FULL_MB = 128;    // 量化路径完全命中阈值 (MB)
constexpr double AGMM_L2_QUANT_UTIL_PART = 0.95;   // 量化: ≤PART 时 cube 利用率
constexpr double AGMM_L2_QUANT_UTIL_FULL = 0.85;   // 量化: PART~FULL 时 cube 利用率
constexpr double AGMM_L2_QUANT_UTIL_OUT = 0.75;    // 量化: 超出 L2 时 cube 利用率
constexpr uint64_t AGMM_L2_FP16_PART_MB = 80;      // 半精度路径部分命中阈值 (MB)
constexpr uint64_t AGMM_L2_FP16_FULL_MB = 128;     // 半精度路径完全命中阈值 (MB)
constexpr double AGMM_L2_FP16_UTIL_PART = 0.95;    // 半精度: ≤PART 时 cube 利用率
constexpr double AGMM_L2_FP16_UTIL_FULL = 0.85;    // 半精度: PART~FULL 时 cube 利用率
constexpr double AGMM_L2_FP16_UTIL_OUT = 0.75;     // 半精度: 超出 L2 时 cube 利用率
constexpr uint64_t AGMM_L2_DEFAULT_PART_MB = 128;  // 回退档部分命中阈值 (MB)
constexpr uint64_t AGMM_L2_DEFAULT_FULL_MB = 192;  // 回退档完全命中阈值 (MB)
constexpr double AGMM_L2_DEFAULT_UTIL_PART = 0.85; // 回退: ≤PART 时 cube 利用率
constexpr double AGMM_L2_DEFAULT_UTIL_FULL = 0.75; // 回退: PART~FULL 时 cube 利用率
constexpr double AGMM_L2_DEFAULT_UTIL_OUT = 0.65;  // 回退: 超出 L2 时 cube 利用率

// ---- 7. matmul 最优基本块（硬件结构性参数, 通常勿动）----
constexpr uint32_t AGMM_BEST_BASE_M = 256;    // SOC950 最优 baseM
constexpr uint32_t AGMM_BEST_BASE_N = 256;    // SOC950 最优 baseN
constexpr uint32_t AGMM_BEST_BASE_K = 128;    // SOC950 最优 baseK
constexpr uint32_t AGMM_DEFAULT_BASE_M = 128; // 非 950 soc 回退 baseM
constexpr uint32_t AGMM_DEFAULT_BASE_N = 256; // 非 950 soc 回退 baseN
constexpr uint32_t AGMM_DEFAULT_BASE_K = 64;  // 非 950 soc 回退 baseK

// ============================================================================
// 二、标定表组装（勿动: 仅包装上方常量; 新增标定形态时在此按同 schema 加行）
// ============================================================================

static const apace::fit_based_balance::CommFitRow ARCH35_COMM_FIT_ROWS[] = {
    {AGMM_COMM_4P_KEY,
     apace::fit_based_balance::MakeLinearCommFit(AGMM_COMM_4P_SLOPE_US_PER_MB, AGMM_COMM_4P_INTERCEPT_US)},
    // 8P 行占位: 标定后取消下行注释即生效(未接线时 8P 走 DEFAULT 回退行+折算缩放)
    // {AGMM_COMM_8P_KEY, apace::fit_based_balance::MakeLinearCommFit(AGMM_COMM_8P_SLOPE_US_PER_MB,
    //                                                                AGMM_COMM_8P_INTERCEPT_US)},
};

// 未标定 key 的回退行（勿删: rankDim=2/8 等未标定形态依赖此行走折算因子缩放路径）
static const apace::fit_based_balance::HCCLFittingParameters ARCH35_COMM_FIT_DEFAULT = {
    AGMM_COMM_DEFAULT_SMALL_BOUNDARY_MB,
    AGMM_COMM_DEFAULT_LARGE_BOUNDARY_MB,
    AGMM_COMM_DEFAULT_SLOPE_US_PER_MB,
    AGMM_COMM_DEFAULT_INTERCEPT_US,
    0 /* 抛物线 a: 本行未启用 */,
    AGMM_COMM_DEFAULT_PARABOLIC_B,
    AGMM_COMM_DEFAULT_PARABOLIC_C,
    AGMM_COMM_DEFAULT_INVERSE_BOUNDARY1_US,
    AGMM_COMM_DEFAULT_INVERSE_BOUNDARY2_US,
    0,
    0,
    0 /* 反解抛物线参数: 本行未启用 */};

static const apace::fit_based_balance::CommFitTable ARCH35_COMM_FIT_TABLE = {
    ARCH35_COMM_FIT_ROWS, sizeof(ARCH35_COMM_FIT_ROWS) / sizeof(ARCH35_COMM_FIT_ROWS[0]), ARCH35_COMM_FIT_DEFAULT};

static const apace::fit_based_balance::CubeCalcRow CUBE_CALC_ROWS[] = {
    {AGMM_MM_KEY_QUANT, AGMM_MM_QUANT_MACS_PER_CYCLE},
};

static const apace::fit_based_balance::L2ParamRow L2_PARAM_ROWS[] = {
    {AGMM_MM_KEY_FP16,
     {AGMM_L2_FP16_PART_MB, AGMM_L2_FP16_FULL_MB, AGMM_L2_FP16_UTIL_PART, AGMM_L2_FP16_UTIL_FULL,
      AGMM_L2_FP16_UTIL_OUT}},
    {AGMM_MM_KEY_QUANT,
     {AGMM_L2_QUANT_PART_MB, AGMM_L2_QUANT_FULL_MB, AGMM_L2_QUANT_UTIL_PART, AGMM_L2_QUANT_UTIL_FULL,
      AGMM_L2_QUANT_UTIL_OUT}},
};

static const apace::fit_based_balance::BestBaseRow BEST_BASE_ROWS[] = {
    {apace::fit_based_balance::SocVersion::SOC950, {AGMM_BEST_BASE_M, AGMM_BEST_BASE_N, AGMM_BEST_BASE_K}},
};

static const apace::fit_based_balance::MatmulPerfTable MATMUL_PERF_TABLE = {
    CUBE_CALC_ROWS,
    sizeof(CUBE_CALC_ROWS) / sizeof(CUBE_CALC_ROWS[0]),
    AGMM_MM_DEFAULT_MACS_PER_CYCLE,
    L2_PARAM_ROWS,
    sizeof(L2_PARAM_ROWS) / sizeof(L2_PARAM_ROWS[0]),
    {AGMM_L2_DEFAULT_PART_MB, AGMM_L2_DEFAULT_FULL_MB, AGMM_L2_DEFAULT_UTIL_PART, AGMM_L2_DEFAULT_UTIL_FULL,
     AGMM_L2_DEFAULT_UTIL_OUT},
    BEST_BASE_ROWS,
    sizeof(BEST_BASE_ROWS) / sizeof(BEST_BASE_ROWS[0]),
    {AGMM_DEFAULT_BASE_M, AGMM_DEFAULT_BASE_N, AGMM_DEFAULT_BASE_K}};

// ============================================================================
// 三、算子特有钩子（= 老 all_gather_fit_balance_tiling.cpp 逐行拷贝, 状态存本结构体）
// ============================================================================

struct AgMmV2Hooks {
    double frontMMTime = 0;

    void EstimateMMCommTime(apace::fit_based_balance::FitBasedBalanceCtx &ctx)
    {
        ctx.matmulPerf.FindCubeUtil(ctx.rankTileNum);
        ctx.matmulPerf.GetMatmulGradient();

        // Find total matmul time and comm time
        double totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.mmInfo.mValue, ctx.rankDim);
        double totalTpTime = ctx.commPerf.CommTime(ctx.mmInfo.mValue);
        double frontUtil = ctx.matmulPerf.FindCubeUtilByL2Usage(ctx.mmInfo.mValue, 1);
        if (std::abs(frontUtil) > 1e-10) {
            frontMMTime = ctx.matmulPerf.MatmulTime(ctx.mmInfo.mValue, 1) / frontUtil;
        } else {
            frontMMTime = ctx.matmulPerf.MatmulTime(ctx.mmInfo.mValue, 1);
        }
        ctx.tilingM.cutRes.shortTileAtBack = true;
        ctx.ratioCalcComm = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));

        uint64_t sizeOfComm = ctx.mmInfo.mValue * ctx.mmInfo.kValue * (ctx.rankDim - 1) /
                              apace::fit_based_balance::ONE_MBYTE * ctx.commPerf.GetCommDTypeSize();
        OPS_LOG_D("AllGatherMatmul",
                  "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
                  "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
                  "rankDim_ %lu, rankTile %lu",
                  ctx.mmInfo.mValue, ctx.mmInfo.nValue, ctx.mmInfo.kValue, ctx.matmulPerf.cubeUtil_, sizeOfComm,
                  totalMatmulTime, totalTpTime, ctx.tilingM.GetMinLen(), ctx.tilingM.GetAlignLength(),
                  ctx.commPerf.commTimeFactor_, ctx.rankDim, ctx.rankTileNum);
    }

    void SetLongTileLen(apace::fit_based_balance::FitBasedBalanceCtx &ctx)
    {
        // Long tile should overlap the cost time of local tile
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(frontMMTime);
    }

    void SetShortTileLen(apace::fit_based_balance::FitBasedBalanceCtx &ctx)
    {
        // Encourage split more if the comm and calc is balanced and the cost of cutLen is sufficiently high
        bool isCalcCommBalance = ctx.ratioCalcComm < apace::fit_based_balance::CALC_COMM_RATIO;
        uint64_t cutLen = ctx.tilingM.GetAlignLength() / apace::fit_based_balance::TWO;
        double mmCost = ctx.matmulPerf.MatmulTime(cutLen, ctx.rankDim);
        double commCost = ctx.commPerf.CommTime(cutLen);
        if (isCalcCommBalance && (mmCost > AGMM_MM_EXPANSION_TIME_US) && (commCost > AGMM_COMM_EXPANSION_TIME_US)) {
            ctx.tilingM.SetAlignLength(cutLen);
            ctx.tilingM.SetMinLenByMin(cutLen);
        }

        ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
        ctx.tilingM.cutRes.numShortTile = 1U;
    }

    void AdjustLongShortTileLen(apace::fit_based_balance::FitBasedBalanceCtx &ctx)
    {
        ctx.tilingM.FitTileLengthDiscrete(false, true, true);
        // When the long and short tiles are equal, the long and short pieces become one.
        if (ctx.tilingM.cutRes.shortTileLen == ctx.tilingM.cutRes.longTileLen) {
            ctx.tilingM.cutRes.shortTileLen = 0U;
            ctx.tilingM.cutRes.numShortTile = 0U;
            ctx.tilingM.cutRes.numLongTile++;
        }
    }
};

// ============================================================================
// 四、边界转换 + 组装与新旧切换
// ============================================================================

inline apace::fit_based_balance::TilingArgs ToApaceTilingArgs(const mc2tiling::TilingArgs &args)
{
    apace::fit_based_balance::TilingArgs out{};
    out.cmdType = static_cast<apace::fit_based_balance::AicpuComType>(args.cmdType);
    out.rankDim = args.rankDim;
    out.usedCoreNum = args.usedCoreNum;
    out.orgMValue = args.orgMValue;
    out.orgNValue = args.orgNValue;
    out.orgKValue = args.orgKValue;
    out.batchValue = args.batchValue;
    out.mValue = args.mValue;
    out.kValue = args.kValue;
    out.nValue = args.nValue;
    out.baseMLimit = args.baseMLimit;
    out.inputDtypeSize = args.inputDtypeSize;
    out.outputDtypeSize = args.outputDtypeSize;
    out.aicCoreNum = args.aicCoreNum;
    out.commTurn = args.commTurn;
    out.rankTileNum = args.rankTileNum;
    out.commAlg = args.commAlg;
    out.isATrans = args.isATrans;
    out.isBTrans = args.isBTrans;
    out.enablePad = args.enablePad;
    out.enableSplitK = args.enableSplitK;
    out.isBias = args.isBias;
    out.isStorageGather = args.isStorageGather;
    out.isLocal = args.isLocal;
    out.geAType = args.geAType;
    out.geBType = args.geBType;
    out.geCType = args.geCType;
    out.geBiasType = args.geBiasType;
    out.antiquantscaleDType = args.antiquantscaleDType;
    out.aType = args.aType;
    out.bType = args.bType;
    out.cType = args.cType;
    out.biasType = args.biasType;
    return out;
}

inline CutResult ToLegacyCut(const apace::fit_based_balance::CutResult &cut)
{
    CutResult out{};
    out.shortTileAtBack = cut.shortTileAtBack;
    out.totalTileCnt = cut.totalTileCnt;
    out.numLongTile = cut.numLongTile;
    out.numShortTile = cut.numShortTile;
    out.longTileLen = cut.longTileLen;
    out.shortTileLen = cut.shortTileLen;
    return out;
}

// apace 路径: 上下文构造 + 算子后置初始化(= 老类构造函数体逐行对应) + 统一调度
// baseMOut(可空): 回传 matmul 基本块(调用点 AdjustCutResultForAICPU 需要)
inline CutResult ScheduleApace(const mc2tiling::TilingArgs &args, uint32_t *baseMOut = nullptr)
{
    namespace fb = apace::fit_based_balance;
    fb::TilingArgs apaceArgs = ToApaceTilingArgs(args);
    auto ctx = fb::InitCtx(apaceArgs, fb::KernelType::ALL_GATHER, ARCH35_COMM_FIT_TABLE, MATMUL_PERF_TABLE,
                           fb::TopoType::STANDARD_CARD, fb::SocVersion::SOC950);
    ctx.commPerf.SetCommShapeLen(ctx.mmInfo.kValue);
    // mxfp4(DT_FLOAT4_E2M1)每元素占0.5字节，uint64_t无法表达0.5
    // 此处按"1字节打包2个fp4元素"建模：通信侧用SetCommDTypeSize(1)+ExpansionFraction(2)表达0.5字节/元素
    // (复用HCCL现成子字节机制，CommTime/InverseCommTime/GetLinearThresholdLen均经GetRealDtypeSizes()取0.5)；
    // matmul侧inMatrixADtypeSize置1(L2占用按2x高估，避免0导致memUsage/kLength退化为0)。
    if (apaceArgs.geAType == ge::DataType::DT_FLOAT4_E2M1) {
        ctx.commPerf.SetCommDTypeSize(1);
        ctx.commPerf.SetCommDtypeSizeExpansionFraction(AGMM_FP4_EXPANSION_RATIO);
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
        ctx.mmInfo.inMatrixADtypeSize = 1;
        ctx.mmInfo.inMatrixBDtypeSize = 1;
        // fp4属量化dtype, 但公共层 SetCalcType 不识别fp4(ConvertGeTypeToMmType无fp4映射→DT_MAX),
        // 致 calcType_=FP16、查表落空→computesPerCycle 退回默认
        // (fp8 拿到 8192), totalMatmulTime 被高估4x。此处把fp4归类为QUANT并覆盖 computesPerCycle
        // = 4×基础算力(=16384)。GetMachineParameters 已在 InitCtx 按FP16跑过, 此处覆盖。
        ctx.matmulPerf.calcType_ = fb::MatmulCalcType::QUANT;
        ctx.matmulPerf.mmShapeInfo_.computesPerCycle = AGMM_FP4_MACS_RATIO * fb::MatmulPerformance::COMPUTES_PER_CYCLE;
        ctx.mmInfo.computesPerCycle = ctx.matmulPerf.mmShapeInfo_.computesPerCycle;
    } else {
        ctx.commPerf.SetCommDTypeSize(ctx.mmInfo.inMatrixADtypeSize);
    }
    ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.GetBaseM());
    ctx.tilingM.SetAlignLength(ctx.matmulPerf.GetBaseM());

    AgMmV2Hooks hooks;
    CutResult cut = ToLegacyCut(fb::Schedule(ctx, hooks));
    if (baseMOut != nullptr) {
        *baseMOut = ctx.matmulPerf.GetBaseM();
    }
    return cut;
}

// 新旧切换开关: 置 false 并重新编译即整体回退老类路径（出问题时的回退手段）
constexpr bool kUseApace = true;

// 对外唯一入口：两个 GetTilingResult 调用点共用
inline CutResult GetFitBalanceCut(const mc2tiling::TilingArgs &args, uint32_t *baseMOut = nullptr)
{
    if (kUseApace) {
        return ScheduleApace(args, baseMOut);
    }
    AllGatherMMFitBalanceTiling tileFormulate(args, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    if (baseMOut != nullptr) {
        *baseMOut = tileFormulate.matmulPerf_.GetBaseM();
    }
    return tileFormulate.GetTiling();
}

} // namespace agmm_v2_fit_balance

#endif // __ALL_GATHER_MATMUL_V2_FIT_BALANCE_TILING_H__
