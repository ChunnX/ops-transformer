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
 * 内容三段（迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md §4.9 统一接口模式）:
 *   1. 本算子标定数据 —— 客户更新拟合结果只改本文件的表, 不碰任何功能代码;
 *   2. 算子特有钩子（原派生类 override 的逐行拷贝, 常量改 apace::fit_based_balance 限定）;
 *   3. 组装与新旧切换: GetFitBalanceCut(args) 一行调用; kUseApace 置 false 即回老类
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

// ====== 新旧类型边界转换（迁移胶水, 迁移收尾后随老类一并删除） ======
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

// ============================================================================
// 一、本算子标定数据（客户唯一编辑面, 纯数据无逻辑）
// ====== 客户标定修改点：更新拟合结果只需修改下面的表，不碰任何功能代码 ======
//
// 数据来源与口径（当前值为旧公共层 ARCH35_FITTING_PARAMETER_MAP /
// CUBE_CALC_PER_CYCLE_MAP / L2_PARAMETER_MAP / TILING_BEST_BASE_MAP 中本算子
// 可达行的原样切片, 2026-09-14 提取; 重新标定后按同 schema 回填新值即可）:
//   - 通信表 key: "{soc}_{kernelType}_{topoType}_{rankDim}", 本算子 kernelType 恒为
//     ALL_GATHER(1)、soc 恒为 SOC950(4)、topo 由调用点传入（恒 STANDARD_CARD(0)）;
//     可达 key "4_1_0_4"(950 标卡 4P); 其余 rankDim 未标定 → defaultParams 回退
//     + rankDim 折算因子缩放（与旧全量表 miss 行为一致）;
//   - matmul 表 key: "{soc}_{calcType}_{aDtype}_{bDtype}_{cDtype}"(字节数),
//     量化(QUANT=1) 1B 输入可达 "4_1_1_1_2"; 未命中 → default* 回退。
// ============================================================================

// "4_1_0_4": 950 AllGather 标卡 4P —— 线性段 y = 5.841842*x + 8.590666 (us)
static const apace::fit_based_balance::CommFitRow ARCH35_COMM_FIT_ROWS[] = {
    {"4_1_0_4", {0, 0, 5.841842, 8.590666, 0, 0, 0, 0, 0, 0, 0, 0}},
};

// 未标定 key 的回退行（旧 ARCH35 表 DEFAULT_KEY "0_0_0_0" 内容, 勿删——
// rankDim=2/8 等未标定形态依赖此行走折算因子缩放路径）
static const apace::fit_based_balance::HCCLFittingParameters ARCH35_COMM_FIT_DEFAULT = {
    0.032, 1.371, 2.59568, 5.8887, 0, 2.86702, 5.608, 5.6997, 9.448, 0, 0, 0};

static const apace::fit_based_balance::CommFitTable ARCH35_COMM_FIT_TABLE = {
    ARCH35_COMM_FIT_ROWS, sizeof(ARCH35_COMM_FIT_ROWS) / sizeof(ARCH35_COMM_FIT_ROWS[0]), ARCH35_COMM_FIT_DEFAULT};

// 有效算力(MACs/cycle): 量化路径 "4_1_1_1_2" → 8192; 半精度/其他 key 未命中 → 默认 4096
static const apace::fit_based_balance::CubeCalcRow CUBE_CALC_ROWS[] = {
    {"4_1_1_1_2", 8192},
};
static const double CUBE_CALC_DEFAULT = 4096;

// L2 档位利用率: 量化与半精度两行可达; 其余 key 未命中 → 默认行
static const apace::fit_based_balance::L2ParamRow L2_PARAM_ROWS[] = {
    {"4_0_2_2_2", {80, 128, 0.95, 0.85, 0.75}},
    {"4_1_1_1_2", {80, 128, 0.95, 0.85, 0.75}},
};
static const apace::fit_based_balance::L2CacheEstimateParameters L2_PARAM_DEFAULT = {128, 192, 0.85, 0.75, 0.65};

// arch35 最优基本块（SOC950 → 256/256/128）
static const apace::fit_based_balance::BestBaseRow BEST_BASE_ROWS[] = {
    {apace::fit_based_balance::SocVersion::SOC950, {256, 256, 128}},
};
static const apace::fit_based_balance::TilingBestBaseBlock BEST_BASE_DEFAULT = {128, 256, 64};

static const apace::fit_based_balance::MatmulPerfTable MATMUL_PERF_TABLE = {
    CUBE_CALC_ROWS, sizeof(CUBE_CALC_ROWS) / sizeof(CUBE_CALC_ROWS[0]), CUBE_CALC_DEFAULT,
    L2_PARAM_ROWS,  sizeof(L2_PARAM_ROWS) / sizeof(L2_PARAM_ROWS[0]),   L2_PARAM_DEFAULT,
    BEST_BASE_ROWS, sizeof(BEST_BASE_ROWS) / sizeof(BEST_BASE_ROWS[0]), BEST_BASE_DEFAULT};

// ============================================================================
// 二、算子特有钩子（= 老 all_gather_fit_balance_tiling.cpp 逐行拷贝, 状态存本结构体）
// ============================================================================

constexpr uint64_t FP4_EXPANSION_RATIO_AG = 2;
constexpr double FP4_COMPUTE_PER_CYCLE_RATIO_AG = 4;
constexpr static double MM_EXPANSION_TIME = 40;
constexpr static double COMM_EXPANSION_TIME = 40;

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
        if (isCalcCommBalance && (mmCost > MM_EXPANSION_TIME) && (commCost > COMM_EXPANSION_TIME)) {
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
// 三、组装与新旧切换
// ============================================================================

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
        ctx.commPerf.SetCommDtypeSizeExpansionFraction(FP4_EXPANSION_RATIO_AG);
        ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1;
        ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
        ctx.mmInfo.inMatrixADtypeSize = 1;
        ctx.mmInfo.inMatrixBDtypeSize = 1;
        // fp4属量化dtype, 但公共层 SetCalcType 不识别fp4(ConvertGeTypeToMmType无fp4映射→DT_MAX),
        // 致 calcType_=FP16、查表落空→computesPerCycle 退回默认4096
        // (fp8 拿到 8192), totalMatmulTime 被高估4x。此处把fp4归类为QUANT并覆盖 computesPerCycle
        // =4*COMPUTES_PER_CYCLE(=16384);。GetMachineParameters 已在 InitCtx 按FP16跑过, 此处覆盖。
        ctx.matmulPerf.calcType_ = fb::MatmulCalcType::QUANT;
        ctx.matmulPerf.mmShapeInfo_.computesPerCycle =
            FP4_COMPUTE_PER_CYCLE_RATIO_AG * fb::MatmulPerformance::COMPUTES_PER_CYCLE;
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
