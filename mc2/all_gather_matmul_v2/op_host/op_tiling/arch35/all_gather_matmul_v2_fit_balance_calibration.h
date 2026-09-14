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
 * \file all_gather_matmul_v2_fit_balance_calibration.h
 * \brief AllGatherMatmulV2 fit-balance 标定数据（客户唯一编辑面, 纯数据无逻辑）
 *
 * ====== 客户标定修改点：更新拟合结果只需修改本文件，不碰任何功能代码 ======
 *
 * 数据来源与口径（当前值为旧公共层 ARCH35_FITTING_PARAMETER_MAP /
 * CUBE_CALC_PER_CYCLE_MAP / L2_PARAMETER_MAP / TILING_BEST_BASE_MAP 中本算子
 * 可达行的原样切片, 2026-09-14 提取; 重新标定后按同 schema 回填新值即可）：
 *
 * 1. 通信拟合表（ARCH35_COMM_FIT）
 *    - key 规则: "{soc}_{kernelType}_{topoType}_{rankDim}"，本算子 kernelType
 *      恒为 ALL_GATHER(1)、soc 恒为 SOC950(4)、topo 由调用点传入（当前恒为
 *      STANDARD_CARD(0)），rankDim 随运行环境；
 *    - 可达 key: "4_1_0_4"（950 标卡 4P AllGather, 命中行）；其余 rankDim
 *      （2/8 等）未标定 → 走 defaultParams 回退 + rankDim 折算因子缩放
 *      （与旧全量表 miss 行为一致）；
 *    - 行内 12 参数为分段时延拟合: y = a*x^2+b*x+c（小包段）/ y = k*x+m（大包段）
 *      及其反解, x = 单轮单 rank 通信量(MB)。
 *
 * 2. matmul 模型参数表（MATMUL_PERF）
 *    - key 规则: "{soc}_{calcType}_{aDtype}_{bDtype}_{cDtype}"（字节数）；
 *    - cubeCalcRows: 有效算力(MACs/cycle)，量化(QUANT=1) 1B 输入可达 "4_1_1_1_2"；
 *    - l2Rows: L2 档位利用率参数，量化与半精度(2B)两行可达；
 *    - key 未命中 → 走 default*（与旧全量表 miss 行为一致）；
 *    - bestBaseRows: arch35 最优基本块。
 */

#ifndef __ALL_GATHER_MATMUL_V2_FIT_BALANCE_CALIBRATION_H__
#define __ALL_GATHER_MATMUL_V2_FIT_BALANCE_CALIBRATION_H__

#pragma once

#include "apace/tiling/legacy/comm_perf_model.h"
#include "apace/tiling/legacy/matmul_perf_model.h"

namespace apace {
namespace legacy {
namespace all_gather_matmul_v2_calibration {

// ====== 通信拟合表（本算子可达行 + 回退默认行） ======
// "4_1_0_4": 950 AllGather 标卡 4P —— 线性段 y = 5.841842*x + 8.590666 (us)
static const CommFitRow ARCH35_COMM_FIT_ROWS[] = {
    {"4_1_0_4", {0, 0, 5.841842, 8.590666, 0, 0, 0, 0, 0, 0, 0, 0}},
};

// 未标定 key 的回退行（旧 ARCH35 表 DEFAULT_KEY "0_0_0_0" 内容, 勿删——
// rankDim=2/8 等未标定形态依赖此行走折算因子缩放路径）
static const HCCLFittingParameters ARCH35_COMM_FIT_DEFAULT = {0.032, 1.371,  2.59568, 5.8887, 0, 2.86702,
                                                              5.608, 5.6997, 9.448,   0,      0, 0};

static const CommFitTable ARCH35_COMM_FIT_TABLE = {
    ARCH35_COMM_FIT_ROWS, sizeof(ARCH35_COMM_FIT_ROWS) / sizeof(ARCH35_COMM_FIT_ROWS[0]), ARCH35_COMM_FIT_DEFAULT};

// ====== matmul 模型参数表（本算子可达行 + 回退默认值） ======
// 有效算力(MACs/cycle): 量化路径 "4_1_1_1_2"(950/QUANT/1B/1B/2B) → 8192;
// 半精度/其他 key 未命中 → defaultComputesPerCycle(4096, 旧默认常量)
static const CubeCalcRow CUBE_CALC_ROWS[] = {
    {"4_1_1_1_2", 8192},
};
static const double CUBE_CALC_DEFAULT = 4096;

// L2 档位利用率: 量化与半精度两行可达; 其余 key 未命中 → DEFAULT_L2 行
static const L2ParamRow L2_PARAM_ROWS[] = {
    {"4_0_2_2_2", {80, 128, 0.95, 0.85, 0.75}},
    {"4_1_1_1_2", {80, 128, 0.95, 0.85, 0.75}},
};
static const L2CacheEstimateParameters L2_PARAM_DEFAULT = {128, 192, 0.85, 0.75, 0.65};

// arch35 最优基本块（SOC950 → 256/256/128; 其余 soc → 默认基本块）
static const BestBaseRow BEST_BASE_ROWS[] = {
    {SocVersion::SOC950, {256, 256, 128}},
};
static const TilingBestBaseBlock BEST_BASE_DEFAULT = {128, 256, 64};

static const MatmulPerfTable MATMUL_PERF_TABLE = {
    CUBE_CALC_ROWS, sizeof(CUBE_CALC_ROWS) / sizeof(CUBE_CALC_ROWS[0]), CUBE_CALC_DEFAULT,
    L2_PARAM_ROWS,  sizeof(L2_PARAM_ROWS) / sizeof(L2_PARAM_ROWS[0]),   L2_PARAM_DEFAULT,
    BEST_BASE_ROWS, sizeof(BEST_BASE_ROWS) / sizeof(BEST_BASE_ROWS[0]), BEST_BASE_DEFAULT};

} // namespace all_gather_matmul_v2_calibration
} // namespace legacy
} // namespace apace

#endif // __ALL_GATHER_MATMUL_V2_FIT_BALANCE_CALIBRATION_H__
