/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 客户标定数据总集(FitBalanceCalibration, §4.39 拆分):
// 算子侧唯一需要接触的标定类型——CommProfile 结构参数(§4.38 原语语义终结在标定层)
// + 通信拟合行(k/b/c, k 为引擎 x 口径) + matmul 算力档/基本块。
#pragma once

#include "fit_balance_types.h"

// ============================================================================
// 客户标定数据总集（算子侧唯一需要接触的标定类型, 2026-09-15 评审裁定:
// 单一大结构体替代"引擎侧一堆小结构体 + 算子侧一堆具名 constexpr"）
//
// 用法（算子侧标定头, 填一个实例传给 InitCtx 即可）:
//   static const FitBalanceCalibration CALIBRATION = {
//       /* aicpu */  { /* rows */ { {"4_1_0_4", 斜率, 截距} } },
//       /* ccu */    { /* rows */ { {"4_1_0_4", 斜率, 截距} } },
//       /* matmul */ { /* cubeCalc */ { {MatmulDtypeKind, isFp4, aB, bB, cB, 算力} }, ... },
//   };
//
// 填写规则:
//   - 通信行按 rankDim 匹配(未标定形态由 op 层分派走核利用率优先, §4.29——引擎不设回退);
//   - matmul 行按 (calcType, A/B/C 字节数) 匹配, 未填字段取 NSDMI 默认; 行按前缀填, 0 值行=未填;
//   - 通信拟合公式 y = k·x + N·b + c(§4.37): 客户标定产出 k/b/c 三个数。
// ============================================================================

struct FitBalanceCalibration {
    static constexpr size_t MAX_COMM_FIT_ROWS = 4;   // 已标定通信形态数上限
    static constexpr size_t MAX_CUBE_CALC_ROWS = 4;  // matmul 算力档位数上限
    static constexpr size_t MAX_BASE_BLOCK_ROWS = 2; // 最优基本块档位数上限

    // —— 通信结构参数(§4.38: 原语语义终结在标定层——每算子一份, 引擎不区分通信原语) ——
    // 倍数依赖运行时 rankDim 故存模式枚举, 引擎经 RankMulValue() 一处集中换算。
    struct CommProfile {
        CommAxis axis = CommAxis::K;         // 通信承载侧: K=A 侧输入(传 A 行), N=C 侧输出(传 C 行)
        RankMul mmWorksetMul = RankMul::ONE; // mm 工作集 M 跨度倍数(FindCubeUtil): AG=rankDim−1
                                             // (自己的份不经通信, 本地块单独按 1 建模), RS/A2A=rankDim, AR=1
        RankMul mmTotalMul = RankMul::ONE;    // mm 总行数倍数(MatmulTime): AG/RS=rankDim(聚合全量/
                                              // scatter 除前全量), AR/A2A=1
        RankMul commVolumeMul = RankMul::ONE; // 单次通信调用的物理字节倍数(⑥ 的 256MB 上限判定用,
                                              // 与 k 的标定口径融合无关): AG/RS=rankDim(单次调用搬全量),
                                              // AR/A2A=1(单次调用按单份)
    };
    CommProfile profile{};

    // —— 通信拟合（单引擎）——
    struct CommFit {
        struct Row {
            uint32_t rankDim;             // 卡数(4=标卡4P, 8=8P, ...)
            double slopeUsPerMb;          // 带宽斜率 k (us/MB)
            double roundOverheadUs;       // 每轮启动开销 b (us)
            double fixedOverheadUs = 0.0; // 一次性固定开销 c (us)——未标定填 0(阶段1 兼容旧截距口径)
        };
        Row rows[MAX_COMM_FIT_ROWS] = {}; // 已标定行, rankDim=0 表示未填(哨兵截断)
    };
    CommFit aicpu; // AICPU 引擎(comm_mode != "ccu")
    CommFit ccu;   // CCU 引擎(comm_mode == "ccu")

    // —— matmul 性能拟合（对仗 CommFit; 字段序: 用户常改字段在前, 平台默认在后——聚合初始化
    //    可省尾部, 省略的尾部字段自动取 NSDMI 默认值）——
    struct MatmulFit {
        // 键维度裁定(§4.37 缺陷根治, 2026-09-22): 不含 C 侧字节数——glue 恒 cType=aType,
        // 运行时 cBytes 恒等于 aBytes(零信息量冗余维度, 假设/真实口径曾在其上分叉致算力查表缺陷);
        // 物理上 C dtype 只影响写出带宽不影响 MACs/cycle(老表 _1/_2 行同值 8192 佐证)。
        // 未来 outputDtypeSize 口径独立化(真按输出计)且重标定发现 C 影响算力时, 再按独立 CR 加回。
        struct CubeCalc {
            MatmulDtypeKind calcType;
            bool isFp4;              // FP16 / QUANT / ANTI_QUANT
            uint64_t aBytes;         // 输入 A 每元素字节数
            uint64_t bBytes;         // 输入 B 每元素字节数
            double computesPerCycle; // 有效算力 (MACs/cycle, 含 cube 利用率)
        };
        CubeCalc cubeCalc[MAX_CUBE_CALC_ROWS] = {};
        MatmulBaseBlock baseBlock = {128, 256, 64}; // matmul 最优基本块(单值)
        double defaultComputesPerCycle = 4096.0;    // 算力档回退(通常勿动)
    };
    MatmulFit mmFit;
};
