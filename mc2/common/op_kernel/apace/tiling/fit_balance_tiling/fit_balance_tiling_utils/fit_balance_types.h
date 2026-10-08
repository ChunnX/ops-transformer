/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 公共类型/枚举/常量/标定结构
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

constexpr uint64_t BYTES_PER_KB = 1024;
constexpr uint64_t BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB;
constexpr uint64_t MIN_RANK_DIM = 2;

enum class MatmulDtypeKind {
    FP16, // default
    QUANT,
    ANTI_QUANT,
};

struct MatmulBaseBlock {
    uint32_t baseM = 256;
    uint32_t baseN = 256;
    uint32_t baseK = 128;
};

// matmul shape 画像(标定 baseBlock 与算力档注入)
struct MatmulShapeInfo {
    uint64_t mValue;
    uint64_t nValue;
    uint64_t kValue;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    uint64_t inMatrixADtypeSize;
    uint64_t inMatrixBDtypeSize;
    uint64_t outMatrixCDtypeSize;
    uint64_t coreNum;
    double computesPerCycle; // cube 有效算力(MACs/cycle, 标定查表产出)
    double cyclePerMicroSec; // AIC 频率(cycle/us, 标定注入)
};

// 通信模型画像
struct CommModelInfo {
    uint64_t rankDim;                        // 通信域卡数(构造期钳位 ≥2)
    uint64_t commDtypeSize;                  // 通信每元素字节数
    uint64_t commMatrixLen;                  // 通信矩阵另一维(K 或 N)
    uint64_t commDtypeSizeExpansionFraction; // 子字节折算分母(fp4=2)
};

// 通信时间拟合参数: y = k·x + 轮数·b + c
struct CommFitParams {
    double slopeUsPerMb;    // 带宽斜率 k (us/MB)
    double roundOverheadUs; // 每轮启动开销 b (us)
    double fixedOverheadUs; // 一次性固定开销 c (us)
};

// M 轴切分结果: N 个长块 + 1 个短块(或退化等分/单块)
struct CommCutResult {
    bool shortTileAtBack;
    uint64_t totalTileCnt;
    uint64_t numLongTile;
    uint64_t numShortTile;
    uint64_t longTileLen;
    uint64_t shortTileLen;
};

struct CutParams {
    uint64_t mAlignLen;
    uint64_t maxTileLen;
    uint64_t minTileLen;
    uint64_t maxTileCnt;
};

// 通信轮次硬上限(通信 API 的 FFTS 队列大小限制)
constexpr uint64_t MAX_COMM_ROUNDS = 63;

constexpr uint32_t DEFAULT_BASE_M = 128;
constexpr uint32_t DEFAULT_BASE_N = 256;
constexpr uint32_t DEFAULT_BASE_K = 64;

// 通信承载的数据侧(决定通信矩阵另一维与 dtype 字节取值)
enum class CommAxis {
    K, // A 侧(输入): 传 A 矩阵行, 另一维取 kValue
    N, // C 侧(输出): 传 C 矩阵行, 另一维取 nValue
};

// 通信引擎
enum class TilingCommEngine {
    AICPU,
    CCU,
    AIV_URMA,
};

// comm_mode 属性解析: "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含空指针/默认"ai_cpu")→AICPU
inline TilingCommEngine ParseCommEngine(const char* commMode)
{
    if (commMode == nullptr) {
        return TilingCommEngine::AICPU;
    }
    if (strncmp(commMode, "aiv_urma", 8U) == 0) {
        return TilingCommEngine::AIV_URMA;
    }
    if (strncmp(commMode, "ccu", 3U) == 0) {
        return TilingCommEngine::CCU;
    }
    return TilingCommEngine::AICPU;
}

// 引擎调度入参(纯运行时 shape 事实)
struct FitBalanceTilingArgs {
    uint64_t mValue;
    uint64_t nValue;
    uint64_t kValue;
    uint32_t rankDim;
    uint32_t aicCoreNum;
    matmul_tiling::DataType aType; // mm 域 A dtype(量化类别判定)
    ge::DataType geAType;          // ge 域 A dtype(fp4 识别——DT_FLOAT4_E2M1 无 mm 域表达)
    uint64_t inputDtypeSize;       // A 每元素字节数(fp4 时=0)
    uint64_t outputDtypeSize;      // C 每元素字节数
    TilingCommEngine commEngine;
};

// 通算顺序
enum class CommComputeOrder {
    COMPUTE_FIRST, // 计算在前通信在后(输出侧通信: RS/AR/matmul_allto_all)
    COMM_FIRST,    // 通信在前计算在后(输入侧通信: AG/allto_all_matmul)
};

// 本地 mm 块位置(不经通信的数据块在流水线中的位置)
enum class LocalMMPosition {
    FRONT, // 本地块前置: 流水线首端, 短块退到尾部收尾
    BACK,  // 本地块后置: 流水线尾端, 短块前置启动通信
    NONE,  // 无本地块
};

// 标定数据总集: 算子侧唯一编辑面, 一个实例传给 FitBalanceCut。
//
// 用法(算子侧标定头):
//   static const FitBalanceCalibration CALIBRATION = {
//       /* dataflow */ { CommComputeOrder::..., LocalMMPosition::..., bool },
//       /* commRows */ { {TilingCommEngine::..., 卡数, k, b, c}, ... },
//       /* mmFit */    { /* cubeCalc */ { {MatmulDtypeKind, isFp4, aB, bB, 算力} }, ... },
//   };
//
// 填写规则:
//   - 通信行按 (通信引擎, rankDim) 匹配; 未标定形态不写行(miss 走其他切分策略);
//   - matmul 行按 (calcType, isFp4, A/B 字节数) 四维匹配; 行按前缀填, 哨兵截断;
//   - 平台规格(频率)默认 950, 新芯片按规格填写。
struct FitBalanceCalibration {
    static constexpr size_t MAX_COMM_FIT_ROWS = 8;
    static constexpr size_t MAX_CUBE_CALC_ROWS = 4;

    // 数据流形态(算子接入时定死, 引擎据此推导 axis 与倍数)
    struct DataflowShape {
        CommComputeOrder order = CommComputeOrder::COMM_FIRST;
        LocalMMPosition localMMPosition = LocalMMPosition::NONE;
        bool fullVolume = true; // true: mValue 为每卡份(实算 = mValue × rankDim);
                                // false: mValue 已是全量(×1)
    };
    DataflowShape dataflow{};

    // 通信拟合行(引擎维度在行上, 新增引擎加行即可)
    struct CommFitRow {
        TilingCommEngine commEngine;
        uint32_t rankDim;             // 卡数(0=未填行哨兵)
        double slopeUsPerMb;          // 带宽斜率 k (us/MB)
        double roundOverheadUs;       // 每轮启动开销 b (us)
        double fixedOverheadUs = 0.0; // 一次性固定开销 c (us)
    };
    CommFitRow commRows[MAX_COMM_FIT_ROWS] = {};

    // matmul 性能拟合
    struct MatmulFit {
        // 键不含 C 侧字节数: 运行时 cBytes 恒等于 aBytes(零信息量维度)
        struct CubeCalc {
            MatmulDtypeKind calcType;
            bool isFp4;
            uint64_t aBytes;
            uint64_t bBytes;
            double computesPerCycle; // 有效算力 (MACs/cycle)
        };
        CubeCalc cubeCalc[MAX_CUBE_CALC_ROWS] = {};
        MatmulBaseBlock baseBlock = {128, 256, 64};
        double defaultComputesPerCycle = 4096.0;       // 算力档回退
        double cyclePerMicroSec = 1.65 * BYTES_PER_KB; // AIC 频率(cycle/us)
    };
    MatmulFit mmFit;
};
