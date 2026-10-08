/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 计算模型:
// cube 利用率(聚合 0.7/本地窗口 0.6 双口径)/每行梯度/耗时正反解。
#pragma once

#include "fit_balance_types.h"

// ---- matmul 模型(算力/梯度) ----
// 剥离 310P/910_B4 分支与 4 参 FindCubeUtil(含其独占的 K 对齐修正链)——
// 5 个消费算子经 grep 证实仅用 FindCubeUtil。

namespace MatmulPerfConst {
constexpr double COMPUTES_PER_CYCLE = 4096; // 基础算力(查表未命中回退, 标定头回填同值)
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double LOCAL_MM_UTIL_EXPONENT_COEFFICIENT = 0.6; // 本地窗口(单份数据/流水未满)的 MN 指数
constexpr double CUBE_UTIL_EXPONENT_COEFFICIENT = 0.7;
constexpr double MAX_CUBE_UTIL = 0.95;
constexpr double AVERAGE_CUBE_UTIL = 0.75;
} // namespace MatmulPerfConst

class MatmulPerfModel {
public:
    MatmulShapeInfo mmShapeInfo_;
    MatmulDtypeKind calcType_; // Distinguish the non-quantization, full-quantization and fake-quantization
    bool isFp4_ = false;       // fp4 输入(geAType==DT_FLOAT4_E2M1), 影响 CubeCalc/L2 查表匹配
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;

    // Constructor(平台参数由 InitCtx 从标定注入: 算力/基本块/频率/L2 容量)
    MatmulPerfModel(const FitBalanceTilingArgs& args, double computesPerCycle, MatmulBaseBlock baseBlock,
                    double cyclePerMicroSec, uint64_t l2CacheBytes)
    {
        mmShapeInfo_.coreNum = args.aicCoreNum; // 每die核数
        mmShapeInfo_.inMatrixADtypeSize = args.inputDtypeSize;
        mmShapeInfo_.inMatrixBDtypeSize = mmShapeInfo_.inMatrixADtypeSize;
        mmShapeInfo_.outMatrixCDtypeSize = args.outputDtypeSize;
        mmShapeInfo_.mValue = args.mValue;
        mmShapeInfo_.nValue = args.nValue;
        mmShapeInfo_.kValue = args.kValue;
        mmShapeInfo_.baseM = baseBlock.baseM;
        mmShapeInfo_.baseN = baseBlock.baseN;
        mmShapeInfo_.baseK = baseBlock.baseK;
        isFp4_ = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
        SetCalcType(args);
        mmShapeInfo_.computesPerCycle = computesPerCycle;
        mmShapeInfo_.cyclePerMicroSec = cyclePerMicroSec;
        mmShapeInfo_.l2CacheBytes = l2CacheBytes;
    };

    // dtype 分类(选行匹配键的推导; 查表选行已在 InitCtx 完成):
    // 950 量化语义: fp4/fp8/HIF8 或 A8W8 量化, 其余按 FP16 建模;
    // fp4 经 geAType 识别
    void SetCalcType(const FitBalanceTilingArgs& args)
    {
        if ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
            (args.aType == matmul_tiling::DataType::DT_INT8) || (args.geAType == ge::DataType::DT_FLOAT4_E2M1)) {
            calcType_ = MatmulDtypeKind::QUANT;
        }
    }
    // cube 利用率(950 调和平均公式):
    //   util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^mnExp)
    //   调和平均 = (mSize × commRankCntForMm × n) / (mSize × commRankCntForMm + n)
    // mnExponent: MN 因子指数, 两种口径——聚合稳态 0.7(数据齐/流水满, 增长快)、
    //   AG 本地首块(front) 0.6(只有本卡份/流水未满, 增长平缓), 是两个口径的真实物理差别
    double FindCubeUtil(uint64_t mSize, uint64_t commRankCntForMm, double mnExponent)
    {
        double mnharmonicMean = static_cast<double>(mSize * commRankCntForMm * mmShapeInfo_.nValue) /
                                static_cast<double>(mSize * commRankCntForMm + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerfConst::KVALUE_THRESHOLD, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated =
            std::min(MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
                     std::pow(mnharmonicMean, mnExponent) / std::pow(MatmulPerfConst::MNVALUE_THRESHOLD, mnExponent));
        double result = MatmulPerfConst::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        return std::min(MatmulPerfConst::MAX_CUBE_UTIL, result);
    }
    // 聚合稳态口径(整 m + 全部流转份数, 设 cubeUtil_ 供 GetMatmulGradient 消费)
    void FindCubeUtil(uint64_t commRankCntForMm)
    {
        cubeUtil_ =
            FindCubeUtil(mmShapeInfo_.mValue, commRankCntForMm, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT);
    }
    void GetMatmulGradient()
    {
        const uint64_t tmpN = std::max<uint64_t>(mmShapeInfo_.baseN, mmShapeInfo_.nValue);
        const uint64_t tmpK = std::max<uint64_t>(mmShapeInfo_.baseK, mmShapeInfo_.kValue);
        matmulGradient_ = (tmpN * tmpK) / (mmShapeInfo_.computesPerCycle * mmShapeInfo_.cyclePerMicroSec *
                                           mmShapeInfo_.coreNum * cubeUtil_);
    }
    uint32_t GetBaseM()
    {
        return mmShapeInfo_.baseM;
    };
    // 性能预测
    double MatmulTime(uint64_t tileSize, uint64_t rowMul)
    {
        tileSize = std::max<uint64_t>(mmShapeInfo_.baseM, tileSize);
        return static_cast<double>(tileSize) * static_cast<double>(rowMul) * matmulGradient_;
    }
    uint64_t InverseMatmulTime(double targetTime, uint64_t rowMul)
    {
        uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rowMul) * matmulGradient_));
        tileSize = std::max<uint64_t>(mmShapeInfo_.baseM, tileSize);
        return tileSize;
    }
};
