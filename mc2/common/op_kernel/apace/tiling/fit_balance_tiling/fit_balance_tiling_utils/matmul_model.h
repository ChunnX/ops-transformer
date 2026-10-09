/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 计算模型: cube 利用率/梯度/耗时正反解
#pragma once

#include "fit_balance_types.h"

namespace MatmulPerfConst {
constexpr double COMPUTES_PER_CYCLE = 4096;
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
    MatmulDtypeKind calcType_;
    bool isFp4_ = false;
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;

    // 平台参数(算力/基本块/频率)由 InitCtx 从标定注入
    MatmulPerfModel(const FitBalanceTilingArgs& args, double computesPerCycle, MatmulBaseBlock baseBlock,
                    double cyclePerMicroSec)
    {
        mmShapeInfo_.coreNum = args.aicCoreNum;
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
    };

    // dtype 分类: fp4/fp8/HIF8/INT8 → QUANT, 其余 FP16; fp4 经 geAType 识别
    void SetCalcType(const FitBalanceTilingArgs& args)
    {
        if ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
            (args.aType == matmul_tiling::DataType::DT_INT8) || (args.geAType == ge::DataType::DT_FLOAT4_E2M1)) {
            calcType_ = MatmulDtypeKind::QUANT;
        }
    }

    // cube 利用率: util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^mnExp)
    //   调和平均 = (m × shares × n) / (m × shares + n)
    //   mnExp 双值: 稳态 0.7(数据齐/流水满), 本地窗口 0.6(单份数据/流水未满, 更保守)
    double FindCubeUtil(uint64_t mSize, uint64_t shares, double mnExponent)
    {
        double mnharmonicMean = static_cast<double>(mSize * shares * mmShapeInfo_.nValue) /
                                static_cast<double>(mSize * shares + mmShapeInfo_.nValue);
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

    // 稳态口径(设 cubeUtil_ 供 GetMatmulGradient 消费)
    void FindCubeUtil(uint64_t shares)
    {
        cubeUtil_ = FindCubeUtil(mmShapeInfo_.mValue, shares, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT);
    }

    // 每行耗时系数: gradient = max(baseN,n) × max(baseK,k) / (cpc × 频率 × 核数 × util)
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

    // 块耗时: tileSize × rowMul × gradient(不足 baseM 按 baseM 计)
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
