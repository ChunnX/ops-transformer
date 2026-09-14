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
 * \file matmul_perf_model.h
 * \brief Matmul 理论耗时模型（旧公式化/拟合配平 tiling 家族的 L1 模型层, matmul 分支）
 *
 * 提取来源（逻辑逐行忠实, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/matmul_performance.h / .cpp          (基类模型 + CUBE_CALC_PER_CYCLE_MAP /
 * L2_PARAMETER_MAP)
 *   - mc2/common/op_host/op_tiling/matmul_performance_arch35.h          (arch35 模型 + TILING_BEST_BASE_MAP)
 *
 * 与旧实现的差异（仅以下两点, 其余逐行一致）:
 *   1. CUBE_CALC_PER_CYCLE_MAP / L2_PARAMETER_MAP / TILING_BEST_BASE_MAP 三张静态表
 *      改为 MatmulPerfTable 注入（行数据由算子侧标定头提供, 查表语义 = 精确 key 匹配）;
 *   2. 移除 OP_LOGW 日志。
 *
 * 注: MatmulFormulaicTiling（mm 侧 TCubeTiling 引擎）未随 P0 下沉——它依赖
 * op_def_registry / mc2_hcom_topo_info 等 op_host 框架头, 且属 mm 切分而非通算切分家族,
 * 待首个消费它的算子迁移时一并提取（迁移方案 §4.1 P3 备注）。
 */

#ifndef __APACE_LEGACY_MATMUL_PERF_MODEL_H__
#define __APACE_LEGACY_MATMUL_PERF_MODEL_H__

#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "legacy_tiling_types.h"

namespace apace {
namespace legacy {

// ====== 以下常量来源: matmul_performance.h 的 namespace MatmulPerformance ======
namespace MatmulPerformance {
constexpr double COMPUTES_PER_CYCLE = 4096;
constexpr double K_UNALIGN_UTIL_RATIO_SOC310P = 0.8;
constexpr double K_UNALIGN_UTIL_RATIO = 0.7;
constexpr double K_UNALIGN_UTIL_RATIO_910_B4 = 0.8;
constexpr uint64_t CACHE_LINE_LEN = 512;
constexpr double HALF_K_UNALIGN_UTIL_RATIO = 0.5;
constexpr uint64_t HALF_CACHE_LINE_LEN = 256;
constexpr uint64_t K_ALIGN_LEN = 128;
constexpr uint64_t MIN_M_SIZE_FACTOR = 3;
constexpr uint64_t MARK_CORE_NUM_SOC910B = 20;
// start: cycle per micro second for different soc version
constexpr double CYCLE_PER_MICRO_SEC = 1.8 * ONE_KBYTE;
constexpr double CYCLE_PER_MICRO_SEC_NPUARCH_3510 = 1.65 * ONE_KBYTE;
constexpr double CYCLE_PER_MICRO_SEC_VERSION310_P = 1.08 * ONE_KBYTE;
// end: cycle per micro second for different soc version
constexpr const char *DEFAULT_KEY_FOR_PAR_MAP = "0_0_2_2_2";
constexpr double UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P = 1.5;
constexpr double UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P = 1.25;
constexpr double UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P_NO_QUANT = 1.5;
constexpr double UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P_NO_QUANT = 0.5;
constexpr uint64_t MIN_MSIZE_QUANT_SOC310P = 1024;
constexpr uint64_t INT8_DTYPE_SIZE = 1;
constexpr uint64_t SMALL_RANKTILE = 2;
constexpr uint64_t BMM_DATASIZE_K_BAR = 2048;
constexpr uint64_t BMM_DATASIZE_SMALL_K = 16 * ONE_GBYTE;
constexpr uint64_t BMM_DATASIZE_LARGE_K = 24 * ONE_GBYTE;
constexpr uint64_t MM_MIN_DATASIZE_NPUARCH_3510 = 5 * ONE_GBYTE;
constexpr uint64_t MM_MIN_DATASIZE_OTHER_SOC = 10 * ONE_GBYTE; // M * N * K >= 10G
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double FRONT_UTIL_EXPONENT_COEFFICIENT = 0.6;
constexpr double CUBE_UTIL_EXPONENT_COEFFICIENT = 0.7;
constexpr double MAX_CUBE_UTIL = 0.95;
constexpr double AVERAGE_CUBE_UTIL = 0.75;
} // namespace MatmulPerformance

// ====== 模型参数注入 schema（行数据由算子侧标定头维护） ======
// key 规则见 MatmulPerformanceModel::GetCalcTypeString: "soc_calcType_aDtype_bDtype_cDtype"
struct CubeCalcRow {
    const char *key;
    double computesPerCycle;
};

struct L2ParamRow {
    const char *key;
    L2CacheEstimateParameters params;
};

struct BestBaseRow {
    SocVersion soc;
    TilingBestBaseBlock block;
};

struct MatmulPerfTable {
    // 算力查表（旧 CUBE_CALC_PER_CYCLE_MAP）
    const CubeCalcRow *cubeCalcRows;
    size_t cubeCalcCount;
    double defaultComputesPerCycle; // key 未命中回退（旧实现为常量 COMPUTES_PER_CYCLE, 由标定头回填同值）
    // L2 档位查表（旧 L2_PARAMETER_MAP）
    const L2ParamRow *l2Rows;
    size_t l2Count;
    L2CacheEstimateParameters defaultL2Params; // key 未命中回退（旧 DEFAULT_KEY 行内容 {128, 192, 0.85, 0.75, 0.65}）
    // arch35 最优基本块查表（旧 TILING_BEST_BASE_MAP）
    const BestBaseRow *bestBaseRows;
    size_t bestBaseCount;
    TilingBestBaseBlock defaultBaseBlock; // soc 未命中回退（旧实现为 {BASE_BLOCK_M/N/K}, 由标定头回填同值）
};

class MatmulPerformanceModel {
public:
    MatmulParameters mmShapeInfo_;
    MatmulCalcType calcType_; // Distinguish the non-quantization, full-quantization and fake-quantization
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;
    uint64_t mmMinDataSize_ = MatmulPerformance::MM_MIN_DATASIZE_OTHER_SOC;

    void SetCyclePerMicroSec(SocVersion inputSocVersion)
    {
        mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC;
        if (inputSocVersion == SocVersion::SOC310_P) {
            mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC_VERSION310_P;
        } else if (inputSocVersion == SocVersion::SOC950) {
            mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC_NPUARCH_3510;
            mmMinDataSize_ = MatmulPerformance::MM_MIN_DATASIZE_NPUARCH_3510;
        } else {
            mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC;
        }
    }
    void SetCalcType(const TilingArgs &args)
    {
        if ((mmShapeInfo_.socType == SocVersion::SOC950) &&
            ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
             (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
             (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2))) {
            calcType_ = MatmulCalcType::QUANT;
        } else if ((args.aType == matmul_tiling::DataType::DT_INT8) &&
                   (mmShapeInfo_.socType != SocVersion::SOC910_B)) { // A8W8
            calcType_ = MatmulCalcType::QUANT;
        } else if ((mmShapeInfo_.socType == SocVersion::SOC310_P) &&
                   (args.bType == matmul_tiling::DataType::DT_INT8)) { // A16W8
            mmShapeInfo_.inMatrixBDtypeSize = MatmulPerformance::INT8_DTYPE_SIZE;
            calcType_ = MatmulCalcType::ANTI_QUANT;
        }
    }
    // Constructor
    MatmulPerformanceModel(const TilingArgs &args, const MatmulPerfTable &perfTable,
                           SocVersion inputSocVersion = SocVersion::SOC910_B)
        : calcType_(MatmulCalcType::FP16),
          perfTable_(perfTable)
    {
        mmShapeInfo_.socType = inputSocVersion;
        mmShapeInfo_.coreNum = args.aicCoreNum; // 每die核数
        mmShapeInfo_.inMatrixADtypeSize = args.inputDtypeSize;
        mmShapeInfo_.inMatrixBDtypeSize = mmShapeInfo_.inMatrixADtypeSize;
        mmShapeInfo_.outMatrixCDtypeSize = args.outputDtypeSize;
        mmShapeInfo_.mValue = args.mValue;
        mmShapeInfo_.nValue = args.nValue;
        mmShapeInfo_.kValue = args.kValue;
        // 获取后续计算用的基本块
        mmShapeInfo_.baseM = BASE_BLOCK_M;
        mmShapeInfo_.baseN = BASE_BLOCK_N;
        mmShapeInfo_.baseK = BASE_BLOCK_K;
        mmShapeInfo_.batchSize = 1UL; // 初始值
        SetCalcType(args);
        SetCyclePerMicroSec(inputSocVersion);
        GetMachineParameters();
    };

    std::string GetCalcTypeString()
    {
        return std::to_string(static_cast<int>(mmShapeInfo_.socType)) + "_" +
               std::to_string(static_cast<int>(calcType_)) + "_" + std::to_string(mmShapeInfo_.inMatrixADtypeSize) +
               "_" + std::to_string(mmShapeInfo_.inMatrixBDtypeSize) + "_" +
               std::to_string(mmShapeInfo_.outMatrixCDtypeSize);
    }
    // 根据输入shape，预测cube利用率（取值0 < cubeUtil_ < 1），再用利用率预测耗时
    // 耗时 = M * k * N / 频率 / 每cycle多少计算  /核数 / 利用率
    void GetMachineParameters();
    void CheckKvalueAlignVersion310P();
    void FindCubeUtil(uint64_t mSize, uint64_t rankTileNum, bool flagAllReduce, uint64_t *maxTileLen = nullptr);
    double FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t *maxTileLen = nullptr);
    double FindCubeUtilQuantVersion310P() const;
    double FindCubeUtilVersion310P() const;
    void ChangeCubeUtilByKAlign();
    void GetMatmulGradient();
    void SetBatchSize(uint64_t bSize)
    {
        mmShapeInfo_.batchSize = bSize;
    };
    uint32_t GetBaseM()
    {
        return mmShapeInfo_.baseM;
    };
    uint32_t GetBaseN()
    {
        return mmShapeInfo_.baseN;
    };
    // 返回允许切分的最小数据量
    uint64_t GetLinearThresholdLen(uint64_t rankTileNum);
    // 性能预测
    double MatmulTime(uint64_t tileSize, uint64_t rankTileNum);
    uint64_t InverseMatmulTime(double targetTime, uint64_t rankTileNum);
    virtual ~MatmulPerformanceModel() {}

protected:
    const MatmulPerfTable &perfTable_; // 模型参数表由构造方注入（算子侧标定头）
};

inline void MatmulPerformanceModel::GetMachineParameters()
{
    mmShapeInfo_.computesPerCycle = perfTable_.defaultComputesPerCycle;
    auto key = GetCalcTypeString();
    for (size_t i = 0; i < perfTable_.cubeCalcCount; ++i) {
        if (key == perfTable_.cubeCalcRows[i].key) {
            mmShapeInfo_.computesPerCycle = perfTable_.cubeCalcRows[i].computesPerCycle;
            return;
        }
    }
}

inline void MatmulPerformanceModel::CheckKvalueAlignVersion310P()
{
    if (mmShapeInfo_.kValue % MatmulPerformance::K_ALIGN_LEN != 0) {
        cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO_SOC310P;
    }
}

inline double MatmulPerformanceModel::FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t *maxTileLen)
{
    L2CacheEstimateParameters l2EstimatePar = perfTable_.defaultL2Params;
    auto key = GetCalcTypeString();
    for (size_t i = 0; i < perfTable_.l2Count; ++i) {
        if (key == perfTable_.l2Rows[i].key) {
            l2EstimatePar = perfTable_.l2Rows[i].params;
            break;
        }
    }
    uint64_t tmpBaseM = std::min(mmShapeInfo_.baseM, mSize);
    uint64_t tmpBaseN = mmShapeInfo_.baseN;
    uint64_t mBlockNum = (mSize + tmpBaseM - 1) / tmpBaseM * rankTileNum;
    uint64_t nBlockNum = (mmShapeInfo_.nValue + tmpBaseN - 1) / tmpBaseN;
    uint64_t memUsage =
        std::min(mBlockNum, mmShapeInfo_.coreNum) * tmpBaseM + std::min(nBlockNum, mmShapeInfo_.coreNum) * tmpBaseN;
    memUsage = memUsage * mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize / ONE_MBYTE;
    double tmpUtil;
    double diff = 0;
    if (memUsage <= l2EstimatePar.sizePartL2) {
        tmpUtil = l2EstimatePar.utilPartL2; // 0.85x cube utilization rate
        diff = l2EstimatePar.sizePartL2 - memUsage;
    } else if (memUsage <= l2EstimatePar.sizeFullL2) {
        tmpUtil = l2EstimatePar.utilFullL2; // 0.75x cube utilization rate
        diff = l2EstimatePar.sizeFullL2 - memUsage;
    } else {
        tmpUtil = l2EstimatePar.utilOutL2; // 0.65x cube utilization rate
    }
    tmpUtil *= static_cast<double>(std::min(mBlockNum * nBlockNum, mmShapeInfo_.coreNum) /
                                   static_cast<double>(mmShapeInfo_.coreNum));

    /*950
    cube利用率计算公式：两个输入为M、N的调和平均数和K。MN调和平均数的阈值为820，K阈值为1280，小于这个阈值
    认为提供负增益，大于这个阈值认为提供正增益，但是正增益最高不超过1.4倍。乘方函数用来缓和增长率，减慢变化速度*/
    if (mmShapeInfo_.socType == SocVersion::SOC950) {
        double mnharmonicMean = static_cast<double>(mSize * rankTileNum * mmShapeInfo_.nValue) /
                                static_cast<double>(mSize * rankTileNum + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::KVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerformance::FRONT_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::MNVALUE_THRESHOLD, MatmulPerformance::FRONT_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerformance::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        tmpUtil = std::min(MatmulPerformance::MAX_CUBE_UTIL, result);
    }

    if (maxTileLen != nullptr) {
        *maxTileLen = mmShapeInfo_.mValue;
        uint64_t tmpSize =
            std::min(mBlockNum + rankTileNum, mmShapeInfo_.coreNum) - std::min(mBlockNum, mmShapeInfo_.coreNum);
        tmpSize = tmpSize * mmShapeInfo_.inMatrixADtypeSize * tmpBaseM * mmShapeInfo_.kValue / ONE_MBYTE;
        if (tmpSize > diff && memUsage <= l2EstimatePar.sizeFullL2 && nBlockNum > mmShapeInfo_.coreNum) {
            *maxTileLen = (mSize + mmShapeInfo_.baseM - 1) / mmShapeInfo_.baseM * mmShapeInfo_.baseM;
        }
    }
    return tmpUtil;
}

inline double MatmulPerformanceModel::FindCubeUtilQuantVersion310P() const
{
    return 1.0 / (MatmulPerformance::UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P +
                  MatmulPerformance::UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P * static_cast<double>(ONE_KBYTE) /
                      static_cast<double>(mmShapeInfo_.kValue));
}

inline double MatmulPerformanceModel::FindCubeUtilVersion310P() const
{
    return 1.0 / (MatmulPerformance::UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P_NO_QUANT +
                  MatmulPerformance::UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P_NO_QUANT * static_cast<double>(ONE_KBYTE) /
                      static_cast<double>(mmShapeInfo_.kValue));
}

inline void MatmulPerformanceModel::ChangeCubeUtilByKAlign()
{
    uint64_t kLength = mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize;
    if (mmShapeInfo_.socType == SocVersion::SOC910_B4) {
        if (kLength % MatmulPerformance::HALF_CACHE_LINE_LEN != 0) {
            cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO_910_B4;
        }
        return;
    }

    if (kLength % MatmulPerformance::HALF_CACHE_LINE_LEN != 0) {
        cubeUtil_ *= MatmulPerformance::HALF_K_UNALIGN_UTIL_RATIO;
    } else if (kLength % MatmulPerformance::CACHE_LINE_LEN != 0) {
        cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO;
    }
}

inline void MatmulPerformanceModel::FindCubeUtil(uint64_t mSize, uint64_t rankTileNum, bool flagAllReduce,
                                                 uint64_t *maxTileLen)
{
    if (mmShapeInfo_.socType == SocVersion::SOC310_P) {
        if (calcType_ == MatmulCalcType::QUANT) {
            cubeUtil_ = FindCubeUtilQuantVersion310P();
        } else {
            cubeUtil_ = FindCubeUtilVersion310P();
        }
        CheckKvalueAlignVersion310P();
        return;
    }

    cubeUtil_ = FindCubeUtilByL2Usage(mSize, rankTileNum, maxTileLen);

    if (mmShapeInfo_.socType == SocVersion::SOC950) {
        double mnharmonicMean = static_cast<double>(mmShapeInfo_.mValue * rankTileNum * mmShapeInfo_.nValue) /
                                static_cast<double>(mmShapeInfo_.mValue * rankTileNum + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::KVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::MNVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerformance::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        cubeUtil_ = std::min(MatmulPerformance::MAX_CUBE_UTIL, result);
    }

    if (flagAllReduce) {
        if (calcType_ == MatmulCalcType::QUANT) { // 全量化
            return ChangeCubeUtilByKAlign();
        }
        if (mmShapeInfo_.kValue % MatmulPerformance::K_ALIGN_LEN != 0) { // 非量化和伪量化
            cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO;
        }
    }
}

inline uint64_t MatmulPerformanceModel::GetLinearThresholdLen(uint64_t rankTileNum)
{
    uint64_t tmpN = std::max(mmShapeInfo_.baseM, mmShapeInfo_.nValue);
    uint64_t tmpK = std::max(mmShapeInfo_.baseM, mmShapeInfo_.kValue);
    bool bmmFlag = (mmShapeInfo_.batchSize > 1 ? true : false);
    if (bmmFlag) {
        uint64_t mmMinDataSize = MatmulPerformance::BMM_DATASIZE_LARGE_K;
        if (mmShapeInfo_.kValue < MatmulPerformance::BMM_DATASIZE_K_BAR) {
            mmMinDataSize = MatmulPerformance::BMM_DATASIZE_SMALL_K;
        }
        return std::max(mmMinDataSize / (tmpN * tmpK * rankTileNum), mmShapeInfo_.baseM);
    }

    bool discreteFlag = rankTileNum > MatmulPerformance::SMALL_RANKTILE;
    if (discreteFlag) {
        return std::max(mmMinDataSize_ / (tmpN * tmpK * rankTileNum), mmShapeInfo_.baseM);
    }

    uint64_t thresholdSize = 0;
    MinMatmulShapeParameters minShapePar;
    bool adaptCoreNumFlag = (mmShapeInfo_.socType != SocVersion::SOC310_P);
    if (adaptCoreNumFlag) {
        minShapePar.mmMinDataSize1 =
            minShapePar.mmMinDataSize1 / MatmulPerformance::MARK_CORE_NUM_SOC910B * mmShapeInfo_.coreNum;
        minShapePar.mmMinDataSize2 =
            minShapePar.mmMinDataSize2 / MatmulPerformance::MARK_CORE_NUM_SOC910B * mmShapeInfo_.coreNum;
        minShapePar.minMSize = MatmulPerformance::MIN_M_SIZE_FACTOR * mmShapeInfo_.baseM;
    }
    thresholdSize = std::max(minShapePar.mmMinDataSize1 / (tmpN * tmpK),
                             minShapePar.mmMinDataSize2 / (tmpN * tmpK / ONE_KBYTE + tmpN));
    if (rankTileNum > 1) {
        thresholdSize /= rankTileNum;
    }
    thresholdSize = std::max(minShapePar.minMSize, thresholdSize);

    return thresholdSize;
}

inline void MatmulPerformanceModel::GetMatmulGradient()
{
    uint64_t tmpN = std::max(mmShapeInfo_.baseN, mmShapeInfo_.nValue);
    uint64_t tmpK = std::max(mmShapeInfo_.baseK, mmShapeInfo_.kValue);
    matmulGradient_ = (tmpN * tmpK) / (mmShapeInfo_.computesPerCycle * mmShapeInfo_.cyclePerMicroSec *
                                       mmShapeInfo_.coreNum * cubeUtil_);
}

inline double MatmulPerformanceModel::MatmulTime(uint64_t tileSize, uint64_t rankTileNum)
{
    tileSize = std::max(mmShapeInfo_.baseM, tileSize);
    return static_cast<double>(tileSize) * static_cast<double>(rankTileNum) * matmulGradient_;
}

inline uint64_t MatmulPerformanceModel::InverseMatmulTime(double targetTime, uint64_t rankTileNum)
{
    uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rankTileNum) * matmulGradient_));
    tileSize = std::max(mmShapeInfo_.baseM, tileSize);
    return tileSize;
}

// ====== arch35 模型（来源: matmul_performance_arch35.h） ======
class MatmulPerformanceArch35 : public MatmulPerformanceModel {
public:
    MatmulPerformanceArch35(const TilingArgs &args, const MatmulPerfTable &perfTable,
                            SocVersion inputSocVersion = SocVersion::SOC950)
        : MatmulPerformanceModel(args, perfTable, inputSocVersion)
    {
        TilingBestBaseBlock bestBaseBlock = GetBestBaseBlock(inputSocVersion);
        mmShapeInfo_.baseM = bestBaseBlock.baseM;
        mmShapeInfo_.baseN = bestBaseBlock.baseN;
        mmShapeInfo_.baseK = bestBaseBlock.baseK;
    }

    TilingBestBaseBlock GetBestBaseBlock(SocVersion socVersionIn)
    {
        for (size_t i = 0; i < perfTable_.bestBaseCount; ++i) {
            if (perfTable_.bestBaseRows[i].soc == socVersionIn) {
                return perfTable_.bestBaseRows[i].block;
            }
        }
        return perfTable_.defaultBaseBlock;
    }

    // 单参版本遮蔽基类 FindCubeUtil（与旧实现一致, arch35 对象只应使用本版本）
    void FindCubeUtil(uint64_t rankTileNum)
    {
        double mnharmonicMean = static_cast<double>(mmShapeInfo_.mValue * rankTileNum * mmShapeInfo_.nValue) /
                                static_cast<double>(mmShapeInfo_.mValue * rankTileNum + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::KVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::MNVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerformance::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        cubeUtil_ = std::min(MatmulPerformance::MAX_CUBE_UTIL, result);
    }
};

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_MATMUL_PERF_MODEL_H__
