/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file formulaic_tiling_datatype.h
 * \brief
 */
#ifndef __FORMULAIC_TILING_DATATYPE_H__
#define __FORMULAIC_TILING_DATATYPE_H__

#pragma once
#include <cstdint>

constexpr uint64_t ONE = 1;
constexpr uint64_t TWO = 2;
constexpr uint64_t THREE = 3;
constexpr uint64_t FOUR = 4;
constexpr uint64_t ONE_KBYTE = 1024;
constexpr uint64_t FOUR_KILO = 4096;
constexpr uint64_t SMALL_M = 2048;
constexpr uint64_t ONE_MBYTE = ONE_KBYTE * ONE_KBYTE;
constexpr uint64_t ONE_GBYTE = ONE_KBYTE * ONE_MBYTE;
constexpr uint64_t TEN_GBYTE = 10 * ONE_GBYTE;
constexpr uint64_t FOUR_MBYTE = 4 * ONE_MBYTE;
constexpr double L2_SIZE_PART = 128;  // soc 910B L2 comfort size
constexpr double L2_SIZE_FULL = 192;  // soc 910B L2 full size
constexpr double PART_L2_UTIL = 0.85; // cube utilization
constexpr double FULL_L2_UTIL = 0.75; // cube utilization
constexpr double NO_L2_UTIL = 0.65;   // cube utilization
constexpr uint64_t MIN_DATA_PAR1 = 4 * ONE_GBYTE;
constexpr uint64_t MIN_DATA_PAR2 = 6 * ONE_MBYTE;
constexpr uint64_t MIN_M_SIZE_SOC310P = 512;
constexpr uint64_t MIN_COMM_RANKDIM = 2;

// arch在拟合查表键中的序列号(原SOC950/SOC310_P枚举序值)。
// 枚举不再持有950/310P档位后,由arch(DAV_3510/DAV_2002)经下列常量映射回原序列号,
// 保持拟合表键兼容(如"4_0"、"1_3"、"1_4")。
constexpr int32_t NPUARCH_3510_FITTING_CODE = 4;
constexpr int32_t NPUARCH_2002_FITTING_CODE = 1;

// DAV_2201系(A2/A3)拟合档位枚举:310P(1:1对应DAV_2002)与950系(1:1对应DAV_3510)
// 均已迁出,剩余档位均属DAV_2201,故名。序值显式钉住历史值——拟合表键("2_1"、"3_..."等)
// 内嵌枚举序值,漂移会破坏查表。
enum class SocVersion2201 {
    SOC910_B = 0,
    SOC910_93 = 2, // 显式2:历史序值(原SOC310_P占1),拟合表键"2_..."依赖
    SOC910_B4 = 3, // 显式3:拟合表键"3_..."依赖
};

enum class KernelType {
    ALL_REDUCE, // default
    ALL_GATHER,
    REDUCE_SCATTER,
    ALL_TO_ALL,
    REDUCE_SCATTER_VIA_ALL_TO_ALL,
    ALL_REDUCE_VIA_TWO_SHOT,
};

enum class MatmulCalcType {
    FP16, // default
    QUANT,
    ANTI_QUANT,
};

struct MatmulParameters {
    SocVersion2201 socType;
    uint64_t coreNum;
    uint64_t inMatrixADtypeSize;
    uint64_t inMatrixBDtypeSize;
    uint64_t outMatrixCDtypeSize;
    uint64_t batchSize; // E / Ep
    uint64_t mValue;
    uint64_t nValue;
    uint64_t kValue;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    double computesPerCycle; // cube算力
    double cyclePerMicroSec; // frequency
};

struct MinMatmulShapeParameters {
    MinMatmulShapeParameters()
        : mmMinDataSize1(MIN_DATA_PAR1),
          mmMinDataSize2(MIN_DATA_PAR2),
          minMSize(MIN_M_SIZE_SOC310P) {};
    uint64_t mmMinDataSize1;
    uint64_t mmMinDataSize2;
    uint64_t minMSize;
};

struct L2CacheEstimateParameters {
    uint64_t sizePartL2;
    uint64_t sizeFullL2;
    double utilPartL2;
    double utilFullL2;
    double utilOutL2;
};

enum class HCCLType {
    FULL_MESH,
    DOUBLE_RING,
    SWITCH,
    RING_RANK2_310P,
    RING_RANK4_310P,
};

enum class TopoType {
    STANDARD_CARD, // 0 - 标卡 4P
    EIGHT_P,       // 1 - 8P场景（POD8P/Server8P）
};

struct HCCLInfo {
    uint64_t rankDim;
    uint64_t commDtypeSize; // data type size in hccl
    uint64_t commMatrixLen; // equal to kValue in gather, equal to nValue in reduce
    KernelType kernelType;
    HCCLType commMethod;
    TopoType topoType;
    uint64_t maxStepSize; // number of steps in a full round of communication
    uint64_t commDtypeSizeExpansionFraction;
};

struct HCCLFittingParameters {
    double sizeToTimeBoundary1;
    double sizeToTimeBoundary2;
    double sizeToTimeLinearGradient;
    double sizeToTimeLinearOffset;
    double sizeToTimeParabolicPar1; // a
    double sizeToTimeParabolicPar2; // b
    double sizeToTimeParabolicPar3; // c
    double timeToSizeBoundary1;
    double timeToSizeBoundary2;
    // y = ax*x + bx + c -->
    // x = - sqrt((y - c + b*b / 4a) / a) - b / 2a
    // x = - sqrt(a' * (y + b')) + c'
    double timeToSizeParabolicPar1; // a'
    double timeToSizeParabolicPar2; // b'
    double timeToSizeParabolicPar3; // c'
};

struct CutResult {
    bool shortTileAtBack;
    uint64_t totalTileCnt;
    uint64_t numLongTile;
    uint64_t numShortTile;
    uint64_t longTileLen;
    uint64_t shortTileLen;
};

struct TileArguments {
    uint64_t mAlignLen;
    uint64_t maxTileLen;
    uint64_t minTileLen; // threshold size, max(commMinLen, matmulMinLen)
    uint64_t maxTileCnt;
};
#endif // __FORMULAIC_TILING_DATATYPE_H__
