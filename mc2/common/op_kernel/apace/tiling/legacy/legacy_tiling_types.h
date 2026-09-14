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
 * \file legacy_tiling_types.h
 * \brief 旧公式化/拟合配平 tiling 家族下沉 apace 的公共数据类型与共享常量
 *
 * 提取来源（逐字段忠实拷贝, 仅命名空间包装, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/formulaic_tiling_datatype.h   (全部类型/常量)
 *   - mc2/common/op_host/op_tiling/matmul_formulaic_tiling.h     (TilingArgs/AicpuComType/BASE_BLOCK_*,
 *     供本家族引擎构造函数消费; MatmulFormulaicTiling 类本体未随 P0 下沉, 见
 *     matmul_perf_model.h 头注)
 *
 * 与旧头的关系: 旧文件原样保留(arch22 及 mc2_tiling_utils.h 等间接依赖仍在用);
 * 本文件置于 apace::legacy 命名空间, 同名类型与旧全局类型可在一个编译单元共存,
 * 算子迁移期间由算子侧胶水做字段级转换。
 */

#ifndef __APACE_LEGACY_TILING_TYPES_H__
#define __APACE_LEGACY_TILING_TYPES_H__

#pragma once

#include <cstdint>
#include "graph/ge_error_codes.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace apace {
namespace legacy {

// ====== 以下常量来源: formulaic_tiling_datatype.h ======
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

enum class SocVersion {
    SOC910_B,
    SOC310_P,
    SOC910_93,
    SOC910_B4,
    SOC950,
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
    SocVersion socType;
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

struct TilingBestBaseBlock {
    uint32_t baseM = 256; // 256: init value for baseM
    uint32_t baseN = 256; // 256: init value for baseN
    uint32_t baseK = 128; // 128: init value for baseK
};

// ====== 通信轮次硬上限（来源: hccl_formulaic_tiling.h 的 MAX_TILE_CNT） ======
// 受 FFTS 队列大小限制, 供 fit-based balance 与 one-calc-one-comm 两个引擎共用。
constexpr uint64_t MAX_TILE_CNT = 16;

// ====== 以下来源: matmul_formulaic_tiling.h（TilingArgs 及其依赖） ======
constexpr uint32_t BASE_BLOCK_M = 128;
constexpr uint32_t BASE_BLOCK_N = 256;
constexpr uint32_t BASE_BLOCK_K = 64;

enum class AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_HALFALLTOALLV = 20,
    HCCL_CMD_MAX
};

struct TilingArgs {
    AicpuComType cmdType;
    uint32_t rankDim;
    uint32_t usedCoreNum;
    uint64_t orgMValue;
    uint64_t orgNValue;
    uint64_t orgKValue;
    uint64_t batchValue;
    uint64_t mValue;
    uint64_t kValue;
    uint64_t nValue;
    uint64_t baseMLimit; // 归一方案里，用于多卡多片数据一把传给mmv3做tiling时，表示单卡单片数据的大小
    uint64_t inputDtypeSize;
    uint64_t outputDtypeSize;
    uint64_t aicCoreNum;
    uint64_t commTurn;
    uint64_t rankTileNum;
    uint8_t commAlg;
    bool isATrans;
    bool isBTrans;
    bool enablePad;
    bool enableSplitK;
    bool isBias;
    bool isStorageGather;
    bool isLocal = false;
    ge::DataType geAType;
    ge::DataType geBType;
    ge::DataType geCType;
    ge::DataType geBiasType;
    ge::DataType antiquantscaleDType;
    matmul_tiling::DataType aType;
    matmul_tiling::DataType bType;
    matmul_tiling::DataType cType;
    matmul_tiling::DataType biasType;
};

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_TILING_TYPES_H__
