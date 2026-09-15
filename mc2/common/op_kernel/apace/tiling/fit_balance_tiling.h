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
 * \file fit_balance_tiling.h
 * \brief fit-balance tiling 家族唯一公共文件（统一两代实现, 见迁移方案 §2.2/§4.9）
 *
 * 家族结构与分支选择:
 *   - 第一部分【公共主干·自由长度分支】通信引擎 = HCCL AICPU/CCU（每轮固定开销大,
 *     反解最优轮长压轮数, mm 侧可重切）。消费者: 旧通算融合算子的 fit-balance 路径
 *     （agmm_v2 等, 经算子侧标定头 + ctx/钩子注入调用）。
 *   - 第二部分【URMA 拓展分支·wave-packing】通信引擎 = URMA（每轮开销 ~4.5us 近乎免费,
 *     mm 恒最优, 通信按计算波整数倍打包, estTime 竞争决策）。当算子通信模式为
 *     aiv_urma 时走本分支——当前唯一消费方 allto_all_matmul_v2 即 aiv_urma 实现
 *     （其 kernel 仅 URMA、tiling 无条件走本分支）; 多通信模式算子接入时由算子侧
 *     判定 comm_mode 后选分支。
 *   - 两分支互不复用（模型/求解器/输出契约均不同）, 仅共居本文件; 对齐 P5 统一规划
 *     （迁移方案 §2.2 三层结构的单文件形态）。
 *
 * 提取来源与等价性: 见各部分头部注释（第一部分 2026-09-14 P0 提取经 162 用例
 * bit 级验证; 第二部分为 v2 已验证代码原样迁入, R1~R6 五轮性能回归基线）。
 */
#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

// ge::DataType 经 register/tilingdata_base.h 与 tiling/tiling_api.h 的既有链路提供
// (不直接 include graph 头: 部分 TU 的 ge 链路来自不同 toolkit 路径, 直接引入会双重定义)
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_data_struct_define.h"

namespace apace {

// ============================================================================
// 第一部分【公共主干 · 自由长度分支】通信引擎: HCCL AICPU/CCU
// （原 fit_based_balance_tiling.h 全量迁入; 提取来源与差异说明见本部分内注释）
// ============================================================================

namespace fit_based_balance {

// ============================================================================
// 一、公共类型与常量（来源: formulaic_tiling_datatype.h + matmul_formulaic_tiling.h 公共部分）
// ============================================================================

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

// 通信轮次硬上限（来源: hccl_formulaic_tiling.h 的 MAX_TILE_CNT, 受 FFTS 队列大小限制）
constexpr uint64_t MAX_TILE_CNT = 16;

// 旧基类头文件里的通算比阈值（算子钩子使用）
constexpr double CALC_COMM_RATIO = 2.0;

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

// 调度入参上下文（与旧 mc2tiling::TilingArgs 字段逐一对应, 算子侧边界转换）
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
    uint64_t baseMLimit;
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

// ============================================================================
// 二、通信性能模型（来源: hccl_performance.h/.cpp + hccl_performance_arch35.h/.cpp）
// ============================================================================

constexpr uint64_t HCCL_MIN_TILE_LEN = 64 * ONE_KBYTE;
constexpr uint64_t HCCL_MIN_TILE_LEN_COARSE = 2 * ONE_MBYTE;
constexpr const char *DEFAULT_KEY_FOR_FITTING_MAP = "0_0";
constexpr double FULL_MESH_TIME_FACTOR = 2.0;
constexpr uint64_t DOUBLE_RING_RANKTILE = 2;
constexpr uint64_t RING_TOTAL_STEP_PAR1 = 1;
constexpr uint64_t RING_TOTAL_STEP_PAR2 = 2;
constexpr uint64_t RANK_FOUR = 4;
constexpr uint64_t FITTING_RANK = 8;
constexpr double LOCAL_REDUCE_FACTOR = 0.4; // 确定性Local ReduceScatter额外耗时参数

// ====== 拟合表注入 schema（行数据由算子侧标定头维护） ======
struct CommFitRow {
    const char *key; // 查表 key, 由模型侧按 key 规则生成（见 GetCommMethodString / GetCommInfoString）
    HCCLFittingParameters params;
};

struct CommFitTable {
    const CommFitRow *rows;
    size_t rowCount;
    HCCLFittingParameters
        defaultParams; // key 未命中时的回退行（旧 FITTING_PARAMETER_MAP/ARCH35_MAP 的 DEFAULT_KEY 行内容）
};

// 精确 key 匹配, 未命中返回 nullptr（语义与旧 map::find == end 一致）
inline const HCCLFittingParameters *FindCommFitParams(const CommFitTable &table, const std::string &key)
{
    for (size_t i = 0; i < table.rowCount; ++i) {
        if (key == table.rows[i].key) {
            return &table.rows[i].params;
        }
    }
    return nullptr;
}

// 组装纯线性拟合行（算子侧标定头用: 两段边界置 0, 任意 x>0 走线性分支 y = k·x + b）
inline HCCLFittingParameters MakeLinearCommFit(double slopeUsPerMb, double interceptUs)
{
    return {0, 0, slopeUsPerMb, interceptUs, 0, 0, 0, 0, 0, 0, 0, 0};
}

class HCCLPerformanceModel {
public:
    HCCLFittingParameters commEstimatePar_;
    HCCLInfo commTypeInfo_;
    double commTimeFactor_ = 1.0; // allreduce time is 2.0x allgather or reducescatter
    uint64_t lookUpTileNum_ = 1;  // 数据拟合查表参数
    std::string keyToFittingMap_ = DEFAULT_KEY_FOR_FITTING_MAP;
    SocVersion socVersion_ = SocVersion::SOC910_B;

    void SetCommParametersBaseSocType(SocVersion inputSocVersion)
    {
        if (inputSocVersion == SocVersion::SOC310_P) {
            SetCommMethodVersion310P(); // 310P只支持MatmulAllreduce算子
            socVersion_ = SocVersion::SOC310_P;
        } else if (inputSocVersion == SocVersion::SOC910_93) {
            InitSOC91093();
            socVersion_ = SocVersion::SOC910_93;
        } else if (inputSocVersion == SocVersion::SOC950) {
            InitParametersForFullMesh();
            socVersion_ = SocVersion::SOC950;
        } else {
            InitParametersForFullMesh();
        }
    }
    // Constructor
    HCCLPerformanceModel(uint32_t inputRankDim, KernelType inputKernelType, const CommFitTable &fitTable,
                         SocVersion inputSocVersion = SocVersion::SOC910_B)
        : fitTable_(fitTable)
    {
        commTypeInfo_.kernelType = inputKernelType; // 区分哪个MC2算子
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_COMM_RANKDIM); // 并行维度最小为2
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        SetCommParametersBaseSocType(inputSocVersion);
        SetMaxStepSize();
        keyToFittingMap_ = GetCommMethodString(inputSocVersion);
        GetCommEstimateParameters();
    };
    std::string GetCommMethodString(SocVersion socType)
    {
        return std::to_string(static_cast<int>(socType)) + "_" +
               std::to_string(static_cast<int>(commTypeInfo_.commMethod));
    }

    void SetCommMethodVersion310P();
    void SetMaxStepSize();
    void SetCommShapeLen(uint64_t len)
    {
        commTypeInfo_.commMatrixLen = len;
    }
    void SetCommDTypeSize(uint64_t dTypeSize)
    {
        commTypeInfo_.commDtypeSize = dTypeSize;
    }
    inline void SetCommDtypeSizeExpansionFraction(uint64_t number)
    {
        commTypeInfo_.commDtypeSizeExpansionFraction = number;
    }
    double GetRealDtypeSizes() const
    {
        return static_cast<double>(commTypeInfo_.commDtypeSize) / commTypeInfo_.commDtypeSizeExpansionFraction;
    }
    uint64_t GetCommDTypeSize()
    {
        return commTypeInfo_.commDtypeSize;
    }
    void SetCommTimeFactor(double factor)
    {
        commTimeFactor_ = factor;
    }
    void ChangeCommTimeFactorByDivision(double factor);
    void SetFullMeshCommTimeFactor();
    void SetRingCommTimeFactor();
    void InitParametersForFullMesh();
    void InitSOC91093();
    void SetLocalReduceFactor();

    uint64_t GetFullMeshRankTileNum();
    uint64_t GetRankTileNum();
    void GetCommEstimateParameters();
    uint64_t GetMaxStepSize();
    uint64_t GetLinearThresholdLen();
    uint64_t GetLinearThresholdLenCoarse();

    // 性能拟合函数
    double CommTime(uint64_t mSize) const;
    virtual uint64_t InverseCommTime(double targetTime) const;
    virtual ~HCCLPerformanceModel() {}

protected:
    const CommFitTable &fitTable_; // 拟合表由构造方注入（算子侧标定头）
};

inline void HCCLPerformanceModel::SetCommMethodVersion310P()
{
    if (commTypeInfo_.rankDim == RANK_FOUR) {
        commTypeInfo_.commMethod = HCCLType::RING_RANK4_310P;
        return;
    }
    commTypeInfo_.commMethod = HCCLType::RING_RANK2_310P;
}

inline void HCCLPerformanceModel::ChangeCommTimeFactorByDivision(double factor)
{
    if (factor > 0) {
        commTimeFactor_ /= factor;
    }
}

inline void HCCLPerformanceModel::SetFullMeshCommTimeFactor()
{
    // 通信查表使用总数据量，实际时间 = 总数据量 / rankDim / 带宽
    // 拟合采用8 die通信
    commTimeFactor_ = FULL_MESH_TIME_FACTOR;
    ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(commTypeInfo_.rankDim));
}

inline void HCCLPerformanceModel::SetRingCommTimeFactor()
{
    // 通信查表使用总数据量，实际时间 = 总数据量 * 2 * (rankDim - 1) / rankDim /
    // 带宽 拟合采用8 die通信
    commTimeFactor_ = FULL_MESH_TIME_FACTOR;
    double fittingRatio = (static_cast<double>(FITTING_RANK) - 1.0) / static_cast<double>(FITTING_RANK);
    double currentRatio =
        (static_cast<double>(commTypeInfo_.rankDim) - 1.0) / static_cast<double>(commTypeInfo_.rankDim);
    ChangeCommTimeFactorByDivision(currentRatio / fittingRatio);
}

inline void HCCLPerformanceModel::InitParametersForFullMesh()
{
    commTypeInfo_.commMethod = HCCLType::FULL_MESH;
    if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
        (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER)) {
        SetFullMeshCommTimeFactor();
        lookUpTileNum_ = commTypeInfo_.rankDim;
    }
}

inline void HCCLPerformanceModel::InitSOC91093()
{
    commTypeInfo_.commMethod = HCCLType::DOUBLE_RING;
    SetRingCommTimeFactor();
    lookUpTileNum_ = commTypeInfo_.rankDim;
}

inline void HCCLPerformanceModel::SetLocalReduceFactor()
{
    // 拟合采用8 die通信
    // die的数量越多，local累加的步骤就越多
    double changeFactor =
        1.0 + LOCAL_REDUCE_FACTOR * static_cast<double>(commTypeInfo_.rankDim) / static_cast<double>(FITTING_RANK);
    ChangeCommTimeFactorByDivision(changeFactor);
}

inline uint64_t HCCLPerformanceModel::GetFullMeshRankTileNum()
{
    if (commTypeInfo_.kernelType == KernelType::ALL_GATHER) {
        return commTypeInfo_.rankDim - 1;
    }
    return commTypeInfo_.rankDim;
}

inline uint64_t HCCLPerformanceModel::GetRankTileNum()
{
    if (commTypeInfo_.kernelType != KernelType::ALL_REDUCE) {
        return GetFullMeshRankTileNum();
    }
    return 1;
}

inline void HCCLPerformanceModel::SetMaxStepSize()
{
    if (commTypeInfo_.commMethod == HCCLType::DOUBLE_RING) {
        if (commTypeInfo_.rankDim == MIN_COMM_RANKDIM) { // rankDim = 2 uses full-mesh
            commTypeInfo_.maxStepSize = 1UL;
            return;
        }
        if (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER) {
            commTypeInfo_.maxStepSize = commTypeInfo_.rankDim;
            return;
        }
        if (commTypeInfo_.kernelType == KernelType::ALL_GATHER) {
            commTypeInfo_.maxStepSize = commTypeInfo_.rankDim - 1;
            return;
        }
    }
    commTypeInfo_.maxStepSize = 1;
}

inline uint64_t HCCLPerformanceModel::GetMaxStepSize()
{
    return commTypeInfo_.maxStepSize;
}

inline uint64_t HCCLPerformanceModel::GetLinearThresholdLen()
{
    uint64_t resultLen =
        static_cast<uint64_t>(HCCL_MIN_TILE_LEN / (commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes()));
    return resultLen;
}

inline uint64_t HCCLPerformanceModel::GetLinearThresholdLenCoarse()
{
    uint64_t resultLen = static_cast<uint64_t>(
        HCCL_MIN_TILE_LEN_COARSE / (commTypeInfo_.commMatrixLen * commTypeInfo_.rankDim * GetRealDtypeSizes()));
    return resultLen;
}

inline void HCCLPerformanceModel::GetCommEstimateParameters()
{
    commEstimatePar_ = fitTable_.defaultParams;
    const HCCLFittingParameters *hit = FindCommFitParams(fitTable_, keyToFittingMap_);
    if (hit != nullptr) {
        commEstimatePar_ = *hit;
    }
}

inline double HCCLPerformanceModel::CommTime(uint64_t mSize) const
{
    double commDataSize = mSize * commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes();
    double tmpSize = commDataSize / ONE_MBYTE;
    double result = commEstimatePar_.timeToSizeBoundary1;
    if (tmpSize > commEstimatePar_.sizeToTimeBoundary2) {
        result = commEstimatePar_.sizeToTimeLinearGradient * tmpSize + commEstimatePar_.sizeToTimeLinearOffset;
    } else if (tmpSize > commEstimatePar_.sizeToTimeBoundary1) {
        result = commEstimatePar_.sizeToTimeParabolicPar1 * tmpSize * tmpSize +
                 commEstimatePar_.sizeToTimeParabolicPar2 * tmpSize + commEstimatePar_.sizeToTimeParabolicPar3;
    }

    result /= commTimeFactor_;
    return result;
}

inline uint64_t HCCLPerformanceModel::InverseCommTime(double targetTime) const
{
    targetTime *= commTimeFactor_;
    double tmpSize = commEstimatePar_.sizeToTimeBoundary1;
    if (targetTime > commEstimatePar_.timeToSizeBoundary2) {
        tmpSize = (targetTime - commEstimatePar_.sizeToTimeLinearOffset) / commEstimatePar_.sizeToTimeLinearGradient;
    } else if (targetTime > commEstimatePar_.timeToSizeBoundary1) {
        if (socVersion_ == SocVersion::SOC950) {
            tmpSize =
                (targetTime - commEstimatePar_.timeToSizeParabolicPar3) / commEstimatePar_.timeToSizeParabolicPar2;
        } else {
            tmpSize = (-sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                             (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                       commEstimatePar_.timeToSizeParabolicPar3);
            if (commEstimatePar_.timeToSizeParabolicPar3 < 0) {
                tmpSize = sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                               (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                          commEstimatePar_.timeToSizeParabolicPar3;
            }
        }
    }
    return static_cast<uint64_t>(tmpSize * ONE_MBYTE /
                                 (commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes()));
}

// ====== arch35 模型（来源: hccl_performance_arch35.h/.cpp） ======
class HCCLPerformanceArch35 : public HCCLPerformanceModel {
public:
    HCCLPerformanceArch35(uint32_t inputRankDim, KernelType inputKernelType, const CommFitTable &arch35FitTable,
                          SocVersion socVersion = SocVersion::SOC950, TopoType topoType = TopoType::STANDARD_CARD)
        : HCCLPerformanceModel(inputRankDim, inputKernelType, arch35FitTable, socVersion)
    {
        InitArch35Parameters();
        commTypeInfo_.topoType = topoType;
        GetCommArch35Parameters(socVersion);
    }

    std::string GetCommInfoString(SocVersion socVersion);
    void GetCommArch35Parameters(SocVersion socVersion);
    void InitArch35Parameters();
    uint64_t InverseCommTime(double targetTime) const override;
};

inline std::string HCCLPerformanceArch35::GetCommInfoString(SocVersion socVersion)
{
    return std::to_string(static_cast<int>(socVersion)) + "_" +
           std::to_string(static_cast<int>(commTypeInfo_.kernelType)) + "_" +
           std::to_string(static_cast<int>(commTypeInfo_.topoType)) + "_" +
           std::to_string(static_cast<int>(commTypeInfo_.rankDim));
};

inline void HCCLPerformanceArch35::GetCommArch35Parameters(SocVersion socVersion)
{
    keyToFittingMap_ = GetCommInfoString(socVersion);
    const HCCLFittingParameters *hit = FindCommFitParams(fitTable_, keyToFittingMap_);
    if (hit != nullptr) {
        commEstimatePar_ = *hit;
    } else {
        commEstimatePar_ = fitTable_.defaultParams;
        ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(commTypeInfo_.rankDim));
    }
}

inline void HCCLPerformanceArch35::InitArch35Parameters()
{
    if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
        (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL)) {
        lookUpTileNum_ = commTypeInfo_.rankDim;
        commTimeFactor_ = 1;
    }
}

inline uint64_t HCCLPerformanceArch35::InverseCommTime(double targetTime) const
{
    double tmpSize = commEstimatePar_.sizeToTimeBoundary1;
    if (targetTime > commEstimatePar_.timeToSizeBoundary2) {
        tmpSize = (targetTime - commEstimatePar_.sizeToTimeLinearOffset) / commEstimatePar_.sizeToTimeLinearGradient;
    } else if (targetTime > commEstimatePar_.timeToSizeBoundary1) {
        if (commTypeInfo_.kernelType == KernelType::ALL_REDUCE && commEstimatePar_.timeToSizeParabolicPar2 != 0.0f) {
            tmpSize =
                (targetTime - commEstimatePar_.timeToSizeParabolicPar3) / commEstimatePar_.timeToSizeParabolicPar2;
        } else {
            tmpSize = (-sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                             (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                       commEstimatePar_.timeToSizeParabolicPar3);
            if (commEstimatePar_.timeToSizeParabolicPar3 < 0) {
                tmpSize = sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                               (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                          commEstimatePar_.timeToSizeParabolicPar3;
            }
        }
    }
    return static_cast<uint64_t>(tmpSize * ONE_MBYTE /
                                 (commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes()));
}

// ============================================================================
// 三、matmul 性能模型（来源: matmul_performance.h/.cpp + matmul_performance_arch35.h）
// ============================================================================

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

// ============================================================================
// 四、FormPartition 长短块离散/连续切分引擎（来源: hccl_formulaic_tiling.h/.cpp）
// ============================================================================

constexpr uint64_t LARGE_NK_BAR_BASE = 32;
constexpr uint64_t SMALL_TILECNT_PAR = 2;
constexpr uint64_t OPT_TILE_CNT = 8;
constexpr uint64_t OPT_TILE_LEN_RATIO = 2;
constexpr uint64_t OPT_TILE_LEN_RATIO_SOC310P = 1;
constexpr double BACKTILE_ALIGN_RATIO = 0.75;
constexpr uint64_t LARGE_BACKTILE_COMPARE_PAR = 2;
constexpr uint64_t SOC310_BASE_M = 256;
constexpr uint64_t SMALL_SHORTTILE_BAR = 512;
constexpr uint64_t SHORT_ALIGN_DOWN_DIFF = 16;
constexpr double ALIGN_RATIO_CALC_BOUND = 0.5;
constexpr double ALIGN_RATIO_COMM_BOUND = 0.75;
constexpr uint64_t SMALL_SHAPE_BAR = 2048;

// 切分和对齐
class FormPartition {
public:
    CutResult cutRes;                            // 切分结果
    TileArguments tileArgs;                      // 切分参数
    uint64_t totalLen = ONE;                     // 待切分的长度
    double backTileRatio = BACKTILE_ALIGN_RATIO; // 计算长块个数使用的参数

    // Constructor
    explicit FormPartition(const TilingArgs &args)
    {
        totalLen = args.mValue;
        tileArgs.mAlignLen = BASE_BLOCK_M; // 切分长度照顾mm性能，对齐baseM
        tileArgs.minTileLen = tileArgs.mAlignLen;
        tileArgs.maxTileCnt = ONE;
        tileArgs.maxTileLen = totalLen;

        cutRes.shortTileAtBack = false; // 计算Bound时短块后置，通信bound时短块前置
        NoCutTiling();                  // 初始化为不切分
    };

    void SetBackTileRatio(double newRatio)
    {
        backTileRatio = (newRatio > 0) ? newRatio : backTileRatio;
    };
    void SetMinLenByMax(uint64_t newLen);
    void SetMinLenByMin(uint64_t newLen);
    uint64_t GetMinLen()
    {
        return tileArgs.minTileLen;
    };
    void SetAlignLength(uint64_t newLen)
    {
        tileArgs.mAlignLen = std::max(newLen, ONE);
    };
    uint64_t GetAlignLength()
    {
        return tileArgs.mAlignLen;
    };
    void SetMaxTileCnt(uint64_t newSize)
    {
        tileArgs.maxTileCnt = std::max(newSize, ONE);
    };

    // TilingMethods
    void GenerateInitialPartition(bool smallDimAlignUp = false);
    bool SetShortTileLen(bool noCutFlag = false);
    void NoCutTiling();
    void MaxTileCntUniform();

    // 离散切分对齐 (AllGatherMatmul和MatmulReduceScatter算子)
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront = false);
    // 连续切分对齐 (MatmulAllReduce算子)
    void LargeBackTileCheck(bool soc310Flag, bool largeCalcCommRatio);
    void CalcShortTile();
    void CalcBackTile();
    void SmallFrontTileCheck(bool soc310Flag, bool kGreaterThanN);
    void RedistributeTile();
    void CommBoundShortAlign();
    void FitTileLengthContinuous(bool kGreaterThanN, bool largeCalcCommRatio = true, bool soc310Flag = false);
};

inline void FormPartition::SetMinLenByMax(uint64_t newLen)
{
    tileArgs.minTileLen = std::max(tileArgs.minTileLen, newLen);
}

inline void FormPartition::SetMinLenByMin(uint64_t newLen)
{
    tileArgs.minTileLen = std::min(tileArgs.minTileLen, newLen);
}

inline void FormPartition::AlignTileLarge()
{
    if (cutRes.numLongTile >= tileArgs.mAlignLen) {
        return;
    }
    // Align cutRes.shortTileLen
    cutRes.shortTileLen = (cutRes.shortTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + cutRes.numLongTile - 1) / cutRes.numLongTile;
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
}

inline void FormPartition::AlignTileSmall()
{
    // Front tile already aligned, check backTile
    bool smallBackTile = (cutRes.shortTileLen < tileArgs.mAlignLen) || (cutRes.shortTileLen < tileArgs.minTileLen);
    smallBackTile = smallBackTile && cutRes.shortTileLen > 0U;
    bool largeFrontTile =
        (cutRes.longTileLen > 2 * tileArgs.mAlignLen) && (cutRes.longTileLen > 2 * tileArgs.minTileLen);
    if (smallBackTile && largeFrontTile) {
        uint64_t demand = tileArgs.mAlignLen * cutRes.numLongTile;
        bool reBalance = (cutRes.shortTileLen + demand) <= cutRes.longTileLen;
        if (reBalance) {
            cutRes.shortTileLen += demand;
            cutRes.longTileLen -= tileArgs.mAlignLen;
        }
    }

    // 假如对齐后尾块长度大于主块，尝试校正使尾块长度<=主块长度
    // 校正目标：主块长度为totalTileCnt均分MPerRank后向上对齐mAlignLen，尾块长度为MPerRank剩余大小
    // 校正检查：totalTileCnt均分后，所有主块加起来最多向尾块抽取 mAlignLen *
    // numLongTile 大小，尾块剩余大小为 (MPerRank/totalTileCnt) - mAlignLen *
    // numLongTile 校正检查：检查条件为 (MPerRank/totalTileCnt) > mAlignLen *
    // totalTileCnt, 确保尾块至少剩下mAlignLen大小
    bool largeBackTile = (cutRes.shortTileLen > cutRes.longTileLen) &&
                         ((totalLen / cutRes.totalTileCnt) > (tileArgs.mAlignLen * cutRes.totalTileCnt));
    if (largeBackTile) {
        cutRes.longTileLen =
            (totalLen / cutRes.totalTileCnt + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    }
}

inline void FormPartition::GenerateInitialPartition(bool smallDimAlignUp)
{
    cutRes.longTileLen = std::max(cutRes.longTileLen, cutRes.shortTileLen);
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen - cutRes.shortTileLen);
    uint64_t newTileLen = cutRes.longTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    // 0.5 is half of mAlignLen
    double alignRatio = ALIGN_RATIO_CALC_BOUND;
    if (!cutRes.shortTileAtBack) {
        alignRatio = ALIGN_RATIO_COMM_BOUND;
    }

    bool alignUpFlag = ((cutRes.longTileLen - newTileLen) > (tileArgs.mAlignLen * alignRatio)) || smallDimAlignUp;
    if (alignUpFlag) {
        cutRes.longTileLen = newTileLen + tileArgs.mAlignLen;
    } else {
        cutRes.longTileLen = newTileLen;
    }
    // find cutRes.totalTileCnt
    cutRes.longTileLen = std::min(cutRes.longTileLen, tileArgs.maxTileLen);
    cutRes.numLongTile = (totalLen - static_cast<uint64_t>(static_cast<double>(cutRes.shortTileLen) * backTileRatio)) /
                         cutRes.longTileLen;
    cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

inline void FormPartition::FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront)
{
    // 流水配平，找到初始切分长度和个数
    GenerateInitialPartition(smallDimAlignUp);

    // 处理剩余长度
    uint64_t resLen = totalLen - cutRes.numLongTile * cutRes.longTileLen - cutRes.numShortTile * cutRes.shortTileLen;
    bool largeBackTile =
        (resLen >= cutRes.shortTileLen) && ((SMALL_TILECNT_PAR * cutRes.shortTileLen) >= cutRes.longTileLen);
    goodLinearityShape =
        (goodLinearityShape && (cutRes.shortTileLen >= tileArgs.mAlignLen)) || (!cutRes.shortTileAtBack);
    bool smallCut = cutRes.numLongTile <= SMALL_TILECNT_PAR;
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    if (balanceCut) { // 重新均匀切分
        cutRes.longTileLen = cutRes.shortTileLen;
        cutRes.numLongTile = (totalLen - cutRes.shortTileLen) / cutRes.longTileLen; // floor
        cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
        cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
        largeBackTile = false;
    }
    if (cutRes.totalTileCnt == TWO && !localAtFront) {
        // 如果前面没有local特殊处理，就把剩余长度均分给长块和短块
        RedistributeTile();
        largeBackTile = false;
    }
    bool moreFrontTile = largeBackTile && (cutRes.longTileLen <= cutRes.shortTileLen + resLen);
    if (moreFrontTile) { // 增加长块个数
        cutRes.numLongTile++;
        CalcBackTile();
    }

    // 检查切分轮次总数符合约束
    tileArgs.maxTileCnt = std::max(tileArgs.maxTileCnt, ONE); // 确保maxTileCnt > 0
    if (cutRes.totalTileCnt > tileArgs.maxTileCnt) {
        uint64_t tileLen = totalLen / tileArgs.maxTileCnt;
        bool isDivisible = (tileLen * tileArgs.maxTileCnt == totalLen) && (tileLen % tileArgs.mAlignLen == 0U);
        if (isDivisible) {
            cutRes.numLongTile = tileArgs.maxTileCnt;
            cutRes.longTileLen = tileLen;
            cutRes.numShortTile = 0U;
            cutRes.shortTileLen = 0U;
            cutRes.totalTileCnt = tileArgs.maxTileCnt;
        }
    }
    MaxTileCntUniform();

    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    if (cutRes.longTileLen > ONE_KBYTE) {
        AlignTileLarge();
    } else {
        AlignTileSmall();
    }
}

inline void FormPartition::NoCutTiling()
{
    cutRes.totalTileCnt = 1U;
    cutRes.numShortTile = 0U;
    cutRes.numLongTile = 1U;
    cutRes.shortTileLen = 0U;
    cutRes.longTileLen = totalLen;
}

inline void FormPartition::LargeBackTileCheck(bool soc310Flag, bool largeCalcCommRatio)
{
    bool largeBackTile = cutRes.shortTileLen >= (cutRes.longTileLen + tileArgs.mAlignLen);
    if (soc310Flag) {
        largeBackTile = (cutRes.shortTileLen >= (cutRes.longTileLen + SOC310_BASE_M));
    }
    if (largeBackTile) {
        cutRes.longTileLen =
            (totalLen / cutRes.totalTileCnt + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen = std::min(totalLen, cutRes.longTileLen);
        cutRes.numLongTile = totalLen / cutRes.longTileLen;
        CalcBackTile();
    }
    if (largeCalcCommRatio) {
        uint64_t moveLen = cutRes.numLongTile * tileArgs.mAlignLen;
        largeBackTile = (cutRes.shortTileLen >= (cutRes.longTileLen + tileArgs.mAlignLen)) &&
                        (cutRes.shortTileLen >= (moveLen + tileArgs.mAlignLen * LARGE_BACKTILE_COMPARE_PAR) &&
                         (cutRes.shortTileLen <= moveLen * LARGE_BACKTILE_COMPARE_PAR));
        if (largeBackTile) {
            cutRes.longTileLen += tileArgs.mAlignLen;
            cutRes.longTileLen =
                (cutRes.longTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
            cutRes.longTileLen = std::min(totalLen, cutRes.longTileLen);
            cutRes.numLongTile = totalLen / cutRes.longTileLen;
            CalcBackTile();
        }
    }
}

inline void FormPartition::CalcShortTile()
{
    cutRes.numShortTile = 0;
    cutRes.shortTileLen = 0;
    if (totalLen != cutRes.longTileLen * cutRes.numLongTile) {
        cutRes.numShortTile = 1;
        cutRes.shortTileLen = totalLen - cutRes.longTileLen * cutRes.numLongTile;
    }
}

inline void FormPartition::CalcBackTile()
{
    CalcShortTile();
    bool tinyBackTile = (cutRes.shortTileLen > 0) && (cutRes.shortTileLen < tileArgs.mAlignLen * 0.5);
    if (tinyBackTile && cutRes.numLongTile > 1) {
        cutRes.numLongTile -= 1;
        cutRes.shortTileLen += cutRes.longTileLen;
    }
    cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
}

// 据经验值，切分块数较多且每块长度接近matmul临界值时，ETE性能会劣化
inline void FormPartition::SmallFrontTileCheck(bool soc310Flag, bool kGreaterThanN)
{
    uint64_t optimalTileLen = tileArgs.minTileLen * OPT_TILE_LEN_RATIO;
    if (soc310Flag) {
        optimalTileLen = tileArgs.minTileLen * OPT_TILE_LEN_RATIO_SOC310P;
    }
    bool smallFrontTile = (cutRes.longTileLen < optimalTileLen) && (totalLen > cutRes.longTileLen * OPT_TILE_CNT);
    bool largeFrontFlag = soc310Flag || (cutRes.shortTileAtBack && kGreaterThanN);
    if (smallFrontTile && largeFrontFlag) {
        optimalTileLen = std::max(optimalTileLen, totalLen / OPT_TILE_CNT);
        optimalTileLen = optimalTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen = optimalTileLen;
        cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
        cutRes.numLongTile = totalLen / cutRes.longTileLen;
        CalcBackTile();
    }
}

inline void FormPartition::RedistributeTile()
{
    uint64_t tmpBackTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    bool redistTileFlag = tmpBackTileLen > 2 * cutRes.shortTileLen;
    if (redistTileFlag) {
        uint64_t redistTileLen = (tmpBackTileLen - cutRes.shortTileLen) / cutRes.totalTileCnt;
        redistTileLen = redistTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
        cutRes.longTileLen += redistTileLen;
    }
    CalcBackTile();
}

inline void FormPartition::CommBoundShortAlign()
{
    if (cutRes.shortTileAtBack || cutRes.numLongTile > 1) {
        return;
    }
    uint64_t alignDownLen = cutRes.shortTileLen / tileArgs.mAlignLen * tileArgs.mAlignLen;
    uint64_t alignDownDiff = cutRes.shortTileLen - alignDownLen;
    uint64_t alignUpLen = (cutRes.shortTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    bool alignUpShortFlag = (alignUpLen <= SMALL_SHORTTILE_BAR && cutRes.longTileLen > SMALL_SHORTTILE_BAR) ||
                            alignDownDiff <= SHORT_ALIGN_DOWN_DIFF;
    if (alignUpShortFlag) {
        cutRes.shortTileLen = alignUpLen;
    }
    cutRes.longTileLen = (totalLen - cutRes.shortTileLen + cutRes.numLongTile - 1) / cutRes.numLongTile;
    cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
}

inline void FormPartition::FitTileLengthContinuous(bool kGreaterThanN, bool largeCalcCommRatio, bool soc310Flag)
{
    cutRes.numLongTile = (totalLen - static_cast<uint64_t>(static_cast<double>(cutRes.shortTileLen) * backTileRatio)) /
                         cutRes.longTileLen;
    cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
    RedistributeTile();
    SmallFrontTileCheck(soc310Flag, kGreaterThanN);
    // check totalTileCnt is valid.
    MaxTileCntUniform();

    LargeBackTileCheck(soc310Flag, largeCalcCommRatio);
    CommBoundShortAlign();
}

inline bool FormPartition::SetShortTileLen(bool noCutFlag)
{
    cutRes.shortTileLen = tileArgs.minTileLen;
    cutRes.numShortTile = 1U;
    // do not cut if mValue is too small
    bool smallMPerRank = (totalLen < cutRes.shortTileLen * 2);
    smallMPerRank = smallMPerRank || noCutFlag;
    if (smallMPerRank) {
        NoCutTiling();
    }
    return smallMPerRank;
}

inline void FormPartition::MaxTileCntUniform()
{
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;
    }
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();
        return;
    }

    // 尾块应该小于主块
    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + tileArgs.mAlignLen - 1) / tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();
}

// ============================================================================
// 五、统一调度接口（来源: mc2_fit_based_balance_tiling.h/.cpp, 继承式改钩子注入）
// ============================================================================

// 调度上下文: 与旧 Mc2FitBasedBalanceTiling 基类的公开成员逐一对应
struct FitBasedBalanceCtx {
    MatmulPerformanceArch35 matmulPerf;
    HCCLPerformanceArch35 commPerf;
    FormPartition tilingM;
    MatmulParameters mmInfo;
    uint64_t rankDim = 2;
    uint64_t rankTileNum = 1;
    uint64_t coreNum = 32;
    double ratioCalcComm = 1.0;
};

// 上下文构造 = 旧基类构造函数逐行对应
inline FitBasedBalanceCtx InitCtx(const TilingArgs &args, KernelType kernelType, const CommFitTable &commTable,
                                  const MatmulPerfTable &mmTable, TopoType topoType = TopoType::STANDARD_CARD,
                                  SocVersion socVersion = SocVersion::SOC950)
{
    FitBasedBalanceCtx ctx{MatmulPerformanceArch35(args, mmTable, socVersion),
                           HCCLPerformanceArch35(args.rankDim, kernelType, commTable, socVersion, topoType),
                           FormPartition(args),
                           MatmulParameters{},
                           args.rankDim,
                           1,
                           args.aicCoreNum,
                           1.0};
    ctx.mmInfo = ctx.matmulPerf.mmShapeInfo_;
    ctx.tilingM.SetMaxTileCnt(MAX_TILE_CNT); // Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
    ctx.rankTileNum = ctx.commPerf.GetRankTileNum();
    return ctx;
}

/*!
 * \brief 统一调度入口（= 旧 Mc2FitBasedBalanceTiling::GetTiling 流程逐行对应）
 *
 * 算子侧通过 HooksT 注入 4 个钩子（原派生类的 override, 静态分发）:
 *   hooks.EstimateMMCommTime(ctx) / hooks.SetShortTileLen(ctx)
 *   hooks.SetLongTileLen(ctx) / hooks.AdjustLongShortTileLen(ctx)
 */
template <typename HooksT>
CutResult Schedule(FitBasedBalanceCtx &ctx, HooksT &hooks)
{
    hooks.EstimateMMCommTime(ctx);
    hooks.SetShortTileLen(ctx);
    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
    } else {
        hooks.SetLongTileLen(ctx);
        hooks.AdjustLongShortTileLen(ctx);
    }
    return ctx.tilingM.cutRes;
}

// ============================================================================
// 六、CutResult ↔ head/tail 适配（L3 输出层）
// ============================================================================

// CutResult -> CommTilingData 的切分轴字段（nonSplitAxisSize 由算子侧按自身布局另填）
// 仅支持 shortTileAtBack == true 的切分（短块前置的通信 bound 切分无法用 head/tail 表达,
// 调用方应继续直接消费 CutResult）。返回 false 表示切分不可适配。
inline bool FillCommTilingDataFromCut(const CutResult &cut, CommTilingData &tiling)
{
    if (!cut.shortTileAtBack) {
        return false; // 短块前置无法用 head/tail 表达
    }
    tiling.splitAxisTileSize = cut.longTileLen;
    tiling.splitAxisTileCnt = cut.numLongTile;
    tiling.splitAxisTailSize = cut.shortTileLen;
    tiling.splitAxisTailCnt = cut.numShortTile;
    return true;
}

// CommTilingData 的切分轴字段 -> CutResult（短块恒后置）
inline CutResult CutResultFromCommTiling(const CommTilingData &tiling)
{
    CutResult cut;
    cut.shortTileAtBack = true;
    cut.longTileLen = tiling.splitAxisTileSize;
    cut.numLongTile = tiling.splitAxisTileCnt;
    cut.shortTileLen = tiling.splitAxisTailSize;
    cut.numShortTile = tiling.splitAxisTailCnt;
    cut.totalTileCnt = cut.numLongTile + cut.numShortTile;
    return cut;
}

} // namespace fit_based_balance

// ============================================================================
// 第二部分【URMA 拓展分支 · wave-packing】通信引擎: URMA（comm_mode = "aiv_urma"）
// （v2 allto_all_matmul_v2 已验证代码原样迁入, 逐字节未改; 其模型/调度/门控语义
//  见 ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_PLAN.md）
// ============================================================================

/*!
 * \brief Linear communication latency model: y(us) = slope * x(MB) + intercept(us).
 */
struct CommLinearModel {
    double slopeUsPerMb = 0.0;
    double interceptUs = 0.0;

    double RoundTimeUs(double roundSizeMb) const
    {
        return slopeUsPerMb * roundSizeMb + interceptUs;
    }
};

/*!
 * \brief Theoretical matmul latency model: t(us) = MACs / (effMac * cyclePerUs * coreNum).
 */
struct MatmulTheoryModel {
    double effMacPerCycle = 0.0;
    double cyclePerUs = 1689.6; // A5: 1.65 cycle/ns
    uint64_t coreNum = 1;

    double TimeUs(uint64_t m, uint64_t n, uint64_t kTotal) const
    {
        if ((effMacPerCycle <= 0.0) || (cyclePerUs <= 0.0) || (coreNum == 0UL) || (m == 0UL) || (n == 0UL) ||
            (kTotal == 0UL)) {
            return 0.0;
        }
        const long double macs =
            static_cast<long double>(m) * static_cast<long double>(n) * static_cast<long double>(kTotal);
        const long double denom = static_cast<long double>(effMacPerCycle) * static_cast<long double>(cyclePerUs) *
                                  static_cast<long double>(coreNum);
        return static_cast<double>(macs / denom);
    }
};

/*!
 * \brief Tunables shared by every fit-balance user (values owned by op host headers).
 */
struct FitBalanceConfig {
    bool enabled = false;        // master switch, wired by the op host side
    double balanceRatio = 1.2;   // comm/calc ratio below this keeps k = 1 (waves already match)
    double roundCostUs = 10.0;   // fixed cost per extra round when deriving the round cap
    double tinyTailRatio = 0.5;  // tail rounds below this ratio of head are merged into the last head round
    bool enableTailMerge = true; // fold tiny tail rounds (needs kernel-side explicit round count)
    uint64_t maxTileCnt = 16UL;  // hard cap of communication rounds
    uint64_t minRounds = 3UL;    // lower bound of the derived round cap
};

/*!
 * \brief One compute wave derived from the matmul tiling result.
 */
struct MmWaveDesc {
    uint64_t waveM = 0;   // M span of one full core pass: ceil(usedCoreNum / nTile) * baseM
    uint64_t waveCnt = 0; // ceil(m / waveM)
};

inline MmWaveDesc CalcMmWave(uint64_t m, uint64_t n, uint64_t baseM, uint64_t baseN, uint64_t usedCoreNum)
{
    MmWaveDesc wave;
    if ((baseM == 0UL) || (baseN == 0UL) || (m == 0UL)) {
        return wave;
    }
    const uint64_t nTile = (n + baseN - 1UL) / baseN;
    const uint64_t blocksPerWave = (nTile == 0UL) ? 1UL : ((usedCoreNum + nTile - 1UL) / nTile);
    wave.waveM = (blocksPerWave == 0UL) ? baseM : (blocksPerWave * baseM);
    wave.waveCnt = (m + wave.waveM - 1UL) / wave.waveM;
    return wave;
}

/*!
 * \brief Cut result aligned with the kernel-side CommTilingData head/tail layout.
 *
 * Invariants kept for kernel safety (see the AllToAll URMA impl):
 *   - headSize is a multiple of baseM, so no matmul block straddles two rounds;
 *   - headCnt + tailCnt == ceil(m / headSize), matching the AIC wait mapping.
 */
struct FitBalanceCut {
    uint64_t headSize = 0; // splitAxisTileSize
    uint64_t headCnt = 0;  // splitAxisTileCnt
    uint64_t tailSize = 0; // splitAxisTailSize
    uint64_t tailCnt = 0;  // splitAxisTailCnt (0 or 1)

    // Diagnostics for logging and offline auditing.
    uint64_t chosenK = 1;
    uint64_t rounds = 1;
    double ratioCalcComm = 1.0;
    double estTimeUs = 0.0;
    double estTimeUsK1 = 0.0;
    bool noCut = false;
    bool mergedTail = false; // tail was folded into the last head round
};

/*!
 * \brief Fully data-driven input of the shared scheduling core.
 *
 * perRowBytes / commFanout fold the op-specific traffic formula into the
 * shared round size:  roundSizeMb = commFanout * mSpan * perRowBytes / 1MB.
 */
struct FitBalanceScheduleInput {
    // Problem shape.
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t kTotal = 0;

    // Matmul tiling result (owned by the quant matmul tiling engine).
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t usedCoreNum = 1;

    // Communication traffic folding (AllToAll PUT: fanout = rankDim - 1).
    uint64_t commFanout = 1;
    double perRowBytes = 0.0;

    // Calibrated performance models.
    CommLinearModel comm;
    MatmulTheoryModel mm;

    FitBalanceConfig cfg;
};

namespace detail {
constexpr double kFitBalanceMbBytes = 1024.0 * 1024.0;
} // namespace detail

inline double CalcRoundSizeMb(const FitBalanceScheduleInput &in, uint64_t mSpan)
{
    return static_cast<double>(in.commFanout * mSpan) * in.perRowBytes / detail::kFitBalanceMbBytes;
}

// Comm/calc ratio, always >= 1 (legacy ratioCalcComm_).
inline double CalcCommCalcRatio(double commUs, double mmUs)
{
    const double hi = std::max(commUs, mmUs);
    const double lo = std::min(commUs, mmUs);
    if (lo <= 0.0) {
        return (hi > 0.0) ? 1e12 : 1.0;
    }
    return hi / lo;
}

// Round cap derived from total communication time and the per-round cost.
inline uint64_t CalcMaxRounds(double totalCommUs, const FitBalanceConfig &cfg)
{
    double rounds = totalCommUs / std::max(cfg.roundCostUs, 1e-9);
    rounds = std::max(rounds, static_cast<double>(cfg.minRounds));
    const uint64_t capped = static_cast<uint64_t>(rounds);
    return std::min(capped, cfg.maxTileCnt);
}

/*!
 * \brief Shared fit-balance decision core.
 *
 * Flow: wave maths -> total time / ratio estimate -> no-cut gates -> k search
 * (powers of two, round-capped) -> head/tail emission. The output maps 1:1 to
 * CommTilingData; k = 1 reproduces the plain CoreOccupancyFirstTiling behaviour.
 */
inline FitBalanceCut FitBalanceSchedule(const FitBalanceScheduleInput &in)
{
    FitBalanceCut cut;
    const MmWaveDesc wave = CalcMmWave(in.m, in.n, in.baseM, in.baseN, in.usedCoreNum);
    if (wave.waveM == 0UL) {
        return cut;
    }
    const uint64_t waveM = wave.waveM;

    const double totalCommUs = in.comm.RoundTimeUs(CalcRoundSizeMb(in, in.m));
    const double totalMmUs = in.mm.TimeUs(in.m, in.n, in.kTotal);
    cut.ratioCalcComm = CalcCommCalcRatio(totalCommUs, totalMmUs);

    // Emit a cut from a head size (headCnt full rounds + one tail round).
    auto makeCut = [&in](uint64_t head) {
        FitBalanceCut c;
        c.headSize = head;
        c.headCnt = in.m / head;
        c.tailSize = in.m % head;
        c.tailCnt = (c.tailSize > 0UL) ? 1UL : 0UL;
        c.rounds = c.headCnt + c.tailCnt;
        return c;
    };
    // Single-round cut covering the whole M range. headSize = m keeps the
    // kernel mapping self-consistent for any m: totalTiles = ceil(m / m) = 1
    // matches commTurn = 1. (The legacy headCnt = 0 / tail = m shape is only
    // valid when m <= waveM; emitting it with m > waveM would make the AIC wait
    // on flags the AIV never raises.)
    auto singleCut = [&in]() {
        FitBalanceCut c;
        c.headSize = in.m;
        c.headCnt = 1UL;
        c.tailSize = 0UL;
        c.tailCnt = 0UL;
        c.rounds = 1UL;
        return c;
    };
    // Pipeline critical-path estimate for "headCnt head-sized rounds followed by
    // one last round of lastSize rows" (lastSize == 0 means all rounds uniform):
    //   T = C_head + (headCnt-1) * max(M_head, C_head) + max(M_head, C_last) + M_last
    // Round i's comm overlaps the previous round's compute; the last round ships
    // lastSize rows so its comm is lastSize-sized (tail-aware, perf r3 fix).
    auto pipelineEst = [&in](uint64_t headCnt, uint64_t head, uint64_t lastSize) -> double {
        const double commHead = in.comm.RoundTimeUs(CalcRoundSizeMb(in, head));
        const double mmHead = in.mm.TimeUs(head, in.n, in.kTotal);
        if (lastSize == 0UL) {
            return commHead + static_cast<double>(headCnt - 1UL) * std::max(mmHead, commHead) + mmHead;
        }
        const double commLast = in.comm.RoundTimeUs(CalcRoundSizeMb(in, lastSize));
        const double mmLast = in.mm.TimeUs(lastSize, in.n, in.kTotal);
        return commHead + static_cast<double>(headCnt - 1UL) * std::max(mmHead, commHead) + std::max(mmHead, commLast) +
               mmLast;
    };
    // Critical path estimate for a given head size.
    auto estTime = [&in, &pipelineEst](uint64_t head) -> double {
        if (head >= in.m) {
            // Single round: communication and compute serialise.
            return in.comm.RoundTimeUs(CalcRoundSizeMb(in, in.m)) + in.mm.TimeUs(in.m, in.n, in.kTotal);
        }
        return pipelineEst(in.m / head, head, in.m % head);
    };

    cut.estTimeUsK1 = estTime(waveM);

    // Degenerate fast path: a single wave already covers the whole M range, so
    // the finest cut degenerates to a single round anyway.
    // Note: there is deliberately NO absolute-time "not worth splitting" gate.
    // A single round vs. multi-round cut is decided by the estTime competition
    // below: estTime's leading comm(head) term already prices the first-block
    // start latency, and a fixed threshold was shown to regress compute-bound
    // small shapes by forcing a full-M first round (see tiling-plan perf report).
    if (wave.waveCnt < 2UL) {
        cut = singleCut();
        cut.noCut = true;
        cut.chosenK = 0UL;
        cut.estTimeUs = totalCommUs + totalMmUs;
        return cut;
    }
    // Balanced: rounds naturally match compute waves, keep the k = 1 behaviour.
    if (cut.ratioCalcComm < in.cfg.balanceRatio) {
        cut = makeCut(waveM);
        cut.chosenK = 1UL;
        cut.estTimeUs = cut.estTimeUsK1;
        return cut;
    }

    // k search over powers of two, respecting the derived round cap. k = 1 is
    // the baseline; single-round candidates enter through head >= m.
    const uint64_t maxRounds = CalcMaxRounds(totalCommUs, in.cfg);
    double bestTime = 0.0;
    uint64_t bestHead = 0UL;
    if (wave.waveCnt <= maxRounds) {
        bestTime = cut.estTimeUsK1;
        bestHead = waveM;
    }
    for (uint64_t k = 2UL; k <= wave.waveCnt; k <<= 1UL) {
        const uint64_t head = k * waveM;
        if (head >= in.m) {
            const double t = totalCommUs + totalMmUs;
            if ((bestHead == 0UL) || (t < bestTime)) {
                bestTime = t;
                bestHead = head;
            }
            break; // larger k degenerates to the same single round
        }
        const uint64_t rounds = in.m / head + ((in.m % head > 0UL) ? 1UL : 0UL);
        if (rounds > maxRounds) {
            continue;
        }
        const double t = estTime(head);
        if ((bestHead == 0UL) || (t < bestTime)) {
            bestTime = t;
            bestHead = head;
        }
    }
    if (bestHead == 0UL) {
        // No candidate satisfies the round cap: fall back to a single round.
        cut = singleCut();
        cut.noCut = true;
        cut.estTimeUs = totalCommUs + totalMmUs;
        return cut;
    }
    cut = makeCut(bestHead);
    cut.chosenK = bestHead / waveM;
    cut.estTimeUs = bestTime;
    cut.noCut = (cut.rounds <= 1UL); // single round may legitimately win the estTime search

    // Tiny-tail merge: fold a sub-threshold tail round into the last head
    // round (headCnt - 1 uniform rounds + one bigger tail round). Only applied
    // on the searched path; noCut / balanced gates stay bit-identical to the
    // plain CoreOccupancyFirstTiling behaviour. Kernel contract: the merged
    // round still starts at a multiple of headSize, and the fused urma kernel
    // takes the explicit round count, so the AIC wait mapping stays correct.
    // Guards learned from perf regressions:
    //  - headCnt >= 2: at least one head round must survive the merge — folding
    //    a 2-round cut into a single full-M round destroys the first-block start
    //    latency the estTime search just paid for (perf r3 PERF_10).
    //  - merge must be priced-neutral: the merged schedule is re-estimated with
    //    the same tail-aware pipeline model and may exceed the unmerged estimate
    //    by at most 2% — the saved round's sync cost is real but small, while
    //    the model itself carries ~5% error (calib 表E), so a merge priced worse
    //    than 2% is not trustworthy. A geometrically tiny tail can still merge
    //    into an oversized last round whose comm delays the compute of the rows
    //    it absorbed (perf r4 2560_2048_4096 fp4: +6% priced, -3.5% measured;
    //    perf r3/r4 114172 fp4: +0.3% priced, merge measured fine).
    if (in.cfg.enableTailMerge && (cut.tailCnt == 1UL) && (cut.headCnt >= 2UL) &&
        (static_cast<double>(cut.tailSize) <= in.cfg.tinyTailRatio * static_cast<double>(cut.headSize))) {
        constexpr double kTailMergeMarginRatio = 0.02;
        const double mergedEst = pipelineEst(cut.headCnt - 1UL, cut.headSize, cut.headSize + cut.tailSize);
        if (mergedEst <= (cut.estTimeUs * (1.0 + kTailMergeMarginRatio))) {
            cut.headCnt -= 1UL;
            cut.tailSize += cut.headSize; // merged round = head + tail, still prefix-aligned
            cut.rounds = cut.headCnt + 1UL;
            cut.mergedTail = true;
            cut.estTimeUs = mergedEst;
        }
    }
    return cut;
}

} // namespace apace

#endif // __APACE_FIT_BALANCE_TILING_H__
