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
 * \brief fit-balance tiling 家族唯一公共文件（三层结构, 两分支共居, 迁移方案 §2.2/§4.9）
 *
 * 分支选择（由通信引擎的每轮固定开销经济性决定, 见迁移方案 §2.2 分支表）:
 *   - 自由长度分支（3.1 求解器）: HCCL AICPU/CCU 引擎, 每轮开销大 → 反解最优轮长压轮数;
 *   - wave-packing 拓展分支（3.2 求解器）: URMA 引擎（comm_mode = "aiv_urma"）, 每轮开销
 *     ~4.5us 近乎免费 → mm 恒最优, 通信按计算波整数倍打包, estTime 竞争决策。
 *
 * 三层布局（按依赖顺序编排; 与方案 §2.2 的 L1 模型/L2 求解/L3 切分输出概念对应）:
 *   零、公共类型与常量（两分支共享词汇）
 *   一、模型层: 1.1 HCCL 通信模型 / 1.2 URMA 线性模型与其拟合表 schema /
 *               1.3 matmul 模型（两分支近似式不同——cubeUtil+gradient vs effMac 单常数
 *               理论式——纯重构下不可合并, 并列保留, 勿强行统一）
 *   二、切分与输出层: FormPartition + CutResult 适配
 *   三、求解层: 3.1 自由长度求解器（ctx+钩子注入）/ 3.2 wave-packing 求解器
 *
 * 统一引擎分派（三引擎接入结构完全一致, 见 CommEngine/ParseCommEngine）:
 *   同时支持多通信模式的算子按同一模式接入——AIV_URMA 走 wave-packing 分支
 *   （三、3.2 + AllToAll URMA 预制）, CCU/AICPU 走自由长度分支（三、3.1 +
 *   各引擎拟合表）; 每引擎一组标定常量 + 一张表 + 一个分派点, 无任何引擎特殊路径。
 *
 * 等价性基线: 自由长度分支 2026-09-14 P0 提取经 162 用例 bit 级验证;
 * wave-packing 分支为 v2 已验证代码（R1~R6 回归）原样迁入, 决策 golden 43 用例 bit 级复验。
 */
#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

// ge::DataType 经 register/tilingdata_base.h 与 tiling/tiling_api.h 的既有链路提供
// (不直接 include graph 头: 部分 TU 的 ge 链路来自不同 toolkit 路径, 直接引入会双重定义)
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_data_struct_define.h"

namespace apace {
namespace fit_balance {

// ============================================================================
// 零、公共类型与常量（两分支共享词汇: CutResult/TilingArgs/枚举等）
// ============================================================================
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
constexpr uint64_t MIN_COMM_RANKDIM = 2;

enum class SocVersion {
    SOC950 = 4, // 值 4 为键约定: 查表 key 前缀 "{soc}_" 的 soc 段沿用旧枚举序(950=4), 勿改;
                // 引擎仅支持 arch35/950(旧芯片枚举值随旧芯片逻辑剥离, B 家族原位副本保留完整值域)
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

struct L2CacheEstimateParameters {
    uint64_t sizePartL2;
    uint64_t sizeFullL2;
    double utilPartL2;
    double utilFullL2;
    double utilOutL2;
};

enum class HCCLType {
    FULL_MESH, // 950 仅全互联一种通信形态(旧芯片环网形态随旧芯片逻辑剥离)
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

// ====== 通信引擎枚举与解析（三引擎统一分派的公共词汇）======
enum class CommEngine {
    AICPU,
    CCU,
    AIV_URMA,
};

// comm_mode 属性解析: "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含空指针/默认"ai_cpu")→AICPU
inline CommEngine ParseCommEngine(const char *commMode)
{
    if (commMode == nullptr) {
        return CommEngine::AICPU;
    }
    if (strncmp(commMode, "aiv_urma", 8U) == 0) {
        return CommEngine::AIV_URMA;
    }
    if (strncmp(commMode, "ccu", 3U) == 0) {
        return CommEngine::CCU;
    }
    return CommEngine::AICPU;
}

// ============================================================================
// 一、模型层
// ============================================================================
// ---- 1.1 HCCL 通信模型与拟合表（自由长度分支, arch35/950 专用）----
// 由旧 HCCLPerformanceModel + HCCLPerformanceArch35 合并精简(2026-09-15):
// 本引擎按设计仅在 arch35(SOC950) 标卡 4P/8P 生效, 旧芯片分支(310P 环网/910_93
// 双环/基表两段式查表——其结果恒被 arch35 四段式查表覆盖)经 grep 证实 5 个消费
// 算子零调用后剥离; 老芯片的真实需求由原位保留的 B 家族(mc2/common/op_host/
// op_tiling)满足。活跃 API 面(5 算子钩子实测): CommTime/InverseCommTime/
// SetCommShapeLen/SetCommDTypeSize/GetCommDTypeSize/SetCommDtypeSizeExpansionFraction/
// commTimeFactor_/GetRankTileNum(InitCtx 内部)。

constexpr double FULL_MESH_TIME_FACTOR = 2.0; // allreduce time is 2.0x allgather or reducescatter
constexpr uint64_t FITTING_RANK = 8;          // 拟合采用 8 die 通信

// ====== 拟合表注入 schema（行数据由算子侧标定头维护）======
struct CommFitRow {
    const char *key; // 查表 key: "{soc}_{kernelType}_{topoType}_{rankDim}", soc 段恒为 4(SOC950)
    HCCLFittingParameters params;
};

struct CommFitTable {
    const CommFitRow *rows;
    size_t rowCount;
    HCCLFittingParameters defaultParams; // key 未命中时的回退行 + rankDim 折算因子缩放
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

class HCCLPerformanceArch35 {
public:
    HCCLFittingParameters commEstimatePar_;
    HCCLInfo commTypeInfo_;
    double commTimeFactor_ = 1.0; // allreduce time is 2.0x allgather or reducescatter
    uint64_t lookUpTileNum_ = 1;  // 数据拟合查表参数
    std::string keyToFittingMap_;

    // Constructor（= 旧 基类构造 + Arch35 构造的净状态; soc 恒 SOC950, key soc 段恒为 4）
    HCCLPerformanceArch35(uint32_t inputRankDim, KernelType inputKernelType, const CommFitTable &fitTable,
                          TopoType topoType = TopoType::STANDARD_CARD)
        : fitTable_(fitTable)
    {
        commTypeInfo_.kernelType = inputKernelType; // 区分哪个MC2算子
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_COMM_RANKDIM); // 并行维度最小为2
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        InitParametersForFullMesh();
        commTypeInfo_.maxStepSize = 1;
        InitArch35Parameters();
        commTypeInfo_.topoType = topoType;
        GetCommArch35Parameters();
    };

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
    void ChangeCommTimeFactorByDivision(double factor)
    {
        if (factor > 0) {
            commTimeFactor_ /= factor;
        }
    }
    void SetFullMeshCommTimeFactor()
    {
        // 通信查表使用总数据量，实际时间 = 总数据量 / rankDim / 带宽; 拟合采用 8 die 通信
        commTimeFactor_ = FULL_MESH_TIME_FACTOR;
        ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(commTypeInfo_.rankDim));
    }
    void InitParametersForFullMesh()
    {
        commTypeInfo_.commMethod = HCCLType::FULL_MESH;
        if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
            (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER)) {
            SetFullMeshCommTimeFactor();
            lookUpTileNum_ = commTypeInfo_.rankDim;
        }
    }
    uint64_t GetFullMeshRankTileNum()
    {
        if (commTypeInfo_.kernelType == KernelType::ALL_GATHER) {
            return commTypeInfo_.rankDim - 1;
        }
        return commTypeInfo_.rankDim;
    }
    uint64_t GetRankTileNum()
    {
        if (commTypeInfo_.kernelType != KernelType::ALL_REDUCE) {
            return GetFullMeshRankTileNum();
        }
        return 1;
    }
    void InitArch35Parameters()
    {
        if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
            (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL)) {
            lookUpTileNum_ = commTypeInfo_.rankDim;
            commTimeFactor_ = 1;
        }
    }
    std::string GetCommInfoString()
    {
        // key = "{soc}_{kernelType}_{topoType}_{rankDim}", soc 段恒为 4(SOC950)
        return std::string("4_") + std::to_string(static_cast<int>(commTypeInfo_.kernelType)) + "_" +
               std::to_string(static_cast<int>(commTypeInfo_.topoType)) + "_" +
               std::to_string(static_cast<int>(commTypeInfo_.rankDim));
    };
    void GetCommArch35Parameters()
    {
        keyToFittingMap_ = GetCommInfoString();
        const HCCLFittingParameters *hit = FindCommFitParams(fitTable_, keyToFittingMap_);
        if (hit != nullptr) {
            commEstimatePar_ = *hit;
        } else {
            commEstimatePar_ = fitTable_.defaultParams;
            ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) /
                                           static_cast<double>(commTypeInfo_.rankDim));
        }
    }

    // 性能拟合函数
    double CommTime(uint64_t mSize) const
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

    uint64_t InverseCommTime(double targetTime) const
    {
        double tmpSize = commEstimatePar_.sizeToTimeBoundary1;
        if (targetTime > commEstimatePar_.timeToSizeBoundary2) {
            tmpSize =
                (targetTime - commEstimatePar_.sizeToTimeLinearOffset) / commEstimatePar_.sizeToTimeLinearGradient;
        } else if (targetTime > commEstimatePar_.timeToSizeBoundary1) {
            if (commTypeInfo_.kernelType == KernelType::ALL_REDUCE &&
                commEstimatePar_.timeToSizeParabolicPar2 != 0.0f) {
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

protected:
    const CommFitTable &fitTable_; // 拟合表由构造方注入（算子侧标定头）
};

// ---- 1.2 URMA 线性通信模型与拟合表 schema（wave-packing 分支）----
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

// ====== URMA 拟合表 schema（行数据由算子侧标定头提供, wave-packing 分支专用）======
struct UrmaCommFitRow { // 原 v2 算子侧 CommFitRow, 更名以区别 HCCL 表行(本表按 rankDim 查, 无 key 字符串)
    uint32_t rankDim;
    double slopeUsPerMb;
    double interceptUs;
};

struct RoundCostRow {
    uint32_t rankDim;
    double roundCostUs;
};

struct EffMacRow {
    bool isFp4;
    uint32_t precisionMode; // 0/1 一档(填 0), 2 一档(填 2)
    double effMacPerCycle;
};

inline const UrmaCommFitRow *FindUrmaCommFit(const UrmaCommFitRow *rows, size_t rowCount, uint32_t rankDim)
{
    for (size_t i = 0; i < rowCount; ++i) {
        if (rows[i].rankDim == rankDim) {
            return &rows[i];
        }
    }
    return nullptr;
}

// 未命中返回调用方提供的缺省值
inline double FindRoundCost(const RoundCostRow *rows, size_t rowCount, uint32_t rankDim, double defaultUs)
{
    for (size_t i = 0; i < rowCount; ++i) {
        if (rows[i].rankDim == rankDim) {
            return rows[i].roundCostUs;
        }
    }
    return defaultUs;
}

inline double FindEffMac(const EffMacRow *rows, size_t rowCount, bool isFp4, uint32_t precisionMode)
{
    // mode 0 与 1 共用一档; 归一化后查表, 未命中返回 0 表示缺数据。
    const uint32_t modeClass = (precisionMode == 2U) ? 2U : 0U;
    for (size_t i = 0; i < rowCount; ++i) {
        if (rows[i].isFp4 == isFp4 && rows[i].precisionMode == modeClass) {
            return rows[i].effMacPerCycle;
        }
    }
    return 0.0;
}

// ---- 1.3 matmul 模型（自由长度分支的 HCCL 算力/梯度模型; wave-packing 的 effMac
// 理论式在 1.2 MatmulTheoryModel, 两分支近似式不同并列保留, 勿强行统一）----
// 由旧 MatmulPerformanceModel + MatmulPerformanceArch35 合并精简(2026-09-15),
// 剥离 310P/910_B4 分支与 4 参 FindCubeUtil(含其独占的 K 对齐修正链)——
// 5 个消费算子钩子经 grep 证实仅用 1 参 FindCubeUtil/FindCubeUtilByL2Usage。

namespace MatmulPerformance {
constexpr double COMPUTES_PER_CYCLE = 4096; // 基础算力(查表未命中回退, 标定头回填同值)
// start: cycle per micro second (SOC950)
constexpr double CYCLE_PER_MICRO_SEC_NPUARCH_3510 = 1.65 * ONE_KBYTE;
// end: cycle per micro second
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double FRONT_UTIL_EXPONENT_COEFFICIENT = 0.6;
constexpr double CUBE_UTIL_EXPONENT_COEFFICIENT = 0.7;
constexpr double MAX_CUBE_UTIL = 0.95;
constexpr double AVERAGE_CUBE_UTIL = 0.75;
} // namespace MatmulPerformance

// ====== 模型参数注入 schema（行数据由算子侧标定头维护）======
// key 规则见 MatmulPerformanceArch35::GetCalcTypeString: "soc_calcType_aDtype_bDtype_cDtype",
// soc 段恒为 4(SOC950)
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
    // L2 档位查表（旧 L2_PARAMETER_MAP; FindCubeUtilByL2Usage 的 diff/maxTileLen 仍消费）
    const L2ParamRow *l2Rows;
    size_t l2Count;
    L2CacheEstimateParameters defaultL2Params; // key 未命中回退（旧 DEFAULT_KEY 行内容 {128, 192, 0.85, 0.75, 0.65}）
    // arch35 最优基本块查表（旧 TILING_BEST_BASE_MAP）
    const BestBaseRow *bestBaseRows;
    size_t bestBaseCount;
    TilingBestBaseBlock defaultBaseBlock; // soc 未命中回退（旧实现为 {BASE_BLOCK_M/N/K}, 由标定头回填同值）
};

class MatmulPerformanceArch35 {
public:
    MatmulParameters mmShapeInfo_;
    MatmulCalcType calcType_; // Distinguish the non-quantization, full-quantization and fake-quantization
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;

    // Constructor（= 旧 基类构造 + Arch35 构造的净状态; soc 恒 SOC950, 频率恒 1689.6）
    MatmulPerformanceArch35(const TilingArgs &args, const MatmulPerfTable &perfTable)
        : perfTable_(perfTable)
    {
        mmShapeInfo_.socType = SocVersion::SOC950;
        mmShapeInfo_.coreNum = args.aicCoreNum; // 每die核数
        mmShapeInfo_.inMatrixADtypeSize = args.inputDtypeSize;
        mmShapeInfo_.inMatrixBDtypeSize = mmShapeInfo_.inMatrixADtypeSize;
        mmShapeInfo_.outMatrixCDtypeSize = args.outputDtypeSize;
        mmShapeInfo_.mValue = args.mValue;
        mmShapeInfo_.nValue = args.nValue;
        mmShapeInfo_.kValue = args.kValue;
        // 获取后续计算用的基本块(先默认, 再按 arch35 最优基本块覆盖)
        mmShapeInfo_.baseM = BASE_BLOCK_M;
        mmShapeInfo_.baseN = BASE_BLOCK_N;
        mmShapeInfo_.baseK = BASE_BLOCK_K;
        mmShapeInfo_.batchSize = 1UL; // 初始值
        SetCalcType(args);
        mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC_NPUARCH_3510;
        GetMachineParameters();
        TilingBestBaseBlock bestBaseBlock = GetBestBaseBlock();
        mmShapeInfo_.baseM = bestBaseBlock.baseM;
        mmShapeInfo_.baseN = bestBaseBlock.baseN;
        mmShapeInfo_.baseK = bestBaseBlock.baseK;
    };

    // 950 量化语义: fp8/HIF8 或 A8W8 量化, 其余按 FP16 建模
    // （旧实现的 310P A16W8 与 910_B INT8 排除分支随旧芯片逻辑剥离）
    void SetCalcType(const TilingArgs &args)
    {
        if ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
            (args.aType == matmul_tiling::DataType::DT_INT8)) {
            calcType_ = MatmulCalcType::QUANT;
        }
    }

    std::string GetCalcTypeString()
    {
        return std::to_string(static_cast<int>(mmShapeInfo_.socType)) + "_" +
               std::to_string(static_cast<int>(calcType_)) + "_" + std::to_string(mmShapeInfo_.inMatrixADtypeSize) +
               "_" + std::to_string(mmShapeInfo_.inMatrixBDtypeSize) + "_" +
               std::to_string(mmShapeInfo_.outMatrixCDtypeSize);
    }
    void GetMachineParameters()
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
    // L2 档位利用率 + 950 调和平均公式(950 专用模型, 调和分支无条件执行)
    double FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t *maxTileLen = nullptr)
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

        /*950 cube利用率计算公式：两个输入为M、N的调和平均数和K。MN调和平均数的阈值为820，K阈值为1280，
        小于这个阈值认为提供负增益，大于这个阈值认为提供正增益，但正增益最高不超过1.4倍。*/
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
    // arch35 单参版本(消费算子钩子唯一使用的入口)
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
    TilingBestBaseBlock GetBestBaseBlock()
    {
        for (size_t i = 0; i < perfTable_.bestBaseCount; ++i) {
            if (perfTable_.bestBaseRows[i].soc == mmShapeInfo_.socType) {
                return perfTable_.bestBaseRows[i].block;
            }
        }
        return perfTable_.defaultBaseBlock;
    }
    void GetMatmulGradient()
    {
        uint64_t tmpN = std::max(mmShapeInfo_.baseN, mmShapeInfo_.nValue);
        uint64_t tmpK = std::max(mmShapeInfo_.baseK, mmShapeInfo_.kValue);
        matmulGradient_ = (tmpN * tmpK) / (mmShapeInfo_.computesPerCycle * mmShapeInfo_.cyclePerMicroSec *
                                           mmShapeInfo_.coreNum * cubeUtil_);
    }
    uint32_t GetBaseM()
    {
        return mmShapeInfo_.baseM;
    };
    uint32_t GetBaseN()
    {
        return mmShapeInfo_.baseN;
    };
    // 性能预测
    double MatmulTime(uint64_t tileSize, uint64_t rankTileNum)
    {
        tileSize = std::max(mmShapeInfo_.baseM, tileSize);
        return static_cast<double>(tileSize) * static_cast<double>(rankTileNum) * matmulGradient_;
    }
    uint64_t InverseMatmulTime(double targetTime, uint64_t rankTileNum)
    {
        uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rankTileNum) * matmulGradient_));
        tileSize = std::max(mmShapeInfo_.baseM, tileSize);
        return tileSize;
    }

protected:
    const MatmulPerfTable &perfTable_; // 模型参数表由构造方注入（算子侧标定头）
};

// ---- 1.3b wave-packing 分支的 matmul 理论式(effMac 单常数, 与上方 HCCL 模型并列)----

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

// ============================================================================
// 二、切分与输出层（FormPartition + CutResult 适配）
// ============================================================================
// ============================================================================

// (2026-09-15 精简: 连续切分路径 FitTileLengthContinuous/SmallFrontTileCheck/
//  LargeBackTileCheck/CommBoundShortAlign 为 B 家族(公式化)专用, A 家族 5 算子钩子
//  经 grep 证实零调用, 已随旧芯片逻辑剥离; B 家族原位副本不受影响)
constexpr uint64_t LARGE_NK_BAR_BASE = 32;
constexpr uint64_t SMALL_TILECNT_PAR = 2;
constexpr double BACKTILE_ALIGN_RATIO = 0.75;
constexpr double ALIGN_RATIO_CALC_BOUND = 0.5;
constexpr double ALIGN_RATIO_COMM_BOUND = 0.75;

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

    // 离散切分对齐 (A 家族全部算子的 Adjust 钩子使用)
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront = false);
    void CalcShortTile();
    void CalcBackTile();
    void RedistributeTile();
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

// ============================================================================
// 三、求解层（两分支求解器）
// ============================================================================
// ---- 3.1 自由长度求解器（ctx + 钩子注入, HCCL AICPU/CCU）----
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
// (soc 恒 SOC950: 引擎仅支持 arch35, SocVersion 参数已随旧芯片逻辑剥离)
inline FitBasedBalanceCtx InitCtx(const TilingArgs &args, KernelType kernelType, const CommFitTable &commTable,
                                  const MatmulPerfTable &mmTable, TopoType topoType = TopoType::STANDARD_CARD)
{
    FitBasedBalanceCtx ctx{MatmulPerformanceArch35(args, mmTable),
                           HCCLPerformanceArch35(args.rankDim, kernelType, commTable, topoType),
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
// ---- 3.2 wave-packing 求解器（estTime k 搜索 + 定价尾轮合并, URMA）----
/*!
 * \brief Tunables shared by every fit-balance user (values owned by op host headers).
 */
struct FitBalanceConfig {
    bool enabled = false;        // master switch, wired by the op host side
    double balanceRatio = 1.2;   // comm/calc ratio below this keeps k = 1 (waves already match)
    double roundCostUs = 10.0;   // fixed cost per extra round when deriving the round cap
    double tinyTailRatio = 0.5;  // tail rounds below this ratio of head are merged into the last head round
    bool enableTailMerge = true; // fold tiny tail rounds (needs kernel-side explicit round count)
    double tailMergeMarginRatio =
        0.02; // 尾轮合并定价容差: mergedEst ≤ est×(1+margin) 才放行(原引擎内常量, 提为算子可配)
    uint64_t maxTileCnt = 16UL; // hard cap of communication rounds
    uint64_t minRounds = 3UL;   // lower bound of the derived round cap
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
    // 诊断字段(ratioCalcComm/estTimeUsK1)在下方 makeCut/singleCut 的对象替换中会丢失,
    // 先存局部量, 每个替换点回填(决策字段不受影响)。
    const double ratioCalcDiag = CalcCommCalcRatio(totalCommUs, totalMmUs);
    cut.ratioCalcComm = ratioCalcDiag;

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

    const double estK1Diag = estTime(waveM);
    cut.estTimeUsK1 = estK1Diag;

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
        cut.ratioCalcComm = ratioCalcDiag;
        cut.estTimeUsK1 = estK1Diag;
        return cut;
    }
    // Balanced: rounds naturally match compute waves, keep the k = 1 behaviour.
    if (cut.ratioCalcComm < in.cfg.balanceRatio) {
        cut = makeCut(waveM);
        cut.chosenK = 1UL;
        cut.estTimeUs = estK1Diag; // 原 cut.estTimeUsK1 此前恒为 0(makeCut 重置), 修复后语义=作者本意
        cut.ratioCalcComm = ratioCalcDiag;
        cut.estTimeUsK1 = estK1Diag;
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
        cut.ratioCalcComm = ratioCalcDiag;
        cut.estTimeUsK1 = estK1Diag;
        return cut;
    }
    cut = makeCut(bestHead);
    cut.chosenK = bestHead / waveM;
    cut.estTimeUs = bestTime;
    cut.ratioCalcComm = ratioCalcDiag;
    cut.estTimeUsK1 = estK1Diag;
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
        const double mergedEst = pipelineEst(cut.headCnt - 1UL, cut.headSize, cut.headSize + cut.tailSize);
        if (mergedEst <= (cut.estTimeUs * (1.0 + in.cfg.tailMergeMarginRatio))) {
            cut.headCnt -= 1UL;
            cut.tailSize += cut.headSize; // merged round = head + tail, still prefix-aligned
            cut.rounds = cut.headCnt + 1UL;
            cut.mergedTail = true;
            cut.estTimeUs = mergedEst;
        }
    }
    return cut;
}

// ====== AllToAll→MX 量化 Matmul 的 URMA 预制（wave-packing 分支的算子无关部分;
//      所有"通信前置 AllToAll + MX 量化计算"的 URMA 算子共用, 当前消费方
//      allto_all_matmul_v2; 输出侧 AllToAll(quant_matmul_all_to_all 系)流量方向
//      相反, 不适用本预制）======
struct AllToAllMxUrmaInput {
    // Per-rank problem shape (m is per-rank M, ka is per-rank K).
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t ka = 0;
    uint32_t rankDim = 1;
    bool isFp4 = false;
    uint32_t precisionMode = 0; // 0 REMOTE / 1 DEFERRED_SYNC / 2 GM 累加, 决定 effMac 分档

    // Matmul tiling result from QuantMatmulTilingSwat.
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t usedCoreNum = 1;

    // Calibrated models chosen by the op host (算子侧标定头查表填充).
    CommLinearModel comm;
    MatmulTheoryModel mm;
    FitBalanceConfig cfg;
};

struct AllToAllMxUrmaResult {
    FitBalanceCut cut;
    uint64_t dataNonSplitAxis = 0;  // bytes per row on the data channel (fp4 packs 2 elements per byte)
    uint64_t scaleNonSplitAxis = 0; // bytes per row on the scale channel
};

/*!
 * \brief Per-row bytes shipped by one AllToAll round (data + scale channels merged).
 *
 * MX quantised layouts (ka is per-rank K, ka % 64 == 0 is enforced by the op):
 *   data : fp8 -> ka * 1B ; fp4 -> ka / 2B (two elements packed per byte)
 *   scale: ceil(ka / 64) groups * 2B
 */
inline double MxPerRowBytes(uint64_t ka, bool isFp4)
{
    const double dataBytes = isFp4 ? (static_cast<double>(ka) / 2.0) : static_cast<double>(ka);
    const double scaleBytes = static_cast<double>((ka + 63UL) / 64UL) * 2.0;
    return dataBytes + scaleBytes;
}

// AllToAll 语义的流量折叠与调度: fanout = rankDim-1, kTotal = rankDim×ka,
// 单轮单 rank 发送量 = fanout × mSpan × 每行字节数
inline AllToAllMxUrmaResult ScheduleAllToAllMxUrma(const AllToAllMxUrmaInput &in)
{
    AllToAllMxUrmaResult res;
    FitBalanceScheduleInput core;
    core.m = in.m;
    core.n = in.n;
    core.kTotal = static_cast<uint64_t>(in.rankDim) * in.ka;
    core.baseM = in.baseM;
    core.baseN = in.baseN;
    core.usedCoreNum = in.usedCoreNum;
    core.commFanout = (in.rankDim > 1U) ? (static_cast<uint64_t>(in.rankDim) - 1UL) : 1UL;
    core.perRowBytes = MxPerRowBytes(in.ka, in.isFp4);
    core.comm = in.comm;
    core.mm = in.mm;
    core.mm.coreNum = in.usedCoreNum; // compute runs on the cores the matmul tiling picked
    core.cfg = in.cfg;
    res.cut = FitBalanceSchedule(core);
    res.dataNonSplitAxis = in.isFp4 ? (in.ka / 2UL) : in.ka;
    res.scaleNonSplitAxis = in.ka / 32UL; // MX group size 32: scale 字节数 = ka/64组×2B
    return res;
}

// 双通道 CommTilingData 回填(data/scale 共享切分轴, 非切分轴各自)
inline void FillMxDualChannelCommTiling(const AllToAllMxUrmaResult &res, CommTilingData &dataTiling,
                                        CommTilingData &scaleTiling)
{
    dataTiling.splitAxisTileSize = res.cut.headSize;
    dataTiling.splitAxisTileCnt = res.cut.headCnt;
    dataTiling.splitAxisTailSize = res.cut.tailSize;
    dataTiling.splitAxisTailCnt = res.cut.tailCnt;
    dataTiling.nonSplitAxisSize = res.dataNonSplitAxis;
    scaleTiling = dataTiling;
    scaleTiling.nonSplitAxisSize = res.scaleNonSplitAxis;
}

// ============================================================================
// 四、公共钩子片段（算子特有钩子中可复用的公共逻辑, P2 算子迁移时优先对齐;
//     2026-09-15 五算子钩子逐行对比提取, 覆盖率注明各行）
// ============================================================================

// 短长块等长归一（5/5 算子 Adjust 尾部逐字一致）
inline void MergeEqualTiles(FormPartition &tilingM)
{
    if (tilingM.cutRes.shortTileLen == tilingM.cutRes.longTileLen) {
        tilingM.cutRes.shortTileLen = 0U;
        tilingM.cutRes.numShortTile = 0U;
        tilingM.cutRes.numLongTile++;
    }
}

// 线性度好的 shape（k×n 较大, 通信拟合更可信; 4/5 算子一致, agmm 传字面 true,
// allto_all_matmul 自有离散切分不用）
inline bool IsGoodLinearityShape(const MatmulParameters &mmInfo)
{
    return mmInfo.kValue * mmInfo.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE;
}

// 双向配平长块（3/5 算子一致: rs_v2 / all_reduce / matmul_allto_all）:
//   短块后置(计算 bound) → 长块 = 反解通信时长(掩盖短块计算);
//   短块前置(通信 bound) → 长块 = 反解计算时长(掩盖短块通信)。
// agmm 不用(其长块掩盖的是 frontMMTime 本地首轮计算); allto_all_matmul 自有模型。
inline void BalanceLongTileLen(FitBasedBalanceCtx &ctx, uint64_t rankTileNum)
{
    if (ctx.tilingM.cutRes.shortTileAtBack) {
        const double targetTime = ctx.matmulPerf.MatmulTime(ctx.tilingM.cutRes.shortTileLen, rankTileNum);
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(targetTime);
    } else {
        const double targetTime = ctx.commPerf.CommTime(ctx.tilingM.cutRes.shortTileLen);
        ctx.tilingM.cutRes.longTileLen = ctx.matmulPerf.InverseMatmulTime(targetTime, rankTileNum);
    }
}

// ---- 以下为多算子共有(非全算子)片段: 算子钩子按需调用即为"开关"(调=走公共逻辑,
//      不调=写算子自己的), 不设引擎级 bool 开关——片段与算子逻辑在钩子内交错,
//      引擎级开关无法表达交错顺序且会重演配置汤) ----

// Estimate 骨干（4/5: agmm/rs_v2/all_reduce/matmul_allto_all 一致）:
// FindCubeUtil + GetMatmulGradient + 总量估计 + 通算比; shortTileAtBack 判定不在此
// (各算子语义相反: rs_v2 系 mm>=comm→后置, allto_all_matmul 是 comm>mm→后置, agmm 恒后置)。
// mmRankTileMul: agmm/rs_v2 传 rankDim, all_reduce/matmul_allto_all 传 1;
// mmTimeScale: matmul_allto_all 的 KC_QUANT 模式传 1.43(其余 1.0);
// allto_all_matmul 自有拟合模型不适用本片段。
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalTpTime = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(FitBasedBalanceCtx &ctx, uint64_t mmRankTileMul, double mmTimeScale = 1.0)
{
    ctx.matmulPerf.FindCubeUtil(ctx.rankTileNum);
    ctx.matmulPerf.GetMatmulGradient();
    EstimateTimesResult r;
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.mmInfo.mValue, mmRankTileMul) * mmTimeScale;
    r.totalTpTime = ctx.commPerf.CommTime(ctx.mmInfo.mValue);
    ctx.ratioCalcComm = (std::max(r.totalTpTime, r.totalMatmulTime) / std::min(r.totalTpTime, r.totalMatmulTime));
    return r;
}

// L2 占用判定（3/3 逐字一致: rs_v2/all_reduce/matmul_allto_all, 含 128MB 同值常量）
inline bool ExceedsL2Cache(const MatmulParameters &mmInfo)
{
    constexpr uint64_t L2_CACHE_SIZE = 128UL * ONE_MBYTE;
    const uint64_t l2UseSize = mmInfo.mValue * mmInfo.kValue * mmInfo.inMatrixADtypeSize +
                               mmInfo.kValue * mmInfo.nValue * mmInfo.inMatrixBDtypeSize;
    return l2UseSize > L2_CACHE_SIZE;
}

// L2 超限鼓励大块: minLen×2（3/5; rs_v2 附加 m>minLen×4 守卫 → mGuardMul=4,
// all_reduce/matmul_allto_all 无守卫 → mGuardMul=0; agmm/allto_all_matmul 不用）
inline void ApplyL2MinLenBoost(FormPartition &tilingM, const MatmulParameters &mmInfo, bool exceedsL2,
                               uint64_t mGuardMul = 0UL)
{
    const bool guardOk = (mGuardMul == 0UL) || (mmInfo.mValue > tilingM.GetMinLen() * mGuardMul);
    if (exceedsL2 && guardOk) {
        tilingM.SetMinLenByMax(tilingM.GetMinLen() * TWO);
    }
}

// 小 shape 整 M 单块装下: minLen 提到 m（2/2 逐字一致: rs_v2/matmul_allto_all;
// 判据 m×n ≤ 2×coreNum×baseM×baseN; GetBaseM/N 为非 const 成员故 ctx 取非常量引用）
inline void ApplySmallShapeMinLen(FormPartition &tilingM, FitBasedBalanceCtx &ctx)
{
    if (ctx.mmInfo.mValue * ctx.mmInfo.nValue <=
        ctx.coreNum * ctx.matmulPerf.GetBaseM() * ctx.matmulPerf.GetBaseN() * TWO) {
        tilingM.SetMinLenByMax(ctx.mmInfo.mValue);
    }
}

} // namespace fit_balance
} // namespace apace

#endif // __APACE_FIT_BALANCE_TILING_H__
