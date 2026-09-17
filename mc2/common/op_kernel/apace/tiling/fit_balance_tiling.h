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
 * 当前仅含自由长度分支: HCCL AICPU/CCU 引擎, 每轮开销大 → 反解最优轮长压轮数。
 * 消费者为旧通算融合算子的 fit-balance 路径（950 标卡 4P/8P; 引擎仅支持 950, 后续芯片扩展时更新此注释）。
 *
 * 历史存档: URMA wave-packing 拓展分支（v2 aiv_urma 的 estTime k 搜索方案）曾在此文件,
 * 2026-09-15 评审裁定废弃——与核利用率优先切分大部分结果相同, 少量差异定位为
 * 核利用率优先的实现问题(修复走独立事项)。相关标定方法论与实测数据仍有效,
 * 存档于 tiling-plan/ALLTO_ALL_MATMUL_V2_FIT_BALANCE_*。
 *
 * 布局: 零、引擎自有词汇(TilingCommEngine/顺序/阈值/标定大结构体 FitBalanceCalibration) /
 *       二、切分与输出层(FormPartition——复用旧家族全局类) /
 *       三、求解层(自由长度求解器, ctx+键控分支) / 四、公共逻辑片段
 *
 * 命名空间: global(2026-09-17 裁定——apace/tiling 与 mmtiling 一致; 公共词汇复用
 * 旧家族全局头, 本文件只定义引擎自有符号)。
 * 零特有代码架构(2026-09-17 §4.24 裁定): 钩子机制已废除, 算子差异按
 * (通信原语 KernelType, 通算顺序 CommComputeOrder, 通信引擎 TilingCommEngine, dtype)
 * 键控分支内化于引擎; 阈值经 FitBalanceCalibration.tunables(FitTunables) 注入。
 * 算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用, 无任何算子特有代码。
 *
 * 统一引擎分派（三引擎接入结构完全一致, 见 CommEngine/ParseCommEngine）:
 *   同时支持多通信模式的算子按同一模式接入——CCU/AICPU 走自由长度分支（各引擎拟合表）;
 *   AIV_URMA 当前无求解分支(原 wave-packing 已废弃), 枚举值保留供未来方案接入;
 *   每引擎一组标定常量 + 一张表 + 一个分派点, 无任何引擎特殊路径。
 *
 * 等价性基线: 自由长度分支 2026-09-14 P0 提取经 162 用例 bit 级验证
 * (与老类逐字段比对), 消费方 agmm_v2 双包构建通过。
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
#include "op_host/op_tiling/formulaic_tiling_datatype.h" // 全局公共词汇(枚举/结构体/常量)+ CutResult
#include "op_host/op_tiling/hccl_formulaic_tiling.h" // 全局 FormPartition/MAX_TILE_CNT/CALC_COMM_RATIO(复用, 勿再定义副本)
#include "op_host/op_tiling/matmul_formulaic_tiling.h" // mc2tiling::TilingArgs(同上)

// ============================================================================
// 零、公共类型与常量（两分支共享词汇: CutResult/TilingArgs/枚举等）
// ============================================================================
// 公共词汇复用旧家族全局类型(formulaic_tiling_datatype.h 的枚举/结构体/常量,
// hccl_formulaic_tiling.h 的 FormPartition/MAX_TILE_CNT/CALC_COMM_RATIO 等),
// 本文件不再持有副本(2026-09-17 namespace 去 mc2tiling 化裁定: apace/tiling 与
// mmtiling 一致用 global, 词汇表唯一真值在旧家族头); 以下仅引擎自有词汇。

// 调度入参: 复用全局 mc2tiling::TilingArgs(字段完全一致)
using TilingArgs = ::mc2tiling::TilingArgs;

// ====== 通信引擎枚举与解析(三引擎统一分派; 名字带 Tiling 前缀避免与 HCCL SDK 全局 CommEngine typedef 冲突)======
enum class TilingCommEngine {
    AICPU,
    CCU,
    AIV_URMA,
};

// comm_mode 属性解析: "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含空指针/默认"ai_cpu")→AICPU
inline TilingCommEngine ParseCommEngine(const char *commMode)
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

// ====== 通算顺序枚举(键控维度: 同一原语可有相反顺序, 如 matmul_allto_all=mm→A2A /
//       allto_all_matmul=A2A→mm; 顺序不可由原语推导, 必须显式传入) ======
enum class CommComputeOrder {
    COMPUTE_FIRST, // 计算在前通信在后(输出侧通信: RS/AR/matmul_allto_all)
    COMM_FIRST,    // 通信在前计算在后(输入侧通信: AG/allto_all_matmul)
};

// ====== 键控分支的可调阈值(载体: FitBalanceCalibration.tunables, 缺省=现网值) ======
// 机制随引擎键控分支下沉, 阈值留标定——同键算子将来分叉只改各自 CALIBRATION(§4.22 裁定)
struct FitTunables {
    // E9(AG 键): 短块减半鼓励门——通算平衡且半块双耗时均超阈, 才减半对齐长度(鼓励多切)
    double halveEncourageMmUs = 40;   // 半块 matmul 耗时下限 (us)
    double halveEncourageCommUs = 40; // 半块通信耗时下限 (us)
    // E10(AR 键, P2-MMAR 接线): 切分开销绝对值反解 minLen 的双耗时下限
    double minCutCommUs = 100; // 通信耗时下限 (us)
    double minCutMmUs = 60;    // matmul 耗时下限 (us)
};

// fp4 子字节建模分母: 1 字节打包 2 个 fp4 元素(D6, dtype 键控)
constexpr uint64_t FP4_PACK_RATIO = 2;

// ============================================================================
// 一、模型层
// ============================================================================
// ---- 1.1 HCCL 通信模型与拟合表（自由长度分支, 950 专用）----
// 由旧 HCCLPerformanceModel + HCCLPerformanceArch35 合并精简(2026-09-15, 类名已去 arch35):
// 本引擎按设计仅在 950 标卡 4P/8P 生效, 旧芯片分支(310P 环网/910_93
// 双环/基表两段式查表——其结果恒被 950 查表覆盖)经 grep 证实 5 个消费
// 算子零调用后剥离; 老芯片的真实需求由原位保留的 B 家族(mc2/common/op_host/
// op_tiling)满足。活跃 API 面(键控分支/P2 算子实测): CommTime/InverseCommTime/
// SetCommShapeLen/SetCommDTypeSize/GetCommDTypeSize/SetCommDtypeSizeExpansionFraction/
// commTimeFactor_/GetRankTileNum(InitCtx 内部)。

//      此处类型由 InitCtx 内部适配构造并按值持有, 勿在算子侧直接引用）======
struct CommFitRow {
    uint32_t rankDim; // 匹配字段: 卡数
    HCCLFittingParameters params;
};

struct CommFitTable {
    static constexpr size_t MAX_ROWS = 4;
    CommFitRow rows[MAX_ROWS] = {};
    size_t rowCount = 0;
    HCCLFittingParameters defaultParams = {}; // key 未命中时的回退行 + rankDim 折算因子缩放
};

// 组装纯线性拟合行（算子侧标定头用: 两段边界置 0, 任意 x>0 走线性分支 y = k·x + b）
inline HCCLFittingParameters MakeLinearCommFit(double slopeUsPerMb, double interceptUs)
{
    return {0, 0, slopeUsPerMb, interceptUs, 0, 0, 0, 0, 0, 0, 0, 0};
}

class HCCLCommModel {
public:
    HCCLFittingParameters commEstimatePar_;
    HCCLInfo commTypeInfo_;
    double commTimeFactor_ = 1.0; // allreduce time is 2.0x allgather or reducescatter
    uint64_t lookUpTileNum_ = 1;  // 数据拟合查表参数

    // Constructor（soc 恒 950）
    HCCLCommModel(uint32_t inputRankDim, KernelType inputKernelType, const CommFitTable &fitTable,
                  TopoType topoType = TopoType::STANDARD_CARD)
        : fitTable_(fitTable)
    {
        commTypeInfo_.kernelType = inputKernelType; // 区分哪个MC2算子
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_COMM_RANKDIM); // 并行维度最小为2
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        InitParametersForFullMesh();
        commTypeInfo_.maxStepSize = 1;
        InitCommParameters();
        commTypeInfo_.topoType = topoType;
        GetCommParameters();
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
    uint64_t rankDim() const
    {
        return commTypeInfo_.rankDim;
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
    void InitCommParameters()
    {
        if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
            (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL)) {
            lookUpTileNum_ = commTypeInfo_.rankDim;
            commTimeFactor_ = 1;
        }
    }
    // 按 rankDim 匹配标定行(soc/kernelType 对算子实例为常量, 已在构造参数中区分)
    void GetCommParameters()
    {
        const HCCLFittingParameters *hit = nullptr;
        for (size_t i = 0; i < fitTable_.rowCount; ++i) {
            if (fitTable_.rows[i].rankDim == commTypeInfo_.rankDim) {
                hit = &fitTable_.rows[i].params;
                break;
            }
        }
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
    CommFitTable fitTable_; // 拟合表按值持有(自包含, 由 InitCtx 从大结构体适配)
};

// ---- 1.2 matmul 模型（HCCL 算力/梯度模型, 自由长度分支专用, 950）----
// 由旧 MatmulPerformanceModel + MatmulPerformanceArch35 合并精简(2026-09-15, 类名已去 arch35),
// 剥离 310P/910_B4 分支与 4 参 FindCubeUtil(含其独占的 K 对齐修正链)——
// 5 个消费算子经 grep 证实仅用 1 参 FindCubeUtil/FindCubeUtilByL2Usage。

// ====== 模型参数表（引擎内部结构——算子侧一律使用 FitBalanceCalibration 大结构体,
//      此处类型由 InitCtx 内部适配构造并按值持有, 勿在算子侧直接引用）======
struct CubeCalcRow {
    MatmulCalcType calcType;
    bool isFp4;
    uint64_t aBytes;
    uint64_t bBytes;
    uint64_t cBytes;
    double computesPerCycle;
};

struct L2ParamRow {
    MatmulCalcType calcType;
    bool isFp4;
    uint64_t aBytes;
    uint64_t bBytes;
    uint64_t cBytes;
    L2CacheEstimateParameters params;
};

struct MatmulPerfTable {
    static constexpr size_t MAX_CUBE_ROWS = 4;
    static constexpr size_t MAX_L2_ROWS = 4;
    static constexpr size_t MAX_BASE_ROWS = 2;
    CubeCalcRow cubeCalcRows[MAX_CUBE_ROWS] = {};
    size_t cubeCalcCount = 0;
    double defaultComputesPerCycle = 4096.0; // key 未命中回退
    L2ParamRow l2Rows[MAX_L2_ROWS] = {};
    size_t l2Count = 0;
    L2CacheEstimateParameters defaultL2Params = {128, 192, 0.85, 0.75, 0.65};
    TilingBestBaseBlock bestBaseBlock = {128, 256, 64}; // matmul 最优基本块(单值)
};

class MatmulPerfModel {
public:
    MatmulParameters mmShapeInfo_;
    MatmulCalcType calcType_; // Distinguish the non-quantization, full-quantization and fake-quantization
    bool isFp4_ = false;      // fp4 输入(geAType==DT_FLOAT4_E2M1), 影响 CubeCalc/L2 查表匹配
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;

    // Constructor（soc 恒 950, 频率 1689.6 cycle/us）
    MatmulPerfModel(const TilingArgs &args, const MatmulPerfTable &perfTable)
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
        // 获取后续计算用的基本块(先默认, 再按标定数据 baseBlock 覆盖)
        mmShapeInfo_.baseM = mc2tiling::BASE_BLOCK_M;
        mmShapeInfo_.baseN = mc2tiling::BASE_BLOCK_N;
        mmShapeInfo_.baseK = mc2tiling::BASE_BLOCK_K;
        mmShapeInfo_.batchSize = 1UL; // 初始值
        isFp4_ = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
        SetCalcType(args);
        mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC_NPUARCH_3510; // 950 频率
        GetMachineParameters();
        mmShapeInfo_.baseM = perfTable_.bestBaseBlock.baseM;
        mmShapeInfo_.baseN = perfTable_.bestBaseBlock.baseN;
        mmShapeInfo_.baseK = perfTable_.bestBaseBlock.baseK;
    };

    // 950 量化语义: fp4/fp8/HIF8 或 A8W8 量化, 其余按 FP16 建模
    // fp4 经 geAType 识别(matmul_tiling::DataType 无 fp4 枚举, aType 对 fp4 为 DT_MAX)
    // （旧实现的 310P A16W8 与 910_B INT8 排除分支随旧芯片逻辑剥离）
    void SetCalcType(const TilingArgs &args)
    {
        if ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
            (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2) ||
            (args.aType == matmul_tiling::DataType::DT_INT8) || (args.geAType == ge::DataType::DT_FLOAT4_E2M1)) {
            calcType_ = MatmulCalcType::QUANT;
        }
    }

    // 按 (calcType, A/B/C 字节数) 匹配标定行(soc 对算子实例为常量)
    template <typename RowT>
    bool MatchesDtype(const RowT &row) const
    {
        return (row.calcType == calcType_) && (row.isFp4 == isFp4_) &&
               (row.aBytes == mmShapeInfo_.inMatrixADtypeSize) && (row.bBytes == mmShapeInfo_.inMatrixBDtypeSize) &&
               (row.cBytes == mmShapeInfo_.outMatrixCDtypeSize);
    }
    void GetMachineParameters()
    {
        mmShapeInfo_.computesPerCycle = perfTable_.defaultComputesPerCycle;
        for (size_t i = 0; i < perfTable_.cubeCalcCount; ++i) {
            if (MatchesDtype(perfTable_.cubeCalcRows[i])) {
                mmShapeInfo_.computesPerCycle = perfTable_.cubeCalcRows[i].computesPerCycle;
                return;
            }
        }
    }
    // L2 档位利用率 + 950 调和平均公式(950 专用模型, 调和分支无条件执行)
    double FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t *maxTileLen = nullptr)
    {
        L2CacheEstimateParameters l2EstimatePar = perfTable_.defaultL2Params;
        for (size_t i = 0; i < perfTable_.l2Count; ++i) {
            if (MatchesDtype(perfTable_.l2Rows[i])) {
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
    // 单参版本(消费算子钩子唯一使用的入口)
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
    MatmulPerfTable perfTable_; // 模型参数表按值持有(自包含, 由 InitCtx 从大结构体适配)
};

// ============================================================================
// 二、切分与输出层
// ============================================================================
// FormPartition 复用旧家族全局类(op_host/op_tiling/hccl_formulaic_tiling.h):
// A 家族使用的方法(离散切分路径)与旧实现逐行一致(引擎原副本仅少两条日志),
// B 家族专用方法(连续切分)随旧类一并可见但 A 家族零调用。

// ============================================================================
// 三、求解层（两分支求解器）
// ============================================================================
// ---- 3.1 自由长度求解器（ctx + 键控分支, HCCL AICPU/CCU; 钩子机制已于 §4.24 废除）----
// ============================================================================

// 调度上下文: 引擎三件套 + 键控维度 + 过程量(刻意不含静态标定数据, 在 FitBalanceCalibration)
struct FitBasedBalanceCtx {
    MatmulPerfModel matmulPerf;
    HCCLCommModel commPerf;
    FormPartition tilingM;
    KernelType kernelType;      // 键控维度: 通信原语
    CommComputeOrder order;     // 键控维度: 通算顺序
    FitTunables tunables;       // 键控分支阈值(自 Calibration.tunables 拷贝)
    uint64_t rankTileNum = 1;   // 派生: commPerf.GetRankTileNum()(通信模型内的 rank 份数)
    double ratioCalcComm = 1.0; // 过程量: Estimate 骨干设置的通算比(键控分支消费)
    double frontMMTime = 0; // 过程量(AG 键): 本地首轮计算耗时(本卡 A 块已在手, 首块无通信依赖)
};
// ============================================================================
// 客户标定数据总集（算子侧唯一需要接触的标定类型, 2026-09-15 评审裁定:
// 单一大结构体替代"引擎侧一堆小结构体 + 算子侧一堆具名 constexpr"）
//
// 用法（算子侧标定头, 填一个实例传给 InitCtx 即可）:
//   static const FitBalanceCalibration CALIBRATION = {
//       /* aicpu */  { /* rows */ { {"4_1_0_4", 斜率, 截距} } },
//       /* ccu */    { /* rows */ { {"4_1_0_4", 斜率, 截距} } },
//       /* matmul */ { /* cubeCalc */ { {"4_1_1_1_2", 算力} }, ... },
//   };
//
// 填写规则:
//   - 未填的行/字段自动取默认值(零值或 NSDMI 平台默认), 查表 miss 走回退路径;
//   - 通信行按 rankDim 匹配, matmul 行按 (calcType, A/B/C 字节数) 匹配
//     (soc/kernelType 对算子实例为常量, 不进入匹配); 行按前缀填, 0 值行=未填;
//   - platformDefault/default* 为平台级默认(三段式, 通常勿动), 已给默认值可整段省略;
//   - 通信已标定行为纯线性拟合(客户标定产出即 斜率+截距 两个数), 引擎内部转三段式。
// ============================================================================

struct FitBalanceCalibration {
    static constexpr size_t MAX_COMM_FIT_ROWS = 4;   // 已标定通信形态数上限
    static constexpr size_t MAX_CUBE_CALC_ROWS = 4;  // matmul 算力档位数上限
    static constexpr size_t MAX_L2_PARAM_ROWS = 4;   // L2 档位数上限
    static constexpr size_t MAX_BASE_BLOCK_ROWS = 2; // 最优基本块档位数上限

    // —— 通信拟合（单引擎）——
    struct CommFit {
        struct Row {
            uint32_t rankDim;    // 卡数(4=标卡4P, 8=8P, ...)
            double linearSlope;  // 线性段斜率 (us/MB)
            double linearOffset; // 线性段截距 (us), 含每轮同步固定开销
        };
        Row rows[MAX_COMM_FIT_ROWS] = {}; // 已标定行, rankDim=0 表示未填(跳过)
        // 平台默认回退行(未标定形态走此行 + rankDim 折算缩放; 三段式 12 参数, 勿动)
        HCCLFittingParameters platformDefault = {0.032, 1.371,  2.59568, 5.8887, 0, 2.86702,
                                                 5.608, 5.6997, 9.448,   0,      0, 0};
    };
    CommFit aicpu; // AICPU 引擎(comm_mode != "ccu")
    CommFit ccu;   // CCU 引擎(comm_mode == "ccu")

    // —— matmul 模型（字段序: 用户常改字段在前, 平台默认在后——聚合初始化可省尾部,
    //    省略的尾部字段自动取 NSDMI 默认值）——
    struct Matmul {
        struct CubeCalc {
            MatmulCalcType calcType;
            bool isFp4;              // FP16 / QUANT / ANTI_QUANT
            uint64_t aBytes;         // 输入 A 每元素字节数
            uint64_t bBytes;         // 输入 B 每元素字节数
            uint64_t cBytes;         // 输出 C 每元素字节数
            double computesPerCycle; // 有效算力 (MACs/cycle, 含 cube 利用率)
        };
        struct L2Param {
            MatmulCalcType calcType;
            bool isFp4;
            uint64_t aBytes;
            uint64_t bBytes;
            uint64_t cBytes;
            L2CacheEstimateParameters params;
        };
        CubeCalc cubeCalc[MAX_CUBE_CALC_ROWS] = {};
        L2Param l2Param[MAX_L2_PARAM_ROWS] = {};
        TilingBestBaseBlock baseBlock = {128, 256, 64};                          // matmul 最优基本块(单值)
        double defaultComputesPerCycle = 4096.0;                                 // 算力档回退(通常勿动)
        L2CacheEstimateParameters defaultL2Param = {128, 192, 0.85, 0.75, 0.65}; // L2 档回退(通常勿动)
    };
    Matmul matmul;
    FitTunables tunables; // 键控分支阈值(缺省=现网值, 通常勿动; 尾部可整段省略)
};

// 统一入口: 算子侧只传大结构体标定数据 + 键控枚举(原语/顺序/引擎; soc 恒 950)
inline FitBasedBalanceCtx InitCtx(const TilingArgs &args, KernelType kernelType, CommComputeOrder order,
                                  TilingCommEngine commEngine, const FitBalanceCalibration &calib,
                                  TopoType topoType = TopoType::STANDARD_CARD)
{
    // 引擎分路 + 大结构体 → 引擎内部表适配(线性行转三段式, 按 null key 截断)
    const FitBalanceCalibration::CommFit &commFit = (commEngine == TilingCommEngine::CCU) ? calib.ccu : calib.aicpu;
    CommFitTable commTable;
    commTable.defaultParams = commFit.platformDefault;
    commTable.rowCount = 0;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_COMM_FIT_ROWS; ++i) {
        if (commFit.rows[i].rankDim == 0) {
            break;
        }
        commTable.rows[i] = {commFit.rows[i].rankDim,
                             MakeLinearCommFit(commFit.rows[i].linearSlope, commFit.rows[i].linearOffset)};
        commTable.rowCount = i + 1;
    }
    MatmulPerfTable mmTable;
    mmTable.defaultComputesPerCycle = calib.matmul.defaultComputesPerCycle;
    mmTable.defaultL2Params = calib.matmul.defaultL2Param;
    mmTable.bestBaseBlock = calib.matmul.baseBlock;
    mmTable.cubeCalcCount = 0;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_CUBE_CALC_ROWS; ++i) {
        if (calib.matmul.cubeCalc[i].aBytes == 0) {
            break;
        }
        mmTable.cubeCalcRows[i] = {calib.matmul.cubeCalc[i].calcType, calib.matmul.cubeCalc[i].isFp4,
                                   calib.matmul.cubeCalc[i].aBytes,   calib.matmul.cubeCalc[i].bBytes,
                                   calib.matmul.cubeCalc[i].cBytes,   calib.matmul.cubeCalc[i].computesPerCycle};
        mmTable.cubeCalcCount = i + 1;
    }
    mmTable.l2Count = 0;
    for (size_t i = 0; i < FitBalanceCalibration::MAX_L2_PARAM_ROWS; ++i) {
        if (calib.matmul.l2Param[i].aBytes == 0) {
            break;
        }
        mmTable.l2Rows[i] = {calib.matmul.l2Param[i].calcType, calib.matmul.l2Param[i].isFp4,
                             calib.matmul.l2Param[i].aBytes,   calib.matmul.l2Param[i].bBytes,
                             calib.matmul.l2Param[i].cBytes,   calib.matmul.l2Param[i].params};
        mmTable.l2Count = i + 1;
    }
    mmTable.bestBaseBlock = calib.matmul.baseBlock;
    FitBasedBalanceCtx ctx{MatmulPerfModel(args, mmTable),
                           HCCLCommModel(args.rankDim, kernelType, commTable, topoType),
                           FormPartition(args),
                           kernelType,
                           order,
                           calib.tunables,
                           1,
                           1.0};
    ctx.tilingM.SetMaxTileCnt(MAX_TILE_CNT); // Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
    ctx.rankTileNum = ctx.commPerf.GetRankTileNum();

    // ---- 通算胶水(键控, 原散在各算子构造函数, §4.24 裁定随引擎下沉) ----
    // D4 通信矩阵另一维: AG/A2A-in 沿 K 轴聚散 → kValue; RS/AR/A2A-out 沿 N 轴 → nValue
    const bool commOnKAxis = (kernelType == KernelType::ALL_GATHER) ||
                             ((kernelType == KernelType::ALL_TO_ALL) && (order == CommComputeOrder::COMM_FIRST));
    ctx.commPerf.SetCommShapeLen(commOnKAxis ? ctx.matmulPerf.mmShapeInfo_.kValue : ctx.matmulPerf.mmShapeInfo_.nValue);
    // D5 通信承载 dtype: AG/A2A-in 传 A 侧数据, 其余传 C 侧;
    // D6 A 侧 fp4 无 0.5 字节表达, 按"1 字节打包 2 元素"建模(复用 HCCL 子字节机制)
    if (commOnKAxis) {
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
            ctx.commPerf.SetCommDTypeSize(1);
            ctx.commPerf.SetCommDtypeSizeExpansionFraction(FP4_PACK_RATIO);
            ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize = 1; // L2 占用按 2x 高估, 避免 0 退化
            ctx.matmulPerf.mmShapeInfo_.inMatrixBDtypeSize = 1;
        } else {
            ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.inMatrixADtypeSize);
        }
    } else {
        // [P2 接线] RS 的 A&&B 双 fp4 mm 侧建模变体在 RSMM 迁移时接入
        ctx.commPerf.SetCommDTypeSize(ctx.matmulPerf.mmShapeInfo_.outMatrixCDtypeSize);
    }
    // D7 离散对齐粒度初始化: baseM
    ctx.tilingM.SetMinLenByMax(ctx.matmulPerf.GetBaseM());
    ctx.tilingM.SetAlignLength(ctx.matmulPerf.GetBaseM());
    return ctx;
}

// ============================================================================
// 四、公共逻辑片段（求解器键控分支与后续算子迁移复用的公共逻辑, P2 迁移时优先对齐;
//     2026-09-15 五算子逐行对比提取, 覆盖率注明各行; §4.24 后由键控分支/P2 迁移消费）
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

// ---- 以下为多算子共有(非全算子)片段: 求解器键控分支与 P2 迁移算子按需调用
//      (调=走公共逻辑, 不调=走各自键控分支), 不设引擎级 bool 开关——片段与键控
//      分支逻辑交错, 引擎级开关无法表达交错顺序且会重演配置汤) ----

// Estimate 骨干（4/5: agmm/rs_v2/all_reduce/matmul_allto_all 一致）:
// FindCubeUtil + GetMatmulGradient + 总量估计 + 通算比; shortTileAtBack 判定不在此
// (各算子语义相反: rs_v2 系 mm>=comm→后置, allto_all_matmul 是 comm>mm→后置, agmm 恒后置)。
// mmRankMul: matmul 总量的 rank 倍数——由算子通信语义决定(AG/RS 算子=rankDim,
//   即聚合 rankDim 份后 mm 按 rankDim 倍计; AR 类=1, reduce 已含在语义中), 是语义参数;
// mmTimeScale: matmul_allto_all 的 KC_QUANT 模式传 1.43(其余 1.0);
// allto_all_matmul 自有拟合模型不适用本片段。
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalTpTime = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(FitBasedBalanceCtx &ctx, uint64_t mmRankMul, double mmTimeScale = 1.0)
{
    ctx.matmulPerf.FindCubeUtil(ctx.rankTileNum);
    ctx.matmulPerf.GetMatmulGradient();
    EstimateTimesResult r;
    r.totalMatmulTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, mmRankMul) * mmTimeScale;
    r.totalTpTime = ctx.commPerf.CommTime(ctx.matmulPerf.mmShapeInfo_.mValue);
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
    if (ctx.matmulPerf.mmShapeInfo_.mValue * ctx.matmulPerf.mmShapeInfo_.nValue <=
        ctx.matmulPerf.mmShapeInfo_.coreNum * ctx.matmulPerf.GetBaseM() * ctx.matmulPerf.GetBaseN() * TWO) {
        tilingM.SetMinLenByMax(ctx.matmulPerf.mmShapeInfo_.mValue);
    }
}

// 键控维度: isAg = (通信原语 == ALL_GATHER)。AG 语义——本卡 A 块已在手:
// 首块计算无通信依赖(E5), 长块通信掩盖的是本地首轮计算而非短块(E19), 短块恒后置(E4)
inline CutResult Schedule(FitBasedBalanceCtx &ctx)
{
    const bool isAg = (ctx.kernelType == KernelType::ALL_GATHER);

    // ---- ① Estimate: 公共骨干(E1) + E2 mmRankMul 键控 ----
    // E2: AG/RS 聚合 rankDim 份后 mm 按 rankDim 倍计; AR(reduce 已含)/A2A(分散已含) → 1
    const uint64_t mmRankMul =
        (ctx.kernelType == KernelType::ALL_GATHER || ctx.kernelType == KernelType::REDUCE_SCATTER ||
         ctx.kernelType == KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL) ?
            ctx.commPerf.rankDim() :
            1;
    const auto est = EstimateTotalTimes(ctx, mmRankMul);
    if (isAg) {
        // E5: 首轮独立计算的 L2 利用率与聚合后不同(frontUtil 修正)
        const double frontUtil = ctx.matmulPerf.FindCubeUtilByL2Usage(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        if (std::abs(frontUtil) > 1e-10) {
            ctx.frontMMTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1) / frontUtil;
        } else {
            ctx.frontMMTime = ctx.matmulPerf.MatmulTime(ctx.matmulPerf.mmShapeInfo_.mValue, 1);
        }
        ctx.tilingM.cutRes.shortTileAtBack = true; // E4: AG 恒后置
    } else if (ctx.order == CommComputeOrder::COMPUTE_FIRST) {
        // 先行段为瓶颈 → 短块后置(短块放另一端, 由长块掩盖)
        ctx.tilingM.cutRes.shortTileAtBack = (est.totalMatmulTime >= est.totalTpTime);
    } else {
        ctx.tilingM.cutRes.shortTileAtBack = (est.totalTpTime > est.totalMatmulTime);
    }

    // ---- ② SetShort: E9(AG 键) 减半鼓励门 + 短块落位 ----
    if (isAg) {
        // 通算平衡且半块双耗时均超阈(切了不亏) → 对齐长度减半鼓励多切(阈值见 FitTunables)
        const bool isCalcCommBalance = ctx.ratioCalcComm < CALC_COMM_RATIO;
        const uint64_t cutLen = ctx.tilingM.GetAlignLength() / TWO;
        const double mmCost = ctx.matmulPerf.MatmulTime(cutLen, ctx.commPerf.rankDim());
        const double commCost = ctx.commPerf.CommTime(cutLen);
        if (isCalcCommBalance && (mmCost > ctx.tunables.halveEncourageMmUs) &&
            (commCost > ctx.tunables.halveEncourageCommUs)) {
            ctx.tilingM.SetAlignLength(cutLen);
            ctx.tilingM.SetMinLenByMin(cutLen);
        }
    }
    // [P2 接线] E10(AR 键 阈值反解)/E11-E13(L2/小shape)/E14-E16(E27 等)随各算子迁移接入
    ctx.tilingM.cutRes.shortTileLen = ctx.tilingM.GetMinLen();
    ctx.tilingM.cutRes.numShortTile = 1U;

    if (ctx.tilingM.totalLen < ctx.tilingM.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        ctx.tilingM.NoCutTiling();
        return ctx.tilingM.cutRes;
    }

    // ---- ③ SetLong: E19(AG 键) 长块通信掩盖本地首轮计算 / E18 公共双向配平 ----
    if (isAg) {
        ctx.tilingM.cutRes.longTileLen = ctx.commPerf.InverseCommTime(ctx.frontMMTime);
    } else {
        BalanceLongTileLen(ctx, ctx.rankTileNum);
    }

    // ---- ④ Adjust: E24(goodLinearity: AG 恒 true) + 离散化 + E25 等长归一 ----
    const bool goodLin = isAg ? true : IsGoodLinearityShape(ctx.matmulPerf.mmShapeInfo_);
    ctx.tilingM.FitTileLengthDiscrete(false, goodLin, isAg); // localAtFront: AG 前端有本地块
    MergeEqualTiles(ctx.tilingM);
    return ctx.tilingM.cutRes;
}

#endif // __APACE_FIT_BALANCE_TILING_H__
