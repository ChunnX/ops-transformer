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
 * \file comm_perf_model.h
 * \brief HCCL 通信时延模型（旧公式化/拟合配平 tiling 家族的 L1 模型层, 通信引擎分支）
 *
 * 提取来源（逻辑逐行忠实, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/hccl_performance.h / .cpp          (基类模型 + FITTING_PARAMETER_MAP)
 *   - mc2/common/op_host/op_tiling/hccl_performance_arch35.h / .cpp   (arch35 模型 + ARCH35_FITTING_PARAMETER_MAP)
 *
 * 与旧实现的差异（仅以下三点, 其余逐行一致）:
 *   1. 拟合表不再以静态 std::map 内置, 改为 CommFitTable 注入（行数据由算子侧标定头提供,
 *      查表语义 = 精确 key 匹配, 与 map::find 等价）;
 *   2. 移除 OP_LOGW/OP_LOGD 日志（apace 层无日志依赖, 命中/回退信息由算子侧胶水打印）;
 *   3. 未移植旧头中的死声明 FindStepSize()（全仓无定义无调用）。
 *
 * 分支模型（迁移方案 §2.2）: 本模型承载 AICPU/CCU HCCL 分支的拟合数据;
 * URMA 分支（v2 的线性 kx+b）P5 吸收时扩展表 schema, 不在本文件。
 */

#ifndef __APACE_LEGACY_COMM_PERF_MODEL_H__
#define __APACE_LEGACY_COMM_PERF_MODEL_H__

#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "legacy_tiling_types.h"

namespace apace {
namespace legacy {

// ====== 以下常量来源: hccl_performance.h ======
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

// ====== arch35 模型（来源: hccl_performance_arch35.h / .cpp） ======
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

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_COMM_PERF_MODEL_H__
