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
 * \file one_calc_one_comm.h
 * \brief 单通信域公式化 tiling 基类（L2 求解层, 公式化分支——全形态回退路径）
 *
 * 提取来源（逻辑逐行忠实, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/hccl_formulaic_tiling.h   (OneCalcOneCommBase 类 + 参数常量)
 *   - mc2/common/op_host/op_tiling/hccl_formulaic_tiling.cpp (方法实现)
 *
 * 与旧实现的差异:
 *   1. 拟合表/模型参数表经构造函数注入（算子侧标定头提供）;
 *   2. 移除 OP_LOGD 日志;
 *   3. 虚函数钩子（EstimateKernelTime/SelectTilingMethod）保留继承式设计。
 *
 * 注意: 旧实现中本类持有的是基类版模型（MatmulPerformanceModel/HCCLPerformanceModel）,
 * 与 fit-based-balance 家族持有的 Arch35 版不同——此处保持一致, 不做"顺手统一"。
 */

#ifndef __APACE_LEGACY_ONE_CALC_ONE_COMM_H__
#define __APACE_LEGACY_ONE_CALC_ONE_COMM_H__

#pragma once

#include "comm_perf_model.h"
#include "form_partition.h"
#include "legacy_tiling_types.h"
#include "matmul_perf_model.h"

namespace apace {
namespace legacy {

// ====== OneCalcOneCommBase 参数（来源: hccl_formulaic_tiling.h） ======
constexpr double commGrowRatio = 1.15;
constexpr double ALLGATHERMM_COMMTIME_FACTOR = 2;
constexpr double ALLREDUCE_COMMTIME_FACTOR = 2.0517;
constexpr double REDUCESCATTER_COMMTIME_FACTOR = 2.09136;

// 单通信域算子通信轮次切分基类，每个算子各有一个派生类
class OneCalcOneCommBase {
public:
    MatmulPerformanceModel matmulPerf_; // 计算拟合
    HCCLPerformanceModel commPerf_;     // 通信拟合
    FormPartition tilingM_;             // 切分结果
    MatmulParameters clusterInfo_;
    uint64_t rankDim_ = 2; // Tp并行维度
    uint64_t rankTileNum_ = 1;
    double ratioCalcComm_ = 1.0;
    bool noCutFlag_ = false;
    bool inferFlag_ = false; // 部分逻辑只适配了推理或者训练算子，进行区分

    // Constructor
    OneCalcOneCommBase(const TilingArgs &args, uint32_t inputRankDim, KernelType inputKernelType,
                       const CommFitTable &commTable, const MatmulPerfTable &mmTable,
                       SocVersion inputSocVersion = SocVersion::SOC910_B)
        : matmulPerf_(args, mmTable, inputSocVersion),
          commPerf_(inputRankDim, inputKernelType, commTable, inputSocVersion),
          tilingM_(args)
    {
        rankDim_ = inputRankDim;
        clusterInfo_ = matmulPerf_.mmShapeInfo_;
        tilingM_.SetMaxTileCnt(MAX_TILE_CNT); // 受限ffts队列大小，最多切16轮
    };

    virtual void EstimateKernelTime() {};
    virtual void SelectTilingMethod() {};
    double EstimateTotalMatmulTime();
    double EstimateTotalCommTime();

    // 流水配平优化
    void ShortAtEndCalcBoundBalancing();
    void ShortAtFrontCommBoundBalancing();

    // 调用函数
    void GetTiling();

    virtual ~OneCalcOneCommBase() {}
};

inline double OneCalcOneCommBase::EstimateTotalMatmulTime()
{
    uint64_t tileM = clusterInfo_.mValue;
    if (clusterInfo_.socType == SocVersion::SOC910_93 && rankDim_ > MIN_COMM_RANKDIM) { // double ring
        tileM *= rankTileNum_;
    }
    return matmulPerf_.MatmulTime(tileM, rankTileNum_);
}

inline double OneCalcOneCommBase::EstimateTotalCommTime()
{
    uint64_t totalStepSize = 1U;
    if (clusterInfo_.socType == SocVersion::SOC910_93 && rankDim_ > MIN_COMM_RANKDIM) { // double ring
        totalStepSize = rankDim_;
    }
    return commPerf_.CommTime(clusterInfo_.mValue * totalStepSize);
}

inline void OneCalcOneCommBase::ShortAtEndCalcBoundBalancing()
{
    double targetTime = matmulPerf_.MatmulTime(tilingM_.cutRes.shortTileLen, rankTileNum_);
    tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(targetTime);
}

inline void OneCalcOneCommBase::ShortAtFrontCommBoundBalancing()
{
    double targetTime = commPerf_.CommTime(tilingM_.cutRes.shortTileLen);
    tilingM_.cutRes.longTileLen = matmulPerf_.InverseMatmulTime(targetTime, rankTileNum_);
}

inline void OneCalcOneCommBase::GetTiling()
{
    matmulPerf_.FindCubeUtil(tilingM_.GetMinLen(), rankTileNum_, inferFlag_, &tilingM_.tileArgs.maxTileLen);
    matmulPerf_.GetMatmulGradient();

    EstimateKernelTime();
    SelectTilingMethod();

    // 长块短块一样长时归一
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_ONE_CALC_ONE_COMM_H__
