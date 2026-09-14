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
 * \file fit_based_balance_tiling.h
 * \brief 拟合配平 tiling 基类（L2 求解层, 自由长度分支——AICPU/CCU 每轮开销大, 反解轮长压轮数）
 *
 * 提取来源（逻辑逐行忠实, 见迁移方案 MC2_LEGACY_TILING_APACE_MIGRATION_PLAN.md）:
 *   - mc2/common/op_host/op_tiling/mc2_fit_based_balance_tiling.h / .cpp
 *
 * 与旧实现的差异:
 *   1. 拟合表/模型参数表经构造函数注入（算子侧标定头提供）;
 *   2. 移除 OPS_LOG_D 日志;
 *   3. 4 个虚函数钩子（EstimateMMCommTime/SetShortTileLen/SetLongTileLen/AdjustLongShortTileLen）
 *      保留继承式设计——算子侧派生类仅改基类限定与表注入, 迁移 diff 最小。
 *
 * 分支归属: 本类为"轮次稀缺"分支的求解器（free_length）; URMA 的 wave_packing 求解器
 * 见 apace/tiling/fit_balance_tiling.h（P5 吸收进本目录, 迁移方案 §2.2）。
 */

#ifndef __APACE_LEGACY_FIT_BASED_BALANCE_TILING_H__
#define __APACE_LEGACY_FIT_BASED_BALANCE_TILING_H__

#pragma once

#include "comm_perf_model.h"
#include "form_partition.h"
#include "legacy_tiling_types.h"
#include "matmul_perf_model.h"

namespace apace {
namespace legacy {

constexpr double CALC_COMM_RATIO = 2.0;

class Mc2FitBasedBalanceTiling {
public:
    MatmulPerformanceArch35 matmulPerf_;
    HCCLPerformanceArch35 commPerf_;
    FormPartition tilingM_;
    MatmulParameters mmInfo_;
    uint64_t rankDim_ = 2;
    uint64_t rankTileNum_ = 1;
    uint64_t coreNum_ = 32;
    double ratioCalcComm_ = 1.0;

    Mc2FitBasedBalanceTiling(const TilingArgs &args, KernelType kernelType, const CommFitTable &commTable,
                             const MatmulPerfTable &mmTable, TopoType topoType = TopoType::STANDARD_CARD,
                             SocVersion socVersion = SocVersion::SOC950)
        : matmulPerf_(args, mmTable, socVersion),
          commPerf_(args.rankDim, kernelType, commTable, socVersion, topoType),
          tilingM_(args)
    {
        rankDim_ = args.rankDim;
        mmInfo_ = matmulPerf_.mmShapeInfo_;
        tilingM_.SetMaxTileCnt(MAX_TILE_CNT); // Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
        rankTileNum_ = commPerf_.GetRankTileNum();
        coreNum_ = args.aicCoreNum;
    };

    virtual CutResult GetTiling();
    virtual void EstimateMMCommTime() {};
    virtual void SetShortTileLen() {};
    virtual void SetLongTileLen() {};
    virtual void AdjustLongShortTileLen() {};

    virtual ~Mc2FitBasedBalanceTiling() {}
};

inline CutResult Mc2FitBasedBalanceTiling::GetTiling()
{
    EstimateMMCommTime();
    SetShortTileLen();
    if (tilingM_.totalLen < tilingM_.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        tilingM_.NoCutTiling();
    } else {
        SetLongTileLen();
        AdjustLongShortTileLen();
    }
    return tilingM_.cutRes;
}

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_FIT_BASED_BALANCE_TILING_H__
