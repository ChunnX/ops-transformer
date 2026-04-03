/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file allto_all_matmul_fit_balance_tiling.cpp
 * \brief Arch35 formulaic tiling implementation for AlltoAllMatmul
 */
#include <iostream>
#include "mc2_log.h"
#include "./allto_all_matmul_fit_balance_tiling.h"

namespace MC2Tiling {
    namespace {
        constexpr static double MM_EXPANSION_TIME = 40;
        constexpr static double COMM_EXPANSION_TIME = 40;
    }

void AlltoAllMatmulFitBalanceTiling::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(1);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    OP_LOGI("alltoallmatmul-test", "mmInfo_.mValue:%lu", mmInfo_.mValue);
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));

    // AlltoAllMatmul is comm-first pattern, short tiles at back for compute-bound balancing
    tilingM_.cutRes.shortTileAtBack = true;

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.kValue * (rankDim_ - 1) / ONE_MBYTE * commPerf_.GetCommDTypeSize();
}

void AlltoAllMatmulFitBalanceTiling::SetShortTileLen()
{
    // Encourage split more if the comm and calc is balanced and the cost of cutLen is sufficiently high
    bool isCalcCommBalance = ratioCalcComm_ < CALC_COMM_RATIO;
    uint64_t cutLen = tilingM_.GetAlignLength() / TWO;
    double mmCost = matmulPerf_.MatmulTime(cutLen, rankDim_);
    double commCost = commPerf_.CommTime(cutLen);
    if (isCalcCommBalance && (mmCost > MM_EXPANSION_TIME) && (commCost > COMM_EXPANSION_TIME)) {
        tilingM_.SetAlignLength(cutLen);
        tilingM_.SetMinLenByMin(cutLen);
    }

    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}

void AlltoAllMatmulFitBalanceTiling::SetLongTileLen()
{
    // For comm-first pattern: long tile length is determined by inverse comm time of single-rank matmul
    double singleRankMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
    tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(singleRankMatmulTime);
}

void AlltoAllMatmulFitBalanceTiling::AdjustLongShortTileLen()
{
    tilingM_.FitTileLengthDiscrete(false, true, true);
    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}
}