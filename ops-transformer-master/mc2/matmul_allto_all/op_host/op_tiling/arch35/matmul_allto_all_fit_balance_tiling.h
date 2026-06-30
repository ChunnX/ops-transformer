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
 * \file matmul_allto_all_fit_balance_tiling.h
 * \brief arch35 架构的 AllToAll Fit Balance Tiling 实现
 */
#ifndef __MATMUL_ALLTO_ALL_FIT_BALANCE_TILING_H__
#define __MATMUL_ALLTO_ALL_FIT_BALANCE_TILING_H__

#pragma once
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"
#include "../common/matmul_allto_all_util_tiling.h"
#include "../common/allto_all_formulaic_tiling.h"

namespace MC2Tiling {

class MatmulAlltoAllFitBalanceTiling : public Mc2FitBasedBalanceTiling {
public:
    explicit MatmulAlltoAllFitBalanceTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                        TopoType topoType = TopoType::STANDARD_CARD,
                                        SocVersion socVersion = SocVersion::SOC950,
                                        QuantMode quantMode = QuantMode::NON_QUANT)
        : Mc2FitBasedBalanceTiling(args, kernelType, topoType, socVersion),
          quantMode_(quantMode)
    {
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(mmInfo_.outMatrixCDtypeSize);
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
        isQuantMatmul_ = (mmInfo_.inMatrixADtypeSize == 1) && (mmInfo_.inMatrixBDtypeSize == 1);
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;

private:
    void AdjustLongShortTileLenWhenCalcBound();

    bool isLargerThanL2Cache_ = false;
    bool isQuantMatmul_ = false;
    QuantMode quantMode_;
};

inline CutResult GetArch35TilingResult(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                       SocVersion socVersion, NpuArch npuArch, QuantMode quantMode)
{
    if (mc2tiling::IsStandardCard4P(args.rankDim, npuArch)) {
        OP_LOGD("Arch35TilingResult", "Using fit balance tiling for arch35 standard card 4P");
        MatmulAlltoAllFitBalanceTiling fitBalanceTiling(args, kernelType, TopoType::STANDARD_CARD, socVersion,
                                                        quantMode);
        return fitBalanceTiling.GetTiling();
    } else if (mc2tiling::Is8P(args.rankDim, npuArch)) {
        OP_LOGD("Arch35TilingResult", "Using fit balance tiling for arch35 and 8p");
        MatmulAlltoAllFitBalanceTiling fitBalanceTiling(args, kernelType, TopoType::EIGHT_P, socVersion, quantMode);
        return fitBalanceTiling.GetTiling();
    }
    OP_LOGD("Arch35TilingResult", "Falling back to formulaic tiling");
    AlltoAllMM formulaicTiling(args, args.rankDim, kernelType, socVersion);
    formulaicTiling.GetTiling();
    return formulaicTiling.tilingM_.cutRes;
}

} // namespace MC2Tiling

#endif // __MATMUL_ALLTO_ALL_FIT_BALANCE_TILING_H__
