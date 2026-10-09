/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 通信模型: 线性拟合 y = k·x + 轮数·b + c
#pragma once

#include "fit_balance_types.h"

constexpr uint64_t FP4_PACK_RATIO = 2;

class HCCLCommModel {
public:
    CommFitParams commEstimatePar_;
    CommModelInfo commTypeInfo_;
    double commTimeFactor_ = 1.0; // 时间折算因子, 恒 1

    // rankDim 下限钳位 2: 防 uint64 下溢(FRONT 的 rankDim−1)、单卡无集合通信可建模
    HCCLCommModel(uint32_t inputRankDim, CommFitParams commFitParams)
    {
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_RANK_DIM);
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        commEstimatePar_ = commFitParams;
    };

    void SetCommShapeLen(uint64_t len)
    {
        commTypeInfo_.commMatrixLen = len;
    }
    void SetCommDTypeSize(uint64_t dTypeSize)
    {
        commTypeInfo_.commDtypeSize = dTypeSize;
    }
    void SetCommDtypeSizeExpansionFraction(uint64_t number)
    {
        commTypeInfo_.commDtypeSizeExpansionFraction = number;
    }
    // 有效每元素字节数 = dtypeSize / fraction(fp4: 1/2 = 0.5B)
    double GetRealDtypeSizes() const
    {
        return static_cast<double>(commTypeInfo_.commDtypeSize) / commTypeInfo_.commDtypeSizeExpansionFraction;
    }
    uint64_t rankDim() const
    {
        return commTypeInfo_.rankDim;
    }

    // 单轮通信耗时: y = k·x + b(c 不进单轮口径)
    double CommTime(uint64_t mSize) const
    {
        const double commDataSize = static_cast<double>(mSize) * commTypeInfo_.commMatrixLen * GetRealDtypeSizes();
        const double tmpSize = commDataSize / BYTES_PER_MB;
        return (commEstimatePar_.slopeUsPerMb * tmpSize + commEstimatePar_.roundOverheadUs) / commTimeFactor_;
    }

    // 反解: x = (t − b) / k, 换算回 M 行数
    uint64_t InverseCommTime(double targetTime) const
    {
        const double tmpSize =
            (targetTime * commTimeFactor_ - commEstimatePar_.roundOverheadUs) / commEstimatePar_.slopeUsPerMb;
        return static_cast<uint64_t>(tmpSize * BYTES_PER_MB / (commTypeInfo_.commMatrixLen * GetRealDtypeSizes()));
    }

    // 总通信耗时: y = k·X + 轮数·b + c
    double TotalCommTime(uint64_t totalLen, uint64_t numRounds) const
    {
        const double commDataSize = static_cast<double>(totalLen) * commTypeInfo_.commMatrixLen * GetRealDtypeSizes();
        const double tmpSize = commDataSize / BYTES_PER_MB;
        return (commEstimatePar_.slopeUsPerMb * tmpSize +
                commEstimatePar_.roundOverheadUs * static_cast<double>(numRounds) + commEstimatePar_.fixedOverheadUs) /
               commTimeFactor_;
    }
};
