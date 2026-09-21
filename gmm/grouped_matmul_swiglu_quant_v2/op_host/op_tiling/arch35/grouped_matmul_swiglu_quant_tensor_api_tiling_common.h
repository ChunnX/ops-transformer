/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_TENSOR_API_TILING_COMMON_H
#define GROUPED_MATMUL_SWIGLU_QUANT_TENSOR_API_TILING_COMMON_H

#include <algorithm>
#include <cstdint>

#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"

namespace optiling {
namespace GroupedMatmulSwigluQuantTensorApiTiling {

constexpr uint64_t FP8_N_ALIGN = 128U;
constexpr uint64_t K_ALIGN = 128U;
constexpr uint64_t TRANS_B_M_PER_GROUP_LOWER_LIMIT = 128U;
constexpr uint64_t NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT = 512U;
constexpr uint64_t NOT_TRANS_B_M_LOWER_LIMIT = 512U;
constexpr uint64_t AIC_AIV_CORE_RATIO = 2U;
constexpr uint64_t BASE_M = 256U;
constexpr uint64_t BASE_N = 128U;
constexpr uint64_t SWIGLU_SPLIT = 2U;

struct CapabilityParams {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t groupNum;
    uint64_t aicNum;
    uint64_t aivNum;
    bool transB;
    bool platformMemoryReady;
};

inline bool IsShapeAndPlatformCapable(const CapabilityParams &params)
{
    if (!params.platformMemoryReady || params.groupNum == 0U || params.aicNum == 0U ||
        params.aivNum != AIC_AIV_CORE_RATIO * params.aicNum) {
        return false;
    }
    const uint64_t averageM = params.m / params.groupNum;
    const uint64_t mPerGroupLimit =
        params.transB ? TRANS_B_M_PER_GROUP_LOWER_LIMIT : NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT;
    return averageM > mPerGroupLimit && (params.transB || params.m > NOT_TRANS_B_M_LOWER_LIMIT) &&
           params.n % FP8_N_ALIGN == 0U && params.k > 64U && params.k % K_ALIGN == 0U;
}

inline void PrepareBasicBlockForL1(GQmmBasicTiling &tiling)
{
    tiling.baseM = GroupedMatmul::CeilAlign(std::min(tiling.baseM, BASE_M), GmmConstant::CUBE_BLOCK);
    tiling.baseN = GroupedMatmul::CeilAlign(std::min(tiling.baseN, BASE_N), GmmConstant::CUBE_BLOCK);
    tiling.baseN *= SWIGLU_SPLIT;
}

inline void RestoreBasicBlockAfterL1(GQmmBasicTiling &tiling)
{
    tiling.baseN /= SWIGLU_SPLIT;
}

} // namespace GroupedMatmulSwigluQuantTensorApiTiling
} // namespace optiling

#endif // GROUPED_MATMUL_SWIGLU_QUANT_TENSOR_API_TILING_COMMON_H
