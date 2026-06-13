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
 * \file inplace_partial_rotary_mul_grad_tiling_data.h
 * \brief tiling data struct for InplacePartialRotaryMulGrad (arch35 / ascend950)
 *
 * Only the x == nullptr scenario is supported: dy is updated in place to become dx over the
 * [partialStart, partialStart + sliceLen) slice of its last dim, dcos / dsin are not computed.
 * Only BSND layout and B/S/N broadcast (111D, B1ND, BS1D, ...) are supported.
 */

#ifndef INPLACE_PARTIAL_ROTARY_MUL_GRAD_TILING_DATA_H_
#define INPLACE_PARTIAL_ROTARY_MUL_GRAD_TILING_DATA_H_

#include <cstdint>

// Compact dtype code shared between host tiling and kernel dispatch.
//   0 -> float32, 1 -> float16, 2 -> bfloat16
constexpr int64_t INPLACE_PARTIAL_ROTARY_DTYPE_FP32 = 0;
constexpr int64_t INPLACE_PARTIAL_ROTARY_DTYPE_FP16 = 1;
constexpr int64_t INPLACE_PARTIAL_ROTARY_DTYPE_BF16 = 2;

struct InplacePartialRotaryMulGradTilingData {
    int64_t b;              // dy dim B (BSND)
    int64_t s;              // dy dim S
    int64_t n;              // dy dim N
    int64_t d;              // dy dim D (full last dim)
    int64_t partialStart;   // slice start on D
    int64_t sliceLen;       // slice length (end - start), i.e. cos/sin last dim
    int64_t rotaryMode;     // 0=half, 1=interleave, 2=quarter, 3=interleave-half
    int64_t dSplitCoef;     // rotary split coefficient: 1 (interleave), 2 (half/interleave-half), 4 (quarter)
    int64_t dyDtypeCode;    // dtype code of dy (and dx output)
    int64_t cosDtypeCode;   // dtype code of cos/sin
    // broadcast strides of cos/sin over the (B,S,N) grid, in units of sliceLen-element rows.
    // A stride of 0 means that axis is broadcast (== 1).
    int64_t cosStrideB;
    int64_t cosStrideS;
    int64_t cosStrideN;
    int64_t usedCoreNum;    // number of cores actually used
    int64_t blockFactorM;   // rows (B*S*N) handled per core (except tail)
    int64_t ubFactorM;      // rows handled per UB tile
    int64_t isFullBroadcast; // 1 when cos/sin broadcast fully (cosB==cosS==cosN==1)
};

#endif // INPLACE_PARTIAL_ROTARY_MUL_GRAD_TILING_DATA_H_
