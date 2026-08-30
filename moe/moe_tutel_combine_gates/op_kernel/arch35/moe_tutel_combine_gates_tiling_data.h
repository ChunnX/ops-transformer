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
 * \file moe_tutel_combine_gates_tiling_data.h
 * \brief Host and Kernel shared tiling data for Ascend 950.
 */
#ifndef MOE_TUTEL_COMBINE_GATES_TILING_DATA_H_
#define MOE_TUTEL_COMBINE_GATES_TILING_DATA_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Axis names:
//   S = x.shape[0], H = x.shape[1] = y_grad.shape[2]
//   E = y_grad.shape[0], C = y_grad.shape[1]
//   K = indices.shape[0] = locations.shape[0] = gates_grad.shape[0]
// Route tensors are [K, S]. K is independent of E except for 1 <= K <= E.
//
// Core mapping: blockDim = sCoreNum * kCoreNum and
// blockIdx b -> (cs, ck) = (b % sCoreNum, b / sCoreNum).
namespace optiling {
constexpr uint64_t MTCG_CUT_H_ROUTE_BATCH = 2;

struct MoeTutelCombineGatesTilingData {
    uint64_t samples;           // S
    uint64_t hidden;            // H
    uint64_t experts;           // E
    uint64_t capacity;          // C
    uint64_t routes;            // K
    uint64_t sCoreNum;          // Number of cores along S; kCoreNum = blockDim / sCoreNum.
    uint64_t sPerCoreBase;      // S / sCoreNum
    uint64_t sCoreRemain;       // S % sCoreNum.
    uint64_t kPerCoreBase;      // K / kCoreNum.
    uint64_t kCoreRemain;       // K % kCoreNum.
    uint64_t sBlock;            // Logical output tile width along S.
    uint64_t sBlockAlign;       // sBlock aligned to the fp32 vector length.
    uint64_t kBlock;            // Output tile height along K.
    uint64_t hFactor;           // H elements copied per chunk; hidden on NOT_CUT_H.
    uint64_t hFactorAlign;      // hFactor aligned to the fp32 vector length.
    uint64_t numChunks;         // ceil(hidden / hFactor), or zero when H is zero.
    uint64_t hTail;             // Logical H length of the last chunk.
    uint64_t vlPartialAlign;    // Power-of-two VL partial tree length, at least VL_FP32.
    uint64_t chunkPartialAlign; // Power-of-two chunk tree length, CUT_H only.
};
} // namespace optiling

#endif // MOE_TUTEL_COMBINE_GATES_TILING_DATA_H_
