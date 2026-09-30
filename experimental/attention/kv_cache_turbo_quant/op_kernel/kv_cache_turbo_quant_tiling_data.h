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
 * \file kv_cache_turbo_quant_tiling_data.h
 * \brief TilingData shared by host tiling and device kernel (keep the two sides identical).
 */

#ifndef KV_CACHE_TURBO_QUANT_TILING_DATA_H
#define KV_CACHE_TURBO_QUANT_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

// The codebook of this operator is the d=128 Lloyd-Max table, so headDim is fixed by design.
constexpr uint32_t KCTQ_HEAD_DIM = 128;
constexpr uint32_t KCTQ_QJL_DIM = 128;
// Task scope: standard MHA/GQA KV heads live in [4, 32]; out-of-scope inputs are rejected by tiling.
constexpr uint32_t KCTQ_MIN_HEADS = 4;
constexpr uint32_t KCTQ_MAX_HEADS = 32;
// One cube core serves this many vector cores (the kernel is declared MIX_AIC_1_2).
constexpr uint32_t KCTQ_AIV_PER_AIC = 2;

// Tiling key == mse_bits: one template instantiation per supported primary width.
constexpr uint64_t KCTQ_TILING_KEY_MSE2 = 2;
constexpr uint64_t KCTQ_TILING_KEY_MSE3 = 3;
constexpr uint64_t KCTQ_TILING_KEY_MSE4 = 4;

// Workspace layout (byte offsets from the user workspace base), all 512B aligned.
// The workspace reserves aivRowStride rows per AIV (baseM aligned), not rowsPerCore / AIV per AIC:
// the C tiles are written by the cube straight into the workspace, so the last, partial tile of an
// AIV still occupies a whole baseM block and must not reach into the next AIV's rows.
struct KvCacheTurboQuantTilingData {
    TCubeTiling cubeTiling;  // single-core view of the (rowsPerCore x headDim) x (headDim x headDim) GEMM
    uint32_t mseBits;        // 2 / 3 / 4
    uint32_t usedCoreNum;    // AI core blocks (AIC and AIV are 1:1)
    uint32_t totalRows;      // numTokens * numKvHeads
    uint32_t rowsPerCore;    // rows owned by one core (16 aligned)
    uint32_t alignedRows;    // rowsPerCore * usedCoreNum
    uint32_t aivRowStride;   // rows reserved per AIV in the workspace (baseM aligned)
    uint32_t headDim;        // 128
    uint32_t idxBytesPerRow; // headDim * mseBits / 8
    uint32_t qjlBytesPerRow; // headDim / 8
    uint64_t wsU;            // normalized vectors (fp32)  [wsRows, headDim]
    uint64_t wsY;            // rotated vectors   (fp32)  [wsRows, headDim]
    uint64_t wsR;            // residual unit vectors (fp32) [wsRows, headDim] (aliases wsU)
    uint64_t wsP;            // residual projections  (fp32) [wsRows, headDim] (aliases wsY)
    uint64_t wsNorm;         // per row L2 norm   (fp32)  [wsRows]
    uint64_t wsBytes;        // total user workspace bytes
    uint32_t matrixBf16;     // 1 = rotation_matrix / qjl_matrix arrive as bf16 and are upcast to fp32
    uint32_t metadataFp16;   // 1 = quant_norm / quant_gamma are fp16 instead of bf16
    uint64_t wsRotMat;       // fp32 copy of rotation_matrix (matrixBf16 only) [headDim, headDim]
    uint64_t wsQjlMat;       // fp32 copy of qjl_matrix     (matrixBf16 only) [headDim, headDim]
};

} // namespace optiling

#endif // KV_CACHE_TURBO_QUANT_TILING_DATA_H
