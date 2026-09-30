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

constexpr uint32_t KCTQ_HEAD_DIM = 128;
constexpr uint32_t KCTQ_MIN_HEADS = 4;
constexpr uint32_t KCTQ_MAX_HEADS = 32;
constexpr uint32_t KCTQ_AIV_PER_AIC = 2;
constexpr int64_t KCTQ_MSE_BITS_MIN = 2;
constexpr int64_t KCTQ_MSE_BITS_MAX = 4;
constexpr int64_t KCTQ_MSE_BITS_DEFAULT = 3;

// Device-side POD mirror. Field order and types must match the host TilingData.
// The ws* fields are byte offsets into the user workspace (512B aligned); wsR aliases wsU,
// wsP aliases wsY (the four row buffers are used in disjoint phases of the pipeline).
struct KvCacheTurboQuantTilingData {
    TCubeTiling cubeTiling;
    uint32_t mseBits;
    uint32_t usedCoreNum;
    uint32_t totalRows;
    uint32_t rowsPerCore;      // 16 aligned
    uint32_t alignedRows;
    uint32_t aivRowStride;
    uint32_t headDim;
    uint32_t idxBytesPerRow;
    uint32_t qjlBytesPerRow;
    uint64_t wsU;              // [wsRows, headDim]
    uint64_t wsY;
    uint64_t wsR;
    uint64_t wsP;
    uint64_t wsNorm;           // [wsRows]
    uint64_t wsBytes;
    uint32_t matrixBf16;       // 1 = matrices arrive as bf16, upcast to fp32
    uint32_t metadataFp16;     // 1 = quant_norm / quant_gamma are fp16
    uint64_t wsRotMat;         // fp32 copy of rotation_matrix (matrixBf16 only)
    uint64_t wsQjlMat;         // fp32 copy of qjl_matrix      (matrixBf16 only)
};

} // namespace optiling

#endif // KV_CACHE_TURBO_QUANT_TILING_DATA_H
