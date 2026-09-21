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
 * \file qbsa_mxfp8_tiling_data.h
 * \brief Independent MXFP8 host/kernel tiling payload.
 */

#ifndef QBSA_MXFP8_TILING_DATA_H_
#define QBSA_MXFP8_TILING_DATA_H_

#include "../common/quant_block_sparse_attn_tiling_data.h"

namespace optiling {
constexpr uint8_t MX_PSCALE_DTYPE_E8M0 = 0U;
constexpr uint8_t MX_PSCALE_DTYPE_FP32 = 1U;

#pragma pack(push, 8)

struct QuantBlockSparseAttnMxStrideParams {
    uint64_t bnStride;
    uint64_t n2Stride;
};

// MXFP8 group-scale layout and the independent PA scale strides.
struct QuantBlockSparseAttnMxParams {
    QuantBlockSparseAttnMxStrideParams keyStrides;
    QuantBlockSparseAttnMxStrideParams valueStrides;
    QuantBlockSparseAttnMxStrideParams kScaleStrides;
    QuantBlockSparseAttnMxStrideParams vScaleStrides;
    uint32_t scaleGroupSize;
    uint32_t scaleLastDim;
    uint32_t queryScaleDSize;
    uint32_t keyScaleDSize;
    uint32_t valueScaleBlockSize;
    uint32_t valueScaleDSize;
    uint8_t pScaleDtype;
};

struct QuantBlockSparseAttnMxfp8TilingData {
    QuantBlockSparseAttnCommonParams commonParams;
    QuantBlockSparseAttnMxParams mxParams;
};

#pragma pack(pop)

static_assert(sizeof(QuantBlockSparseAttnMxParams) == 96U, "QBSA MXFP8 parameters ABI changed");
static_assert(offsetof(QuantBlockSparseAttnMxfp8TilingData, commonParams) == 0U, "QBSA common tiling offset changed");
static_assert(offsetof(QuantBlockSparseAttnMxfp8TilingData, mxParams) == 120U, "QBSA MXFP8 tiling offset changed");
static_assert(offsetof(QuantBlockSparseAttnMxParams, kScaleStrides) == 32U, "QBSA K scale offset changed");
static_assert(offsetof(QuantBlockSparseAttnMxParams, vScaleStrides) == 48U, "QBSA V scale offset changed");
static_assert(offsetof(QuantBlockSparseAttnMxParams, scaleLastDim) == 68U, "QBSA scale last dimension offset changed");
static_assert(sizeof(QuantBlockSparseAttnMxfp8TilingData) == 216U, "QBSA MXFP8 tiling wire size changed");
static_assert(std::is_standard_layout<QuantBlockSparseAttnMxfp8TilingData>::value,
              "QBSA MXFP8 tiling data must have a standard layout");
static_assert(std::is_trivial<QuantBlockSparseAttnMxfp8TilingData>::value,
              "QBSA MXFP8 tiling data must remain a trivial POD");
static_assert(std::is_trivially_copyable<QuantBlockSparseAttnMxfp8TilingData>::value,
              "QBSA MXFP8 tiling data must support byte-wise serialization");
} // namespace optiling

#endif // QBSA_MXFP8_TILING_DATA_H_
