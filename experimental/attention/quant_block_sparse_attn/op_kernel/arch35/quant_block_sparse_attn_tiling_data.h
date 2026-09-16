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
 * \file quant_block_sparse_attn_tiling_data.h
 * \brief Shared host/kernel tiling data for FP8 and MXFP8.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
#define QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_

#include <cstdint>
#include <type_traits>

namespace optiling {
constexpr uint32_t QBSA_MAX_CORE_NUM = 36U;
constexpr uint32_t QBSA_CORE_SPLIT_NUM = QBSA_MAX_CORE_NUM + 1U;
constexpr uint8_t MX_PSCALE_DTYPE_E8M0 = 0U;
constexpr uint8_t MX_PSCALE_DTYPE_FP32 = 1U;

struct QuantBlockSparseAttnBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t coreNum = 0;
    uint32_t pScaleShapeSize = 0;
    float scaleValue = 0.0F;
};

struct QuantBlockSparseAttnPaParams {
    uint32_t blockTableDim2 = 0;
    uint32_t paBlockNumSum = 0;
    uint32_t paLayoutType = 0;
    uint32_t blockSize = 0;
    uint32_t kvBlockSize = 0;
    uint32_t qBlockSize = 0;
    uint32_t paBlockStride = 0;
    uint8_t isRowInvalid = 0;
};

struct QuantBlockSparseAttnSparseParams {
    uint32_t sparseSeqLenStride = 0;
    uint32_t sparseIndicesStride = 0;
};

struct QuantBlockSparseAttnMaskParams {
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    uint32_t attenMaskS2Size = 0;
    uint8_t attenMaskCompressMode = 0;
};

struct QuantBlockSparseAttnOutputParams {
    int64_t totalOutputSize = 0;
    int64_t totalSoftMaxLseOutputSize = 0;
};

struct QuantBlockSparseAttnCommonParams {
    QuantBlockSparseAttnBaseParams baseParams;
    QuantBlockSparseAttnPaParams paParams;
    QuantBlockSparseAttnSparseParams sparseParams;
    QuantBlockSparseAttnMaskParams maskParams;
    QuantBlockSparseAttnOutputParams outputParams;
};

// FP8 RegBase initialization and task-boundary fields.
struct QuantBlockSparseAttnFp8Params {
    uint32_t bandIndex = 0;
    uint32_t seqUsedQlenSize = 0;
    uint32_t seqUsedKvlenSize = 0;
    uint8_t layoutType = 0;
    uint8_t isKvContinuous = 0;
    uint8_t fromFused = 0;
    uint8_t isGqa = 0;
    int64_t s1OuterSize = 0;
    uint32_t bnStartIdx[QBSA_CORE_SPLIT_NUM] = {};
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
};

struct QuantBlockSparseAttnMxStrideParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

// MXFP8 group-scale layout and the independent PA scale strides.
struct QuantBlockSparseAttnMxParams {
    QuantBlockSparseAttnMxStrideParams keyStrides;
    QuantBlockSparseAttnMxStrideParams valueStrides;
    QuantBlockSparseAttnMxStrideParams kScaleStrides;
    QuantBlockSparseAttnMxStrideParams vScaleStrides;
    uint32_t scaleGroupSize = 0;
    uint32_t scaleLastDim = 0;
    uint32_t queryScaleDSize = 0;
    uint32_t keyScaleDSize = 0;
    uint32_t valueScaleBlockSize = 0;
    uint32_t valueScaleDSize = 0;
    uint8_t pScaleDtype = 0;
};

// One serialization format for both modes. Mode-specific blocks are initialized
// only by their feature branch; kernels consume the matching block selected by the tiling key.
struct QuantBlockSparseAttnTilingData {
    QuantBlockSparseAttnCommonParams commonParams;
    QuantBlockSparseAttnFp8Params fp8Params;
    QuantBlockSparseAttnMxParams mxParams;
};

static_assert(std::is_standard_layout<QuantBlockSparseAttnTilingData>::value,
              "QBSA tiling data must have a standard layout");
static_assert(std::is_trivially_copyable<QuantBlockSparseAttnTilingData>::value,
              "QBSA tiling data must support byte-wise serialization");
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
