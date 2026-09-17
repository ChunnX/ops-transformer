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

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace optiling {
constexpr uint32_t QBSA_MAX_CORE_NUM = 36U;
constexpr uint32_t QBSA_CORE_SPLIT_NUM = QBSA_MAX_CORE_NUM + 1U;
constexpr uint8_t MX_PSCALE_DTYPE_E8M0 = 0U;
constexpr uint8_t MX_PSCALE_DTYPE_FP32 = 1U;

struct QuantBlockSparseAttnBaseParams {
    uint32_t bSize;
    uint32_t t1Size;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t coreNum;
    uint32_t pScaleShapeSize;
    float scaleValue;
};

struct QuantBlockSparseAttnPaParams {
    uint32_t blockTableDim2;
    uint32_t paBlockNumSum;
    uint32_t paLayoutType;
    uint32_t blockSize;
    uint32_t kvBlockSize;
    uint32_t qBlockSize;
    uint32_t paBlockStride;
    uint8_t isRowInvalid;
};

struct QuantBlockSparseAttnSparseParams {
    uint32_t sparseSeqLenStride;
    uint32_t sparseIndicesStride;
};

struct QuantBlockSparseAttnMaskParams {
    int64_t preTokens;
    int64_t nextTokens;
    uint32_t attenMaskS2Size;
    uint8_t attenMaskCompressMode;
};

struct QuantBlockSparseAttnOutputParams {
    int64_t totalOutputSize;
    int64_t totalSoftMaxLseOutputSize;
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
    uint32_t bandIndex;
    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;
    uint8_t layoutType;
    uint8_t isKvContinuous;
    uint8_t fromFused;
    uint8_t isGqa;
    int64_t s1OuterSize;
    uint32_t bnStartIdx[QBSA_CORE_SPLIT_NUM];
    uint32_t singleCoreSize;
    uint8_t needInit;
};

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

// One serialization format for both modes. The mode-specific blocks share storage
// because one invocation can only select one quantization mode and its matching tiling key.
struct QuantBlockSparseAttnTilingData {
    QuantBlockSparseAttnCommonParams commonParams;
    union {
        QuantBlockSparseAttnFp8Params fp8Params;
        QuantBlockSparseAttnMxParams mxParams;
    };
};

constexpr std::size_t QBSA_MODE_PARAMS_SIZE = sizeof(QuantBlockSparseAttnFp8Params) >
                                                      sizeof(QuantBlockSparseAttnMxParams) ?
                                                  sizeof(QuantBlockSparseAttnFp8Params) :
                                                  sizeof(QuantBlockSparseAttnMxParams);
static_assert(sizeof(QuantBlockSparseAttnTilingData) ==
                  sizeof(QuantBlockSparseAttnCommonParams) + QBSA_MODE_PARAMS_SIZE,
              "QBSA mode-specific tiling data must share one payload");
static_assert(std::is_standard_layout<QuantBlockSparseAttnTilingData>::value,
              "QBSA tiling data must have a standard layout");
static_assert(std::is_trivial<QuantBlockSparseAttnTilingData>::value, "QBSA tiling data must remain a trivial POD");
static_assert(std::is_trivially_copyable<QuantBlockSparseAttnTilingData>::value,
              "QBSA tiling data must support byte-wise serialization");
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
