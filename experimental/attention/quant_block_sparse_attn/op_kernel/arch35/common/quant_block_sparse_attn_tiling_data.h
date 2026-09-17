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
 * \brief Common tiling fields shared by the independent FP8 and MXFP8 payloads.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
#define QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace optiling {
constexpr uint32_t QBSA_MAX_CORE_NUM = 36U;

// Host and Kernel exchange raw bytes. Do not inherit packing from SDK/generated headers.
#pragma pack(push, 8)

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

#pragma pack(pop)

// Fixed wire offsets must agree in both compilers; equal total sizes alone are insufficient.
static_assert(sizeof(QuantBlockSparseAttnCommonParams) == 120U, "QBSA common tiling ABI changed");
static_assert(offsetof(QuantBlockSparseAttnCommonParams, paParams) == 36U, "QBSA PA tiling offset changed");
static_assert(offsetof(QuantBlockSparseAttnCommonParams, sparseParams) == 68U, "QBSA sparse tiling offset changed");
static_assert(offsetof(QuantBlockSparseAttnCommonParams, maskParams) == 80U, "QBSA mask tiling offset changed");
static_assert(offsetof(QuantBlockSparseAttnCommonParams, outputParams) == 104U, "QBSA output tiling offset changed");
static_assert(std::is_standard_layout<QuantBlockSparseAttnCommonParams>::value,
              "QBSA common tiling fields must have a standard layout");
static_assert(std::is_trivially_copyable<QuantBlockSparseAttnCommonParams>::value,
              "QBSA common tiling fields must support byte-wise serialization");
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
