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
 * \file quant_block_sparse_attn_mx_tiling_data.h
 * \brief host tiling 与 MX kernel 共享的 MXFP8 全量化 tiling data。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_TILING_DATA_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_TILING_DATA_H_

#include <cstdint>

namespace optiling {
constexpr uint32_t BSA_MX_MAX_CORE_NUM = 48U;
constexpr uint32_t BSA_MX_CORE_SPLIT_NUM = BSA_MX_MAX_CORE_NUM + 1U;

// MX tiling data 有意与现有 FP8 tiling data 隔离。MX kernel 参考 common MX
// 全量化思路，因此运行期字段按特性区域分组，并使用普通 struct，
// 不复用 FP8 的生成式 tiling class。
struct QuantBlockSparseAttnMxStrideParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

struct QuantBlockSparseAttnMxAttrParams {
    uint32_t layoutQ = 0;
    uint32_t layoutKv = 0;
    uint32_t layoutSparseIndices = 0;
    uint32_t quantMode = 0;
    uint32_t maskMode = 0;
    uint32_t returnSoftmaxLse = 0;
};

struct QuantBlockSparseAttnMxBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t t2Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeRope = 0;
    uint32_t actualSeqLengthsQSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    float scaleValue = 0.0F;
    uint8_t isKvContinuous = 0;
    uint8_t isSoftMaxLseEnable = 0;
    uint8_t isActualSeqLengthsNull = 0;
    uint8_t isActualSeqLengthsKVNull = 0;
    uint32_t coreNum = 0;
    uint32_t outputLayout = 0;
    // K/V 数据以及 K/V scale 的 PA BNBD stride。这里以元素个数保存，
    // kernel helper 在 block_table 完成物理 block 映射后据此计算偏移。
    QuantBlockSparseAttnMxStrideParams keyStrides;
    QuantBlockSparseAttnMxStrideParams valueStrides;
    QuantBlockSparseAttnMxStrideParams kScaleStrides;
    QuantBlockSparseAttnMxStrideParams vScaleStrides;
};

struct QuantBlockSparseAttnMxAttenMaskParams {
    uint8_t sparseMode = 0;
    uint8_t attenMaskDataType = 0;
    uint8_t attenMaskCompressMode = 0;
    uint8_t isRowInvalidOpen = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint8_t isExistRowInvalid = 0;
    uint8_t reserve[3] = {0, 0, 0};
};

struct QuantBlockSparseAttnMxPageAttentionParams {
    uint8_t paLayoutType = 0;
    uint8_t reserve[3] = {0, 0, 0};
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t paBlockNumSum = 0;
    uint32_t paBlockStride = 0;
    uint32_t qBlockSize = 0;
    uint32_t kvBlockSize = 0;
};

struct QuantBlockSparseAttnMxSparseParams {
    uint32_t gS1OuterSize = 0;
    uint32_t sparseSeqLenStride = 0;
    uint32_t sparseIndicesStride = 0;
    uint32_t maxQb = 0;
    uint32_t maxKb = 0;
    uint32_t sparseCount = 0;
};

struct QuantBlockSparseAttnMxMultiCoreParams {
    int32_t coreNum = 0;
    int64_t totalSize = 0;
    int64_t s1OuterSize = 0;
    int64_t splitFactorSize = 0;
    int64_t splitFactorTailSize = 0;
    uint32_t bnStartIdx[BSA_MX_CORE_SPLIT_NUM] = {};
    int64_t sparseStartIdx[BSA_MX_CORE_SPLIT_NUM] = {};
    int64_t firstFullLoadS1OuterIdx = 0;
    uint8_t splitCoreMode = 0;
    uint8_t reserve[3] = {0, 0, 0};
};

struct QuantBlockSparseAttnMxWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct QuantBlockSparseAttnMxScaleParams {
    // quant_mode=2 的 scale 语义约束：
    //   quantScale1:        [1]，静态 P 量化 scale，fp8_e8m0
    //   dequantScaleQuery:  [T, N, D/64, 2]，per-token-group，fp8_e8m0
    //   keyAntiquantScale:  [Tkv, N, D/64, 2]，per-token-group，fp8_e8m0
    //   valueAntiquantScale:[Tkv/64, N, DV, 2]，per-channel-group，fp8_e8m0
    uint32_t scaleGroupSize = 0;
    uint32_t scaleLastDim = 0;
    uint32_t queryScaleDSize = 0;
    uint32_t keyScaleDSize = 0;
    uint32_t valueScaleBlockSize = 0;
    uint32_t valueScaleDSize = 0;
    uint32_t pScaleShapeSize = 0;
    uint32_t queryQuantMode = 0;
    uint32_t keyAntiquantMode = 0;
    uint32_t valueAntiquantMode = 0;
};

struct QuantBlockSparseAttnMxEmptyTensorParams {
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
    uint8_t isOneN = 0;
    uint8_t reserve[2] = {0, 0};
    uint64_t totalOutputSize = 0;
    uint64_t totalSoftMaxLseOutputSize = 0;
};

struct QuantBlockSparseAttnMxTilingData {
    // 各参数组保持显式分离。host 只在 MXFullQuantMode 下填充该 struct；
    // FP8 仍继续使用 QuantBlockSparseAttnTilingData。
    QuantBlockSparseAttnMxAttrParams attrParams;
    QuantBlockSparseAttnMxBaseParams baseParams;
    QuantBlockSparseAttnMxAttenMaskParams attenMaskParams;
    QuantBlockSparseAttnMxPageAttentionParams pageAttentionParams;
    QuantBlockSparseAttnMxSparseParams sparseParams;
    QuantBlockSparseAttnMxMultiCoreParams multiCoreParams;
    QuantBlockSparseAttnMxWorkspaceParams workspaceParams;
    QuantBlockSparseAttnMxScaleParams scaleParams;
    QuantBlockSparseAttnMxEmptyTensorParams emptyTensorParams;
};
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_TILING_DATA_H_
