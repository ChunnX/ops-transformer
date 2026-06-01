/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTRACT_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTRACT_H_

#include <cstdint>

namespace optiling {
namespace qkv_rms_norm_rope_cache_with_k_scale {

constexpr uint32_t kHeadDimFast = 128;
constexpr uint32_t kMaxTokenTile = 16;
constexpr uint32_t kL0cMaxRows = 256;
constexpr uint32_t kUbInputRowsPerAiv = 160;
constexpr uint32_t kUbOutputRowsPerAiv = 128;
constexpr uint32_t kDimTileFast = 128;
constexpr uint32_t kAivPerAic = 2;

enum class ContractStatus : uint32_t {
    kSuccess = 0,
    kInvalidTotalTokens,
    kInvalidBatch,
    kInvalidHeadNums,
    kInvalidDtype,
    kUnsupportedNvNotEqualNk,
    kUnsupportedHeadDim,
    kInvalidMaxSeqLen,
    kInvalidBlockShape,
    kMissingRotation,
    kMissingVScale,
    kInvalidQkvShape,
    kInvalidGammaShape,
    kInvalidCosSinShape,
    kInvalidSlotMappingShape,
    kInvalidCuSeqLenShape,
    kInvalidRotationShape,
    kInvalidVScaleShape,
    kInvalidCacheShape,
    kInvalidScaleShape,
    kUnsupportedResource
};

enum class ContractDtype : uint32_t {
    kUnknown = 0,
    kBf16,
    kFloat,
    kInt32,
    kFloat8E4M3fn
};

struct TensorShapeInfo {
    uint32_t rank = 0;
    uint64_t dim0 = 0;
    uint64_t dim1 = 0;
    uint64_t dim2 = 0;
    uint64_t dim3 = 0;
    bool present = true;
};

struct TensorContractInfo {
    TensorShapeInfo shape;
    ContractDtype dtype = ContractDtype::kUnknown;
};

struct ContractInput {
    uint32_t totalTokens = 0;
    uint32_t batch = 0;
    uint32_t numQHeads = 0;
    uint32_t numKHeads = 0;
    uint32_t numVHeads = 0;
    uint32_t headDim = 0;
    uint32_t maxSeqLen = 0;
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t aicNum = 0;

    TensorContractInfo qkv;
    TensorContractInfo qGamma;
    TensorContractInfo kGamma;
    TensorContractInfo cosSin;
    TensorContractInfo slotMapping;
    TensorContractInfo kCache;
    TensorContractInfo vCache;
    TensorContractInfo kScaleCache;
    TensorContractInfo cuSeqLen;
    TensorContractInfo rotation;
    TensorContractInfo vScale;
};

struct ContractTiling {
    uint32_t coreTokenTile = 0;
    uint32_t coreGroupNum = 0;
    uint32_t tokenTile = 0;
    uint32_t tokenTilePerAiv = 0;
    uint32_t rowTile = 0;
    uint32_t rowTileAligned = 0;
    uint32_t qkHeadTile = 0;
    uint32_t vHeadTile = 0;
    uint32_t dimTile = 0;
};

struct TileParam {
    uint32_t tileId = 0;
    uint32_t bufferId = 0;
    uint32_t tokenOffset = 0;
    uint32_t tokenSize = 0;
    uint32_t qkHeadOffset = 0;
    uint32_t qkHeadSize = 0;
    uint32_t vHeadOffset = 0;
    uint32_t vHeadSize = 0;
    uint32_t rowOffset = 0;
    uint32_t rowSize = 0;
};

struct AivSlice {
    uint32_t tokenOffset = 0;
    uint32_t tokenSize = 0;
    uint32_t rowOffset = 0;
    uint32_t rowSize = 0;
};

uint32_t AlignUp(uint32_t value, uint32_t align);
uint32_t CeilDiv(uint32_t value, uint32_t divisor);
ContractStatus ValidateAndBuildTiling(const ContractInput &input, ContractTiling &tiling);
uint32_t CalcTileNum(uint32_t totalTokens, uint32_t tokenTile);
uint32_t CalcCoreTokenTile(uint32_t totalTokens, uint32_t aicNum);
uint32_t CalcCoreGroupNum(uint32_t totalTokens, uint32_t aicNum);
AivSlice MakeAivSlice(const TileParam &tile, uint32_t qkHeadTile, uint32_t aivLocalId);
const char *ContractStatusName(ContractStatus status);

} // namespace qkv_rms_norm_rope_cache_with_k_scale
} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTRACT_H_
