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

constexpr uint64_t kHeadDimFast = 128;
constexpr uint64_t kMaxTokenTile = 16;
constexpr uint64_t kL0aMaxRows = 128;
constexpr uint64_t kL0cMaxRows = 256;
constexpr uint64_t kUbInputRowsPerAiv = 160;
constexpr uint64_t kUbOutputRowsPerAiv = 128;
constexpr uint64_t kDimTileFast = 128;
constexpr uint64_t kAivPerAic = 2;

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
    kMissingQRotation,
    kMissingKRotation,
    kMissingVScale,
    kInvalidQkvShape,
    kInvalidGammaShape,
    kInvalidCosSinShape,
    kInvalidSlotMappingShape,
    kInvalidCuSeqLenShape,
    kInvalidActualSeqLensShape,
    kInvalidQRotationShape,
    kInvalidKRotationShape,
    kInvalidVScaleShape,
    kInvalidCacheShape,
    kInvalidScaleShape,
    kInvalidCacheStride,
    kInvalidScaleStride,
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

struct TensorStrideInfo {
    uint32_t rank = 0;
    uint64_t stride0 = 0;
    uint64_t stride1 = 0;
    uint64_t stride2 = 0;
    uint64_t stride3 = 0;
    bool present = true;
};

struct TensorContractInfo {
    TensorShapeInfo shape;
    TensorStrideInfo stride;
    ContractDtype dtype = ContractDtype::kUnknown;
};

struct ContractInput {
    uint64_t totalTokens = 0;
    uint64_t batch = 0;
    uint64_t numQHeads = 0;
    uint64_t numKHeads = 0;
    uint64_t numVHeads = 0;
    uint64_t headDim = 0;
    uint64_t maxSeqLen = 0;
    uint64_t blockNum = 0;
    uint64_t blockSize = 0;
    uint64_t aicNum = 0;

    TensorContractInfo qkv;
    TensorContractInfo qGamma;
    TensorContractInfo kGamma;
    TensorContractInfo cosSin;
    TensorContractInfo slotMapping;
    TensorContractInfo kCache;
    TensorContractInfo vCache;
    TensorContractInfo kScaleCache;
    TensorContractInfo cuSeqLen;
    TensorContractInfo actualSeqLens;
    TensorContractInfo qRotation;
    TensorContractInfo kRotation;
    TensorContractInfo vScale;
};

struct ContractTiling {
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t tokenTile = 0;
    uint64_t tokenTilePerAiv = 0;
    uint64_t rowTile = 0;
    uint64_t rowTileAligned = 0;
    uint64_t qkHeadTile = 0;
    uint64_t vHeadTile = 0;
    uint64_t dimTile = 0;
    uint64_t kCacheStrideBlock = 0;
    uint64_t kCacheStrideHead = 0;
    uint64_t kCacheStrideToken = 0;
    uint64_t vCacheStrideBlock = 0;
    uint64_t vCacheStrideHead = 0;
    uint64_t vCacheStrideToken = 0;
    uint64_t kScaleCacheStrideBlock = 0;
    uint64_t kScaleCacheStrideHead = 0;
    uint64_t kScaleCacheStrideToken = 0;
};

struct TileParam {
    uint64_t tileId = 0;
    uint64_t bufferId = 0;
    uint64_t tokenOffset = 0;
    uint64_t tokenSize = 0;
    uint64_t qkHeadOffset = 0;
    uint64_t qkHeadSize = 0;
    uint64_t vHeadOffset = 0;
    uint64_t vHeadSize = 0;
    uint64_t rowOffset = 0;
    uint64_t rowSize = 0;
};

struct AivSlice {
    uint64_t tokenOffset = 0;
    uint64_t tokenSize = 0;
    uint64_t rowOffset = 0;
    uint64_t rowSize = 0;
};

uint64_t AlignUp(uint64_t value, uint64_t align);
uint64_t CeilDiv(uint64_t value, uint64_t divisor);
uint64_t CalcCubeTokenSize(uint64_t tokenSize);
uint64_t CalcCubeHalfTokenSize(uint64_t tokenSize);
ContractStatus ValidateAndBuildTiling(const ContractInput &input, ContractTiling &tiling,
                                      const char *opName = nullptr);
uint64_t CalcTileNum(uint64_t totalTokens, uint64_t tokenTile);
uint64_t CalcCoreTokenTile(uint64_t totalTokens, uint64_t aicNum);
uint64_t CalcCoreGroupNum(uint64_t totalTokens, uint64_t aicNum);
AivSlice MakeAivSlice(const TileParam &tile, uint64_t qkHeadTile, uint64_t aivLocalId);
const char *ContractStatusName(ContractStatus status);

} // namespace qkv_rms_norm_rope_cache_with_k_scale
} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTRACT_H_
