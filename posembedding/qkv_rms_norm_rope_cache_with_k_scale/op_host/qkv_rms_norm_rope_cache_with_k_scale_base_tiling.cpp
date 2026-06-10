/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h"

#include <algorithm>
#include <limits>
#include <string>

#include "../op_kernel/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h"
#include "log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "securec.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace QkvRmsNormRopeCacheWithKScale {
constexpr size_t kQkvIndex = 0;
constexpr size_t kQGammaIndex = 1;
constexpr size_t kKGammaIndex = 2;
constexpr size_t kCosSinIndex = 3;
constexpr size_t kSlotMappingIndex = 4;
constexpr size_t kKCacheIndex = 5;
constexpr size_t kVCacheIndex = 6;
constexpr size_t kKScaleCacheIndex = 7;
constexpr size_t kCuSeqLenIndex = 8;
constexpr size_t kQRotationIndex = 9;
constexpr size_t kKRotationIndex = 10;
constexpr size_t kVScaleIndex = 11;

constexpr size_t kHeadNumsAttrIndex = 0;
constexpr size_t kEpsilonAttrIndex = 1;

constexpr float kDefaultEpsilon = 1e-6f;
constexpr int32_t kTilingTemplatePriority = 1000;
constexpr size_t kWorkspaceCount = 1;
constexpr size_t kReservedWorkspaceSize = 16UL * 1024UL * 1024UL;

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape1D(const TensorShapeInfo &shape, uint64_t d0)
{
    return shape.present && shape.rank == 1 && shape.dim0 == d0;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape2D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1)
{
    return shape.present && shape.rank == 2 && shape.dim0 == d0 && shape.dim1 == d1;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape3D(
    const TensorShapeInfo &shape, uint64_t d0, uint64_t d1, uint64_t d2)
{
    return shape.present && shape.rank == 3 && shape.dim0 == d0 && shape.dim1 == d1 && shape.dim2 == d2;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape4D(
    const TensorShapeInfo &shape, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    return shape.present && shape.rank == 4 && shape.dim0 == d0 && shape.dim1 == d1 && shape.dim2 == d2 &&
           shape.dim3 == d3;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::HasDtype(const TensorContractInfo &tensor, ContractDtype expected)
{
    return tensor.dtype == expected;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsCacheStrideValid(const TensorStrideInfo &stride, uint64_t headDim)
{
    return stride.present && stride.rank == 4 && stride.stride0 > 0 && stride.stride1 >= headDim &&
           stride.stride2 >= headDim && stride.stride3 == 1;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsScaleStrideValid(const TensorStrideInfo &stride)
{
    return stride.present && stride.rank == 4 && stride.stride0 > 0 && stride.stride1 >= 1 && stride.stride2 >= 1 &&
           stride.stride3 == 1;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::CacheNonInnermostStridesMatch(
    const TensorStrideInfo &lhs, const TensorStrideInfo &rhs)
{
    return lhs.stride0 == rhs.stride0 && lhs.stride1 == rhs.stride1 && lhs.stride2 == rhs.stride2;
}

uint64_t QkvRmsNormRopeCacheWithKScaleTiling::CalcQkAlignedRows(
    uint64_t tokenTile, uint64_t numQHeads, uint64_t numKHeads)
{
    return Ops::Base::CeilAlign(tokenTile * numQHeads, 16UL) +
           Ops::Base::CeilAlign(tokenTile * numKHeads, 16UL);
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogContractInputSummary(
    const std::string &opName, const ContractInput &input, uint64_t qkN, uint64_t qkvN)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract input: T=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu qkN=%lu qkvN=%lu "
            "headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.totalTokens, input.batch, input.numQHeads, input.numKHeads, input.numVHeads, qkN, qkvN,
            input.headDim, input.maxSeqLen, input.blockNum, input.blockSize, input.aicNum);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale resource constants: maxTokenTile=%lu ubInputRowsPerAiv=%lu "
            "ubOutputRowsPerAiv=%lu fp8NtdStagingRowsPerAiv=%lu l0aMaxRows=%lu l0cMaxRows=%lu aivPerAic=%lu.",
            kMaxTokenTile, kUbInputRowsPerAiv, kUbOutputRowsPerAiv, kFp8NtdStagingRowsPerAiv, kL0aMaxRows,
            kL0cMaxRows, kAivPerAic);
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogTokenTileCandidate(
    const std::string &opName, uint64_t tokenTilePerAiv, uint64_t tokenTile, uint64_t rowTile,
    uint64_t rowTileAligned, uint64_t qkAlignedRows, bool fitsUb, bool fitsL0a, bool fitsL0c, bool accepted)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale token tile candidate: tokenTilePerAiv=%lu tokenTile=%lu rowTile=%lu "
            "rowTileAligned=%lu qkAlignedRows=%lu fitsUb=%u fitsL0a=%u fitsL0c=%u accepted=%u.",
            tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb ? 1U : 0U,
            fitsL0a ? 1U : 0U, fitsL0c ? 1U : 0U, accepted ? 1U : 0U);
}

auto QkvRmsNormRopeCacheWithKScaleTiling::CalcDerivedHeadNums(const ContractInput &input) -> DerivedHeadNums
{
    DerivedHeadNums nums;
    nums.qkN = input.numQHeads + input.numKHeads;
    nums.qkvN = nums.qkN + input.numVHeads;
    return nums;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateScalarInputs(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(input.totalTokens == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: totalTokens is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.batch == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: batch is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.numQHeads == 0 || input.numKHeads == 0 || input.numVHeads == 0,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: invalid head nums Nq=%lu Nk=%lu Nv=%lu.",
                        input.numQHeads, input.numKHeads, input.numVHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.numVHeads != input.numKHeads,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: numVHeads[%lu] != numKHeads[%lu].",
                        input.numVHeads, input.numKHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.headDim != kHeadDimFast,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: unsupported headDim=%lu.",
                        input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.maxSeqLen == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: maxSeqLen is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.blockNum == 0 || input.blockSize == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: blockNum=%lu blockSize=%lu.",
                        input.blockNum, input.blockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.aicNum == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: aicNum is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.qRotation.shape.present,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: qRotation is missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.kRotation.shape.present,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: kRotation is missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.vScale.shape.present,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: vScale is missing."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateDerivedHeadNums(
    const DerivedHeadNums &nums, const std::string &opName)
{
    OP_CHECK_IF(nums.qkN == 0 || nums.qkvN == 0,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: derived qkN=%lu qkvN=%lu.",
                        nums.qkN, nums.qkvN),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateDtype(
    const std::string &opName, const char *tensorName, const TensorContractInfo &tensor, ContractDtype expected)
{
    OP_CHECK_IF(!HasDtype(tensor, expected),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: %s dtype check failed, got %s(%u), "
                        "expected %s(%u).",
                        tensorName, ContractDtypeName(tensor.dtype), static_cast<uint32_t>(tensor.dtype),
                        ContractDtypeName(expected), static_cast<uint32_t>(expected)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateRequiredDtypes(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(ValidateDtype(opName, "qkv", input.qkv, ContractDtype::kBf16) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale qkv dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "qGamma", input.qGamma, ContractDtype::kFloat) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale qGamma dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "kGamma", input.kGamma, ContractDtype::kFloat) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale kGamma dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "cosSin", input.cosSin, ContractDtype::kFloat) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale cosSin dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "slotMapping", input.slotMapping, ContractDtype::kInt32) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale slotMapping dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "cuSeqLen", input.cuSeqLen, ContractDtype::kInt32) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale cuSeqLen dtype validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateCacheDtypes(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(ValidateDtype(opName, "kCache", input.kCache, ContractDtype::kFloat8E4M3fn) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale kCache dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "vCache", input.vCache, ContractDtype::kFloat8E4M3fn) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale vCache dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "kScaleCache", input.kScaleCache, ContractDtype::kFloat) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale kScaleCache dtype validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateOptionalDtypes(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(ValidateDtype(opName, "qRotation", input.qRotation, ContractDtype::kBf16) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale qRotation dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "kRotation", input.kRotation, ContractDtype::kBf16) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale kRotation dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(opName, "vScale", input.vScale, ContractDtype::kFloat) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale vScale dtype validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateDtypes(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(ValidateRequiredDtypes(input, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale required dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateCacheDtypes(input, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale cache dtype validation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateOptionalDtypes(input, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale optional dtype validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateShapes(
    const ContractInput &input, const DerivedHeadNums &nums, const std::string &opName)
{
    OP_CHECK_IF(!IsShape3D(input.qkv.shape, nums.qkvN, input.totalTokens, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: qkv shape rank=%u dims={%lu,%lu,%lu,%lu}.",
                        input.qkv.shape.rank, input.qkv.shape.dim0, input.qkv.shape.dim1, input.qkv.shape.dim2,
                        input.qkv.shape.dim3),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.qGamma.shape, input.headDim) || !IsShape1D(input.kGamma.shape, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: gamma shape qGamma rank=%u dim0=%lu, "
                        "kGamma rank=%u dim0=%lu.",
                        input.qGamma.shape.rank, input.qGamma.shape.dim0, input.kGamma.shape.rank,
                        input.kGamma.shape.dim0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape2D(input.cosSin.shape, input.maxSeqLen, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cosSin shape rank=%u dims={%lu,%lu}.",
                        input.cosSin.shape.rank, input.cosSin.shape.dim0, input.cosSin.shape.dim1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.slotMapping.shape, input.totalTokens),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: slotMapping shape rank=%u dim0=%lu.",
                        input.slotMapping.shape.rank, input.slotMapping.shape.dim0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.cuSeqLen.shape, static_cast<uint64_t>(input.batch) + 1),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen shape rank=%u dim0=%lu.",
                        input.cuSeqLen.shape.rank, input.cuSeqLen.shape.dim0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape2D(input.qRotation.shape, input.headDim, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: qRotation shape rank=%u dims={%lu,%lu}.",
                        input.qRotation.shape.rank, input.qRotation.shape.dim0, input.qRotation.shape.dim1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape2D(input.kRotation.shape, input.headDim, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kRotation shape rank=%u dims={%lu,%lu}.",
                        input.kRotation.shape.rank, input.kRotation.shape.dim0, input.kRotation.shape.dim1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.vScale.shape, input.numVHeads),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: vScale shape rank=%u dim0=%lu.",
                        input.vScale.shape.rank, input.vScale.shape.dim0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape4D(input.kCache.shape, input.blockNum, input.numKHeads, input.blockSize, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache shape rank=%u "
                        "dims={%lu,%lu,%lu,%lu}, expected={%lu,%lu,%lu,%lu}.",
                        input.kCache.shape.rank, input.kCache.shape.dim0, input.kCache.shape.dim1,
                        input.kCache.shape.dim2, input.kCache.shape.dim3, input.blockNum, input.numKHeads,
                        input.blockSize, input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape4D(input.vCache.shape, input.blockNum, input.numVHeads, input.blockSize, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: vCache shape rank=%u "
                        "dims={%lu,%lu,%lu,%lu}, expected={%lu,%lu,%lu,%lu}.",
                        input.vCache.shape.rank, input.vCache.shape.dim0, input.vCache.shape.dim1,
                        input.vCache.shape.dim2, input.vCache.shape.dim3, input.blockNum, input.numVHeads,
                        input.blockSize, input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape4D(input.kScaleCache.shape, input.blockNum, input.numKHeads, input.blockSize, 1),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache shape rank=%u "
                        "dims={%lu,%lu,%lu,%lu}.",
                        input.kScaleCache.shape.rank, input.kScaleCache.shape.dim0, input.kScaleCache.shape.dim1,
                        input.kScaleCache.shape.dim2, input.kScaleCache.shape.dim3),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateStrides(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(!IsCacheStrideValid(input.kCache.stride, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache stride present=%u rank=%u "
                        "strides={%lu,%lu,%lu,%lu}, headDim=%lu.",
                        input.kCache.stride.present ? 1U : 0U, input.kCache.stride.rank,
                        input.kCache.stride.stride0, input.kCache.stride.stride1, input.kCache.stride.stride2,
                        input.kCache.stride.stride3, input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsCacheStrideValid(input.vCache.stride, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: vCache stride present=%u rank=%u "
                        "strides={%lu,%lu,%lu,%lu}, headDim=%lu.",
                        input.vCache.stride.present ? 1U : 0U, input.vCache.stride.rank,
                        input.vCache.stride.stride0, input.vCache.stride.stride1, input.vCache.stride.stride2,
                        input.vCache.stride.stride3, input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CacheNonInnermostStridesMatch(input.kCache.stride, input.vCache.stride),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: k/v cache non-innermost strides mismatch "
                        "k={%lu,%lu,%lu} v={%lu,%lu,%lu}.",
                        input.kCache.stride.stride0, input.kCache.stride.stride1, input.kCache.stride.stride2,
                        input.vCache.stride.stride0, input.vCache.stride.stride1, input.vCache.stride.stride2),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScaleStrideValid(input.kScaleCache.stride),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache stride present=%u rank=%u "
                        "strides={%lu,%lu,%lu,%lu}.",
                        input.kScaleCache.stride.present ? 1U : 0U, input.kScaleCache.stride.rank,
                        input.kScaleCache.stride.stride0, input.kScaleCache.stride.stride1,
                        input.kScaleCache.stride.stride2, input.kScaleCache.stride.stride3),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::TrySelectTokenTile(
    const ContractInput &input, const DerivedHeadNums &nums, uint64_t fp8NtdStagingHeads, uint64_t tokenTile,
    const std::string &opName, TokenTileSelection &selection)
{
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tokenTile, kAivPerAic);
    const uint64_t rowTile = tokenTile * nums.qkN;
    const uint64_t rowTileAligned = Ops::Base::CeilAlign(rowTile, 16UL);
    const uint64_t qkAlignedRows = CalcQkAlignedRows(tokenTile, input.numQHeads, input.numKHeads);
    const bool fitsUb = tokenTilePerAiv * nums.qkN <= kUbOutputRowsPerAiv &&
                        tokenTilePerAiv * nums.qkvN <= kUbInputRowsPerAiv &&
                        tokenTilePerAiv * fp8NtdStagingHeads <= kFp8NtdStagingRowsPerAiv;
    const bool fitsL0a = qkAlignedRows <= kL0aMaxRows;
    const bool fitsL0c = rowTile <= kL0cMaxRows && rowTileAligned <= kL0cMaxRows && qkAlignedRows <= kL0cMaxRows;
    const bool accepted = fitsUb && fitsL0a && fitsL0c;
    LogTokenTileCandidate(opName, tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb,
                          fitsL0a, fitsL0c, accepted);
    if (accepted) {
        selection.tokenTile = tokenTile;
        selection.tokenTilePerAiv = tokenTilePerAiv;
        return true;
    }
    return false;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::SelectTokenTile(
    const ContractInput &input, const DerivedHeadNums &nums, const std::string &opName,
    TokenTileSelection &selection)
{
    const uint64_t fp8NtdStagingHeads = std::max({input.numQHeads, input.numKHeads, input.numVHeads});
    const uint64_t initialTokenTilePerAiv =
        std::min({kMaxTokenTile / 2, kUbOutputRowsPerAiv / nums.qkN, kUbInputRowsPerAiv / nums.qkvN,
            kFp8NtdStagingRowsPerAiv / fp8NtdStagingHeads});
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale token tile search start: initialTokenTilePerAiv=%lu "
            "ubOutputLimit=%lu ubInputLimit=%lu fp8NtdStagingLimit=%lu.",
            initialTokenTilePerAiv, kUbOutputRowsPerAiv / nums.qkN, kUbInputRowsPerAiv / nums.qkvN,
            kFp8NtdStagingRowsPerAiv / fp8NtdStagingHeads);

    uint64_t searchTokenTilePerAiv = initialTokenTilePerAiv;
    while (searchTokenTilePerAiv > 0) {
        if (TrySelectTokenTile(input, nums, fp8NtdStagingHeads, 2 * searchTokenTilePerAiv, opName, selection)) {
            break;
        }
        --searchTokenTilePerAiv;
    }
    if (selection.tokenTile == 0) {
        (void)TrySelectTokenTile(input, nums, fp8NtdStagingHeads, 1, opName, selection);
    }
    OP_CHECK_IF(selection.tokenTile == 0,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: no token tile candidate fits."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillContractTiling(
    const ContractInput &input, const DerivedHeadNums &nums, const TokenTileSelection &selection,
    ContractTiling &tiling)
{
    tiling.coreTokenTile = Ops::Base::CeilDiv(input.totalTokens, input.aicNum);
    tiling.coreGroupNum = Ops::Base::CeilDiv(input.totalTokens, tiling.coreTokenTile);
    tiling.tokenTilePerAiv = selection.tokenTilePerAiv;
    tiling.tokenTile = selection.tokenTile;
    tiling.qkHeadTile = nums.qkN;
    tiling.vHeadTile = input.numVHeads;
    tiling.dimTile = kDimTileFast;
    tiling.rowTile = tiling.tokenTile * nums.qkN;
    tiling.rowTileAligned = Ops::Base::CeilAlign(tiling.rowTile, 16UL);
    tiling.kCacheStrideBlock = input.kCache.stride.stride0;
    tiling.kCacheStrideHead = input.kCache.stride.stride1;
    tiling.kCacheStrideToken = input.kCache.stride.stride2;
    tiling.vCacheStrideBlock = input.vCache.stride.stride0;
    tiling.vCacheStrideHead = input.vCache.stride.stride1;
    tiling.vCacheStrideToken = input.vCache.stride.stride2;
    tiling.kScaleCacheStrideBlock = input.kScaleCache.stride.stride0;
    tiling.kScaleCacheStrideHead = input.kScaleCache.stride.stride1;
    tiling.kScaleCacheStrideToken = input.kScaleCache.stride.stride2;
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogContractTilingResult(
    const std::string &opName, const ContractTiling &tiling)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract tiling result: coreTokenTile=%lu coreGroupNum=%lu "
            "tokenTile=%lu tokenTilePerAiv=%lu qkHeadTile=%lu vHeadTile=%lu rowTile=%lu rowTileAligned=%lu "
            "dimTile=%lu.",
            tiling.coreTokenTile, tiling.coreGroupNum, tiling.tokenTile, tiling.tokenTilePerAiv, tiling.qkHeadTile,
            tiling.vHeadTile, tiling.rowTile, tiling.rowTileAligned, tiling.dimTile);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract stride result: kCache={%lu,%lu,%lu} "
            "vCache={%lu,%lu,%lu} kScaleCache={%lu,%lu,%lu}.",
            tiling.kCacheStrideBlock, tiling.kCacheStrideHead, tiling.kCacheStrideToken, tiling.vCacheStrideBlock,
            tiling.vCacheStrideHead, tiling.vCacheStrideToken, tiling.kScaleCacheStrideBlock,
            tiling.kScaleCacheStrideHead, tiling.kScaleCacheStrideToken);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateAndBuildTiling(
    const ContractInput &input, ContractTiling &tiling, const std::string &opName)
{
    tiling = {};
    OP_LOGD(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract validation start.");
    ge::graphStatus status = ValidateScalarInputs(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale scalar validation failed."),
                return ge::GRAPH_FAILED);

    const DerivedHeadNums nums = CalcDerivedHeadNums(input);
    LogContractInputSummary(opName, input, nums.qkN, nums.qkvN);
    status = ValidateDerivedHeadNums(nums, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale derived head validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateDtypes(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale dtype validation failed."),
                return ge::GRAPH_FAILED);
    status = ValidateShapes(input, nums, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale shape validation failed."),
                return ge::GRAPH_FAILED);
    status = ValidateStrides(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale stride validation failed."),
                return ge::GRAPH_FAILED);

    TokenTileSelection selection;
    status = SelectTokenTile(input, nums, opName, selection);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale token tile selection failed."),
                return ge::GRAPH_FAILED);

    FillContractTiling(input, nums, selection, tiling);
    LogContractTilingResult(opName, tiling);
    return ge::GRAPH_SUCCESS;
}

AivSlice QkvRmsNormRopeCacheWithKScaleTiling::MakeAivSlice(
    const TileParam &tile, uint64_t qkHeadTile, uint64_t aivLocalId)
{
    AivSlice slice;
    const uint64_t cubeHalfTokenSize = Ops::Base::CeilDiv(tile.tokenSize, kAivPerAic);
    const uint64_t tokenPrefix = aivLocalId == 0 ? 0 : cubeHalfTokenSize;
    const uint64_t tokenSize = aivLocalId == 0 ? cubeHalfTokenSize : tile.tokenSize - cubeHalfTokenSize;
    slice.tokenOffset = tile.tokenOffset + tokenPrefix;
    slice.tokenSize = tokenSize;
    slice.rowOffset = tokenPrefix * qkHeadTile;
    slice.rowSize = tokenSize * qkHeadTile;
    return slice;
}

TensorShapeInfo QkvRmsNormRopeCacheWithKScaleTiling::ToShapeInfo(const gert::Shape &shape)
{
    TensorShapeInfo info;
    info.rank = static_cast<uint32_t>(shape.GetDimNum());
    if (info.rank > 0) {
        info.dim0 = static_cast<uint64_t>(shape.GetDim(0));
    }
    if (info.rank > 1) {
        info.dim1 = static_cast<uint64_t>(shape.GetDim(1));
    }
    if (info.rank > 2) {
        info.dim2 = static_cast<uint64_t>(shape.GetDim(2));
    }
    if (info.rank > 3) {
        info.dim3 = static_cast<uint64_t>(shape.GetDim(3));
    }
    return info;
}

TensorShapeInfo QkvRmsNormRopeCacheWithKScaleTiling::ToShapeInfo(const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        TensorShapeInfo info;
        info.present = false;
        return info;
    }
    return ToShapeInfo(shape->GetStorageShape());
}

TensorStrideInfo QkvRmsNormRopeCacheWithKScaleTiling::ToStrideInfo(const gert::Stride &stride)
{
    TensorStrideInfo info;
    info.rank = static_cast<uint32_t>(stride.GetDimNum());
    if (info.rank > 0) {
        info.stride0 = static_cast<uint64_t>(stride.GetStride(0));
    }
    if (info.rank > 1) {
        info.stride1 = static_cast<uint64_t>(stride.GetStride(1));
    }
    if (info.rank > 2) {
        info.stride2 = static_cast<uint64_t>(stride.GetStride(2));
    }
    if (info.rank > 3) {
        info.stride3 = static_cast<uint64_t>(stride.GetStride(3));
    }
    return info;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::HasUsableBlockDims(const TensorShapeInfo &shape)
{
    return shape.rank >= 4 && shape.dim0 > 0 && shape.dim2 > 0;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::BlockDimsMatch(const TensorShapeInfo &lhs, const TensorShapeInfo &rhs)
{
    return lhs.dim0 == rhs.dim0 && lhs.dim2 == rhs.dim2;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::MakeContiguousStrideInfo(
    const gert::Shape &shape, const std::string &opName, TensorStrideInfo &strideInfo)
{
    gert::Stride stride;
    stride.SetDimNum(shape.GetDimNum());
    int64_t currentStride = 1;
    for (int32_t dim = static_cast<int32_t>(shape.GetDimNum()) - 1; dim >= 0; --dim) {
        const int64_t dimValue = shape.GetDim(dim);
        OP_CHECK_IF(dimValue < 0,
                    OP_LOGE(opName.c_str(), "cache storage shape dim[%d]=%lld must be non-negative.", dim,
                            static_cast<long long>(dimValue)),
                    return ge::GRAPH_FAILED);
        stride.SetStride(dim, currentStride);
        OP_CHECK_IF(dimValue > 0 && currentStride > std::numeric_limits<int64_t>::max() / dimValue,
                    OP_LOGE(opName.c_str(),
                            "cache contiguous stride overflow: currentStride=%lld dim[%d]=%lld.",
                            static_cast<long long>(currentStride), dim, static_cast<long long>(dimValue)),
                    return ge::GRAPH_FAILED);
        currentStride *= dimValue;
    }
    strideInfo = ToStrideInfo(stride);
    return ge::GRAPH_SUCCESS;
}

ContractDtype QkvRmsNormRopeCacheWithKScaleTiling::ToContractDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16:
            return ContractDtype::kBf16;
        case ge::DT_FLOAT:
            return ContractDtype::kFloat;
        case ge::DT_INT32:
            return ContractDtype::kInt32;
        case ge::DT_FLOAT8_E4M3FN:
            return ContractDtype::kFloat8E4M3fn;
        default:
            return ContractDtype::kUnknown;
    }
}

const char *QkvRmsNormRopeCacheWithKScaleTiling::ContractDtypeName(ContractDtype dtype)
{
    switch (dtype) {
        case ContractDtype::kBf16:
            return "bf16";
        case ContractDtype::kFloat:
            return "float";
        case ContractDtype::kInt32:
            return "int32";
        case ContractDtype::kFloat8E4M3fn:
            return "float8_e4m3fn";
        default:
            return "unknown";
    }
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillInputDtype(
    gert::TilingContext *context, size_t index, TensorContractInfo &info)
{
    const auto desc = context->GetInputDesc(index);
    if (desc != nullptr) {
        info.dtype = ToContractDtype(desc->GetDataType());
    }
}

TensorContractInfo QkvRmsNormRopeCacheWithKScaleTiling::ToTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    info.shape = ToShapeInfo(context->GetInputShape(index));
    FillInputDtype(context, index, info);
    return info;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ToCacheTensorInfo(
    gert::TilingContext *context, const std::string &opName, size_t index, TensorContractInfo &info)
{
    const auto storageShape = context->GetInputShape(index);
    if (storageShape == nullptr) {
        info.shape.present = false;
        info.stride.present = false;
        FillInputDtype(context, index, info);
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu has no storage shape: "
                "shapePresent=0 stridePresent=0.",
                index);
        return ge::GRAPH_SUCCESS;
    }

    const TensorShapeInfo storageShapeInfo = ToShapeInfo(storageShape->GetStorageShape());
    const TensorShapeInfo logicalShapeInfo = ToShapeInfo(storageShape->GetShape());
    const bool inputIsView = context->InputIsView(index);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale cache input index=%zu InputIsView=%u storageRank=%u "
            "storageShape={%lu,%lu,%lu,%lu} logicalRank=%u logicalShape={%lu,%lu,%lu,%lu}.",
            index, inputIsView ? 1U : 0U, storageShapeInfo.rank, storageShapeInfo.dim0, storageShapeInfo.dim1,
            storageShapeInfo.dim2, storageShapeInfo.dim3, logicalShapeInfo.rank, logicalShapeInfo.dim0,
            logicalShapeInfo.dim1, logicalShapeInfo.dim2, logicalShapeInfo.dim3);

    if (inputIsView) {
        const auto inputStride = context->GetInputStride(index);
        const uint32_t inputStrideRank = inputStride == nullptr ? 0U : static_cast<uint32_t>(inputStride->GetDimNum());
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu view stride query: stridePresent=%u "
                "strideRank=%u.",
                index, inputStride == nullptr ? 0U : 1U, inputStrideRank);
        if (inputStride != nullptr && inputStride->GetDimNum() != 0) {
            info.shape = logicalShapeInfo;
            info.stride = ToStrideInfo(*inputStride);
            FillInputDtype(context, index, info);
            OP_LOGD(opName.c_str(),
                    "QkvRmsNormRopeCacheWithKScale cache input index=%zu uses view shape/stride: "
                    "shapeRank=%u shape={%lu,%lu,%lu,%lu} strideRank=%u stride={%lu,%lu,%lu,%lu}.",
                    index, info.shape.rank, info.shape.dim0, info.shape.dim1, info.shape.dim2, info.shape.dim3,
                    info.stride.rank, info.stride.stride0, info.stride.stride1, info.stride.stride2,
                    info.stride.stride3);
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu is view but stride is missing or empty; "
                "fall back to contiguous storage stride.",
                index);
    } else {
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu is not a view; use contiguous storage stride.",
                index);
    }

    const gert::Shape &shape = storageShape->GetStorageShape();
    info.shape = storageShapeInfo;
    OP_CHECK_IF(MakeContiguousStrideInfo(shape, opName, info.stride) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "make contiguous cache stride failed, input index=%zu.", index),
                return ge::GRAPH_FAILED);
    FillInputDtype(context, index, info);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale cache input index=%zu uses contiguous storage shape/stride: "
            "shapeRank=%u shape={%lu,%lu,%lu,%lu} strideRank=%u stride={%lu,%lu,%lu,%lu}.",
            index, info.shape.rank, info.shape.dim0, info.shape.dim1, info.shape.dim2, info.shape.dim3,
            info.stride.rank, info.stride.stride0, info.stride.stride1, info.stride.stride2, info.stride.stride3);
    return ge::GRAPH_SUCCESS;
}

TensorContractInfo QkvRmsNormRopeCacheWithKScaleTiling::ToOptionalTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    info.shape = ToShapeInfo(context->GetOptionalInputShape(index));
    const auto desc = context->GetOptionalInputDesc(index);
    if (desc != nullptr) {
        info.dtype = ToContractDtype(desc->GetDataType());
    }
    return info;
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogTensorInfo(
    const std::string &opName, const char *tensorName, const TensorContractInfo &info)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale input tensor %s: dtype=%s(%u) shapePresent=%u shapeRank=%u "
            "shape={%lu,%lu,%lu,%lu} stridePresent=%u strideRank=%u stride={%lu,%lu,%lu,%lu}.",
            tensorName, ContractDtypeName(info.dtype), static_cast<uint32_t>(info.dtype),
            info.shape.present ? 1U : 0U, info.shape.rank, info.shape.dim0, info.shape.dim1, info.shape.dim2,
            info.shape.dim3, info.stride.present ? 1U : 0U, info.stride.rank, info.stride.stride0,
            info.stride.stride1, info.stride.stride2, info.stride.stride3);
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogContractInput(
    const std::string &opName, const ContractInput &input, float epsilon)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale parsed attrs/input summary: head_nums={%lu,%lu,%lu} epsilon=%f "
            "T=%lu batch=%lu headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.numQHeads, input.numKHeads, input.numVHeads, epsilon, input.totalTokens, input.batch, input.headDim,
            input.maxSeqLen, input.blockNum, input.blockSize, input.aicNum);
    LogTensorInfo(opName, "qkv", input.qkv);
    LogTensorInfo(opName, "qGamma", input.qGamma);
    LogTensorInfo(opName, "kGamma", input.kGamma);
    LogTensorInfo(opName, "cosSin", input.cosSin);
    LogTensorInfo(opName, "slotMapping", input.slotMapping);
    LogTensorInfo(opName, "kCache", input.kCache);
    LogTensorInfo(opName, "vCache", input.vCache);
    LogTensorInfo(opName, "kScaleCache", input.kScaleCache);
    LogTensorInfo(opName, "cuSeqLen", input.cuSeqLen);
    LogTensorInfo(opName, "qRotation", input.qRotation);
    LogTensorInfo(opName, "kRotation", input.kRotation);
    LogTensorInfo(opName, "vScale", input.vScale);
}

size_t QkvRmsNormRopeCacheWithKScaleTiling::CalcOpWorkspaceSize(
    const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo)
{
    return static_cast<size_t>(compileInfo.opWorkspaceSize);
}

size_t QkvRmsNormRopeCacheWithKScaleTiling::CalcWorkspaceSize(size_t opWorkspaceSize)
{
    return opWorkspaceSize + kReservedWorkspaceSize;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::SetWorkspace(
    gert::TilingContext *context, const std::string &opName,
    const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo, size_t &workspaceSize)
{
    size_t *workspaces = context->GetWorkspaceSizes(kWorkspaceCount);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(opName.c_str(), "get workspace failed."), return ge::GRAPH_FAILED);
    const size_t opWorkspaceSize = CalcOpWorkspaceSize(compileInfo);
    workspaceSize = CalcWorkspaceSize(opWorkspaceSize);
    workspaces[0] = workspaceSize;
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale workspace: opWorkspaceSize=%zu reservedWorkspaceSize=%zu total=%zu.",
            opWorkspaceSize, kReservedWorkspaceSize, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogTilingData(
    const std::string &opName, const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
    const ContractTiling &contractTiling, uint64_t tilingKey, uint64_t blockDim, size_t dataSize,
    size_t workspaceSize)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final tilingData: totalTokens=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu "
            "headDim=%lu blockSize=%lu coreTokenTile=%lu coreGroupNum=%lu tokenTile=%lu epsilon=%f.",
            tilingData.totalTokens, tilingData.batch, tilingData.numQHeads, tilingData.numKHeads,
            tilingData.numVHeads, tilingData.headDim, tilingData.blockSize, tilingData.coreTokenTile,
            tilingData.coreGroupNum, tilingData.tokenTile, tilingData.epsilon);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final strides: kCache={%lu,%lu,%lu} vCache={%lu,%lu,%lu} "
            "kScaleCache={%lu,%lu,%lu}.",
            tilingData.kCacheStrideBlock, tilingData.kCacheStrideHead, tilingData.kCacheStrideToken,
            tilingData.vCacheStrideBlock, tilingData.vCacheStrideHead, tilingData.vCacheStrideToken,
            tilingData.kScaleCacheStrideBlock, tilingData.kScaleCacheStrideHead, tilingData.kScaleCacheStrideToken);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final launch: tilingKey=%lu blockDim=%lu tilingDataSize=%zu workspaceSize=%zu "
            "qkHeadTile=%lu vHeadTile=%lu rowTile=%lu rowTileAligned=%lu tokenTilePerAiv=%lu dimTile=%lu.",
            tilingKey, blockDim, dataSize, workspaceSize, contractTiling.qkHeadTile, contractTiling.vHeadTile,
            contractTiling.rowTile, contractTiling.rowTileAligned, contractTiling.tokenTilePerAiv,
            contractTiling.dimTile);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ParseHeadNumsAttr(
    gert::TilingContext *context, const std::string &opName, HeadNumsAttr &headNumsAttr)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(kHeadNumsAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    OP_CHECK_IF(headNums->GetSize() != 3, OP_LOGE(opName.c_str(), "head_nums must have exactly 3 values."),
                return ge::GRAPH_FAILED);
    const auto *headNumsData = headNums->GetData();
    for (size_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(headNumsData[i] <= 0, OP_LOGE(opName.c_str(), "head_nums[%zu] must be positive.", i),
                    return ge::GRAPH_FAILED);
    }

    headNumsAttr.numQHeads = static_cast<uint64_t>(headNumsData[0]);
    headNumsAttr.numKHeads = static_cast<uint64_t>(headNumsData[1]);
    headNumsAttr.numVHeads = static_cast<uint64_t>(headNumsData[2]);
    return ge::GRAPH_SUCCESS;
}

float QkvRmsNormRopeCacheWithKScaleTiling::ParseEpsilonAttr(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const float *epsilonAttr = attrs == nullptr ? nullptr : attrs->GetFloat(kEpsilonAttrIndex);
    return epsilonAttr == nullptr ? kDefaultEpsilon : *epsilonAttr;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillShapeDerivedFields(
    const TensorContractInfo &qkv, const TensorContractInfo &cosSin, const TensorContractInfo &cuSeqLen,
    ContractInput &input)
{
    const auto &qkvShape = qkv.shape;
    const auto &cosSinShape = cosSin.shape;
    const auto &cuSeqLenShape = cuSeqLen.shape;
    if (qkvShape.rank >= 3) {
        input.totalTokens = qkvShape.dim1;
        input.headDim = qkvShape.dim2;
    }
    if (cuSeqLenShape.rank == 1 && cuSeqLenShape.dim0 > 0) {
        input.batch = cuSeqLenShape.dim0 - 1;
    }
    if (cosSinShape.rank >= 1) {
        input.maxSeqLen = cosSinShape.dim0;
    }
}

void QkvRmsNormRopeCacheWithKScaleTiling::DeriveBlockShape(const std::string &opName, ContractInput &input)
{
    const TensorShapeInfo *blockShape = nullptr;
    const char *blockShapeSource = "none";
    const bool kCacheHasBlockDims = HasUsableBlockDims(input.kCache.shape);
    const bool vCacheHasBlockDims = HasUsableBlockDims(input.vCache.shape);
    const bool kScaleCacheHasBlockDims = HasUsableBlockDims(input.kScaleCache.shape);
    if (kCacheHasBlockDims && vCacheHasBlockDims && BlockDimsMatch(input.kCache.shape, input.vCache.shape)) {
        blockShape = &input.kCache.shape;
        blockShapeSource = "kCache/vCache";
    } else if (kCacheHasBlockDims && kScaleCacheHasBlockDims &&
               BlockDimsMatch(input.kCache.shape, input.kScaleCache.shape)) {
        blockShape = &input.kCache.shape;
        blockShapeSource = "kCache/kScaleCache";
    } else if (vCacheHasBlockDims && kScaleCacheHasBlockDims &&
               BlockDimsMatch(input.vCache.shape, input.kScaleCache.shape)) {
        blockShape = &input.vCache.shape;
        blockShapeSource = "vCache/kScaleCache";
    } else if (kCacheHasBlockDims) {
        blockShape = &input.kCache.shape;
        blockShapeSource = "kCache";
    } else if (vCacheHasBlockDims) {
        blockShape = &input.vCache.shape;
        blockShapeSource = "vCache";
    } else if (kScaleCacheHasBlockDims) {
        blockShape = &input.kScaleCache.shape;
        blockShapeSource = "kScaleCache";
    }
    if (blockShape != nullptr) {
        input.blockNum = blockShape->dim0;
        input.blockSize = blockShape->dim2;
    }
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale derived block shape: source=%s blockNum=%lu blockSize=%lu "
            "hasBlockDims={k:%u,v:%u,kScale:%u}.",
            blockShapeSource, input.blockNum, input.blockSize, kCacheHasBlockDims ? 1U : 0U,
            vCacheHasBlockDims ? 1U : 0U, kScaleCacheHasBlockDims ? 1U : 0U);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::FillRequiredTensorInputs(
    gert::TilingContext *context, const std::string &opName, ContractInput &input)
{
    input.qkv = ToTensorInfo(context, kQkvIndex);
    input.qGamma = ToTensorInfo(context, kQGammaIndex);
    input.kGamma = ToTensorInfo(context, kKGammaIndex);
    input.cosSin = ToTensorInfo(context, kCosSinIndex);
    input.slotMapping = ToTensorInfo(context, kSlotMappingIndex);
    OP_CHECK_IF(ToCacheTensorInfo(context, opName, kKCacheIndex, input.kCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse kCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context, opName, kVCacheIndex, input.vCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse vCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context, opName, kKScaleCacheIndex, input.kScaleCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse kScaleCache tensor info."), return ge::GRAPH_FAILED);
    input.cuSeqLen = ToTensorInfo(context, kCuSeqLenIndex);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillOptionalTensorInputs(gert::TilingContext *context, ContractInput &input)
{
    input.qRotation = ToOptionalTensorInfo(context, kQRotationIndex);
    input.kRotation = ToOptionalTensorInfo(context, kKRotationIndex);
    input.vScale = ToOptionalTensorInfo(context, kVScaleIndex);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::BuildContractInput(
    gert::TilingContext *context, const std::string &opName, ContractInput &input, float &epsilon)
{
    HeadNumsAttr headNumsAttr;
    OP_CHECK_IF(ParseHeadNumsAttr(context, opName, headNumsAttr) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse head_nums attr."), return ge::GRAPH_FAILED);

    input.numQHeads = headNumsAttr.numQHeads;
    input.numKHeads = headNumsAttr.numKHeads;
    input.numVHeads = headNumsAttr.numVHeads;
    OP_CHECK_IF(FillRequiredTensorInputs(context, opName, input) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse required tensor inputs."), return ge::GRAPH_FAILED);

    FillShapeDerivedFields(input.qkv, input.cosSin, input.cuSeqLen, input);
    DeriveBlockShape(opName, input);
    FillOptionalTensorInputs(context, input);
    epsilon = ParseEpsilonAttr(context);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillTilingData(
    const ContractInput &input, const ContractTiling &contractTiling, float epsilon,
    QkvRmsNormRopeCacheWithKScaleTilingData &tilingData)
{
    tilingData.totalTokens = input.totalTokens;
    tilingData.batch = input.batch;
    tilingData.numQHeads = input.numQHeads;
    tilingData.numKHeads = input.numKHeads;
    tilingData.numVHeads = input.numVHeads;
    tilingData.headDim = input.headDim;
    tilingData.blockSize = input.blockSize;
    tilingData.coreTokenTile = contractTiling.coreTokenTile;
    tilingData.coreGroupNum = contractTiling.coreGroupNum;
    tilingData.kCacheStrideBlock = contractTiling.kCacheStrideBlock;
    tilingData.kCacheStrideHead = contractTiling.kCacheStrideHead;
    tilingData.kCacheStrideToken = contractTiling.kCacheStrideToken;
    tilingData.vCacheStrideBlock = contractTiling.vCacheStrideBlock;
    tilingData.vCacheStrideHead = contractTiling.vCacheStrideHead;
    tilingData.vCacheStrideToken = contractTiling.vCacheStrideToken;
    tilingData.kScaleCacheStrideBlock = contractTiling.kScaleCacheStrideBlock;
    tilingData.kScaleCacheStrideHead = contractTiling.kScaleCacheStrideHead;
    tilingData.kScaleCacheStrideToken = contractTiling.kScaleCacheStrideToken;
    tilingData.tokenTile = contractTiling.tokenTile;
    tilingData.epsilon = epsilon;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateCompileInfo(
    const std::string &opName, const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale compile info: aicNum=%u aivNum=%u ubSize=%lu l1Size=%lu l0cSize=%lu "
            "opWorkspaceSize=%lu.",
            compileInfo.aicNum, compileInfo.aivNum, compileInfo.ubSize, compileInfo.l1Size,
            compileInfo.l0cSize, compileInfo.opWorkspaceSize);
    OP_CHECK_IF(compileInfo.aicNum == 0, OP_LOGE(opName.c_str(), "aicNum in compile info must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::CopyRawTilingData(
    gert::TilingContext *context, const std::string &opName,
    const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData, size_t &tilingDataSize)
{
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData->GetData());
    tilingDataSize = sizeof(QkvRmsNormRopeCacheWithKScaleTilingData);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE(opName.c_str(), "tiling data size[%zu] is not aligned to 8.", tilingDataSize),
                return ge::GRAPH_FAILED);
    const size_t tilingCapacity = rawTilingData->GetCapacity();
    OP_CHECK_IF(tilingDataSize > tilingCapacity,
                OP_LOGE(opName.c_str(), "tiling data size[%zu] exceeds capacity[%zu].", tilingDataSize,
                        tilingCapacity),
                return ge::GRAPH_FAILED);
    const auto ret = memcpy_s(rawTilingData->GetData(), tilingCapacity,
                              reinterpret_cast<const void *>(&tilingData), tilingDataSize);
    OP_CHECK_IF(ret != EOK,
                OP_LOGE(opName.c_str(), "copy QkvRmsNormRopeCacheWithKScale tiling data failed, ret=%d.", ret),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::Reset(gert::TilingContext *context)
{
    TilingBaseClass::Reset(context);
    Reset();
}

void QkvRmsNormRopeCacheWithKScaleTiling::Reset()
{
    compileInfo_ = nullptr;
    input_ = {};
    contractTiling_ = {};
    tilingData_ = {};
    epsilon_ = kDefaultEpsilon;
    tilingDataSize_ = 0;
    numBlocks_ = 0;
    workspaceSize_ = 0;
    tilingKey_ = 0;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsCapable()
{
    return true;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::GetShapeAttrsInfo()
{
    const std::string opName = context_->GetNodeName();
    OP_CHECK_IF(BuildContractInput(context_, opName, input_, epsilon_) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to build QkvRmsNormRopeCacheWithKScale contract input."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::GetPlatformInfo()
{
    const std::string opName = context_->GetNodeName();
    compileInfo_ = context_->GetCompileInfo<QkvRmsNormRopeCacheWithKScaleCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);
    OP_CHECK_IF(ValidateCompileInfo(opName, *compileInfo_) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "invalid QkvRmsNormRopeCacheWithKScale compile info."),
                return ge::GRAPH_FAILED);
    input_.aicNum = compileInfo_->aicNum;
    LogContractInput(opName, input_, epsilon_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::DoOpTiling()
{
    const std::string opName = context_->GetNodeName();
    OP_CHECK_IF(ValidateAndBuildTiling(input_, contractTiling_, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract validation failed."),
                return ge::GRAPH_FAILED);
    FillTilingData(input_, contractTiling_, epsilon_, tilingData_);
    numBlocks_ = static_cast<uint32_t>(contractTiling_.coreGroupNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t QkvRmsNormRopeCacheWithKScaleTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::GetWorkspaceSize()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);
    OP_CHECK_IF(SetWorkspace(context_, context_->GetNodeName(), *compileInfo_, workspaceSize_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "set QkvRmsNormRopeCacheWithKScale workspace failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::PostTiling()
{
    OP_CHECK_IF(CopyRawTilingData(context_, context_->GetNodeName(), tilingData_, tilingDataSize_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "copy QkvRmsNormRopeCacheWithKScale raw tiling data failed."),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(numBlocks_);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::DumpTilingInfo()
{
    LogTilingData(context_->GetNodeName(), tilingData_, contractTiling_, context_->GetTilingKey(), numBlocks_,
                  tilingDataSize_, workspaceSize_);
}

REGISTER_OPS_TILING_TEMPLATE(
    QkvRmsNormRopeCacheWithKScale, QkvRmsNormRopeCacheWithKScaleTiling, kTilingTemplatePriority);

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling
