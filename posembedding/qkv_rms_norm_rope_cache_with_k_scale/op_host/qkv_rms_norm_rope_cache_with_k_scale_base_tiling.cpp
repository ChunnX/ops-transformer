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
constexpr size_t QKV_INDEX = 0;
constexpr size_t Q_GAMMA_INDEX = 1;
constexpr size_t K_GAMMA_INDEX = 2;
constexpr size_t COS_SIN_INDEX = 3;
constexpr size_t SLOT_MAPPING_INDEX = 4;
constexpr size_t K_CACHE_INDEX = 5;
constexpr size_t V_CACHE_INDEX = 6;
constexpr size_t K_SCALE_CACHE_INDEX = 7;
constexpr size_t CU_SEQ_LEN_INDEX = 8;
constexpr size_t Q_ROTATION_INDEX = 9;
constexpr size_t K_ROTATION_INDEX = 10;
constexpr size_t V_SCALE_INDEX = 11;

constexpr size_t HEAD_NUMS_ATTR_INDEX = 0;
constexpr size_t EPSILON_ATTR_INDEX = 1;

constexpr float DEFAULT_EPSILON = 1e-6f;
constexpr int32_t TILING_TEMPLATE_PRIORITY = 1000;
constexpr size_t WORKSPACE_COUNT = 1;
constexpr size_t RESERVED_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

uint32_t ShapeRank(const gert::Shape &shape)
{
    return static_cast<uint32_t>(shape.GetDimNum());
}

uint64_t ShapeDim(const gert::Shape &shape, size_t index)
{
    return static_cast<uint64_t>(shape.GetDim(index));
}

uint64_t SafeShapeDim(const gert::Shape &shape, size_t index)
{
    return index < static_cast<size_t>(shape.GetDimNum()) ? ShapeDim(shape, index) : 0UL;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape1D(const TensorContractInfo &tensor, uint64_t d0)
{
    return tensor.shapePresent && tensor.shape.GetDimNum() == 1 && ShapeDim(tensor.shape, 0) == d0;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape2D(const TensorContractInfo &tensor, uint64_t d0, uint64_t d1)
{
    return tensor.shapePresent && tensor.shape.GetDimNum() == 2 && ShapeDim(tensor.shape, 0) == d0 &&
           ShapeDim(tensor.shape, 1) == d1;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape3D(const TensorContractInfo &tensor, uint64_t d0, uint64_t d1,
                                                    uint64_t d2)
{
    return tensor.shapePresent && tensor.shape.GetDimNum() == 3 && ShapeDim(tensor.shape, 0) == d0 &&
           ShapeDim(tensor.shape, 1) == d1 && ShapeDim(tensor.shape, 2) == d2;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::IsShape4D(const TensorContractInfo &tensor, uint64_t d0, uint64_t d1,
                                                    uint64_t d2,
                                                    uint64_t d3)
{
    return tensor.shapePresent && tensor.shape.GetDimNum() == 4 && ShapeDim(tensor.shape, 0) == d0 &&
           ShapeDim(tensor.shape, 1) == d1 && ShapeDim(tensor.shape, 2) == d2 && ShapeDim(tensor.shape, 3) == d3;
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

bool QkvRmsNormRopeCacheWithKScaleTiling::CacheNonInnermostStridesMatch(const TensorStrideInfo &lhs,
                                                                        const TensorStrideInfo &rhs)
{
    return lhs.stride0 == rhs.stride0 && lhs.stride1 == rhs.stride1 && lhs.stride2 == rhs.stride2;
}

uint64_t QkvRmsNormRopeCacheWithKScaleTiling::CalcQkAlignedRows(uint64_t tokenTile, uint64_t numQHeads,
                                                                uint64_t numKHeads)
{
    return Ops::Base::CeilAlign(tokenTile * numQHeads, 16UL) + Ops::Base::CeilAlign(tokenTile * numKHeads, 16UL);
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogContractInputSummary(const std::string &opName, const ContractInput &input,
                                                                  uint64_t aicNum, const ContractTiling &tiling)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract input: T=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu Nqk=%lu Nqkv=%lu "
            "headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.totalTokens, input.batch, tiling.numQHeads, tiling.numKHeads, tiling.numVHeads, tiling.numQkHeads,
            tiling.numQkvHeads, input.headDim, input.maxSeqLen, input.blockNum, input.blockSize, aicNum);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale resource constants: maxTokenTile=%lu qkvInputRowsPerAiv=%lu "
            "qkOutputRowsPerAiv=%lu vOutputRowsPerAiv=%lu l0aFullDimSegmentRows=%lu l0cMaxRows=%lu "
            "aivPerAic=%lu.",
            MAX_TOKEN_TILE, QKV_INPUT_ROWS_PER_AIV, QK_OUTPUT_ROWS_PER_AIV, V_OUTPUT_ROWS_PER_AIV,
            L0A_FULL_DIM_SEGMENT_ROWS, L0C_MAX_ROWS, AIV_PER_AIC);
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogTokenTileCandidate(const std::string &opName, uint64_t tokenTilePerAiv,
                                                                uint64_t tokenTile, uint64_t rowTile,
                                                                uint64_t rowTileAligned, uint64_t qkAlignedRows,
                                                                bool fitsUb, bool fitsL0c, bool accepted)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale token tile candidate: tokenTilePerAiv=%lu tokenTile=%lu rowTile=%lu "
            "rowTileAligned=%lu qkAlignedRows=%lu fitsUb=%u fitsSideBySideL0c=%u accepted=%u.",
            tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb ? 1U : 0U,
            fitsL0c ? 1U : 0U, accepted ? 1U : 0U);
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillContractHeadNums(const ContractInput &input, ContractTiling &tiling)
{
    tiling.numQHeads = input.numQHeads;
    tiling.numKHeads = input.numKHeads;
    tiling.numVHeads = input.numVHeads;
    tiling.numQkHeads = tiling.numQHeads + tiling.numKHeads;
    tiling.numQkvHeads = tiling.numQkHeads + tiling.numVHeads;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateHeadNums(const ContractInput &input,
                                                                      const std::string &opName)
{
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
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateScalarInputs(const ContractInput &input,
                                                                          const std::string &opName)
{
    OP_CHECK_IF(input.totalTokens == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: totalTokens is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.batch == 0, OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: batch is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.headDim != HEAD_DIM_FAST,
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
    OP_CHECK_IF(!input.qRotation.shapePresent,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: qRotation is missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.kRotation.shapePresent,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: kRotation is missing."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.vScale.shapePresent,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: vScale is missing."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateDtype(const std::string &opName, const char *tensorName,
                                                                   const TensorContractInfo &tensor,
                                                                   ge::DataType expected)
{
    OP_CHECK_IF(tensor.dtype != expected,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: %s dtype check failed, got %s(%u), "
                        "expected %s(%u).",
                        tensorName, DtypeName(tensor.dtype), static_cast<uint32_t>(tensor.dtype), DtypeName(expected),
                        static_cast<uint32_t>(expected)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateDtypes(const ContractInput &input,
                                                                    const std::string &opName)
{
    struct DtypeRule {
        const char *tensorName;
        const TensorContractInfo *tensor;
        ge::DataType expected;
    };
    const DtypeRule rules[] = {
        {"qkv", &input.qkv, ge::DT_BF16},
        {"qGamma", &input.qGamma, ge::DT_FLOAT},
        {"kGamma", &input.kGamma, ge::DT_FLOAT},
        {"cosSin", &input.cosSin, ge::DT_FLOAT},
        {"slotMapping", &input.slotMapping, ge::DT_INT32},
        {"cuSeqLen", &input.cuSeqLen, ge::DT_INT32},
        {"kCache", &input.kCache, ge::DT_FLOAT8_E4M3FN},
        {"vCache", &input.vCache, ge::DT_FLOAT8_E4M3FN},
        {"kScaleCache", &input.kScaleCache, ge::DT_FLOAT},
        {"qRotation", &input.qRotation, ge::DT_BF16},
        {"kRotation", &input.kRotation, ge::DT_BF16},
        {"vScale", &input.vScale, ge::DT_FLOAT},
    };
    for (const auto &rule : rules) {
        OP_CHECK_IF(
            ValidateDtype(opName, rule.tensorName, *rule.tensor, rule.expected) != ge::GRAPH_SUCCESS,
            OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale %s dtype validation failed.", rule.tensorName),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateShapeInputsForDerivedFields(
    const ContractInput &input, const std::string &opName)
{
    OP_CHECK_IF(!input.qkv.shapePresent || input.qkv.shape.GetDimNum() != 3,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: qkv present=%u rank=%u, expected rank=3.",
                        input.qkv.shapePresent ? 1U : 0U, ShapeRank(input.qkv.shape)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.cosSin.shapePresent || input.cosSin.shape.GetDimNum() != 2,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cosSin present=%u rank=%u, expected rank=2.",
                        input.cosSin.shapePresent ? 1U : 0U, ShapeRank(input.cosSin.shape)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.cuSeqLen.shapePresent || input.cuSeqLen.shape.GetDimNum() != 1 ||
                    ShapeDim(input.cuSeqLen.shape, 0) == 0,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen present=%u rank=%u dim0=%lu, "
                        "expected rank=1 and dim0>0.",
                        input.cuSeqLen.shapePresent ? 1U : 0U, ShapeRank(input.cuSeqLen.shape),
                        SafeShapeDim(input.cuSeqLen.shape, 0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.kCache.shapePresent || input.kCache.shape.GetDimNum() != 4,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache present=%u rank=%u, expected rank=4.",
                        input.kCache.shapePresent ? 1U : 0U, ShapeRank(input.kCache.shape)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateShapes(const ContractInput &input,
                                                                    const std::string &opName)
{
    const uint64_t numQkHeads = input.numQHeads + input.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input.numVHeads;
    OP_CHECK_IF(!IsShape3D(input.qkv, numQkvHeads, input.totalTokens, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: qkv shape rank=%u dims={%lu,%lu,%lu,%lu}.",
                        ShapeRank(input.qkv.shape), SafeShapeDim(input.qkv.shape, 0), SafeShapeDim(input.qkv.shape, 1),
                        SafeShapeDim(input.qkv.shape, 2), SafeShapeDim(input.qkv.shape, 3)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.qGamma, input.headDim) || !IsShape1D(input.kGamma, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: gamma shape qGamma rank=%u dim0=%lu, "
                        "kGamma rank=%u dim0=%lu.",
                        ShapeRank(input.qGamma.shape), SafeShapeDim(input.qGamma.shape, 0),
                        ShapeRank(input.kGamma.shape), SafeShapeDim(input.kGamma.shape, 0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape2D(input.cosSin, input.maxSeqLen, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cosSin shape rank=%u dims={%lu,%lu}.",
                        ShapeRank(input.cosSin.shape), SafeShapeDim(input.cosSin.shape, 0),
                        SafeShapeDim(input.cosSin.shape, 1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.slotMapping, input.totalTokens),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: slotMapping shape rank=%u dim0=%lu.",
                        ShapeRank(input.slotMapping.shape), SafeShapeDim(input.slotMapping.shape, 0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.cuSeqLen, static_cast<uint64_t>(input.batch) + 1),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen shape rank=%u dim0=%lu.",
                        ShapeRank(input.cuSeqLen.shape), SafeShapeDim(input.cuSeqLen.shape, 0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape2D(input.qRotation, input.headDim, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: qRotation shape rank=%u dims={%lu,%lu}.",
                        ShapeRank(input.qRotation.shape), SafeShapeDim(input.qRotation.shape, 0),
                        SafeShapeDim(input.qRotation.shape, 1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape2D(input.kRotation, input.headDim, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kRotation shape rank=%u dims={%lu,%lu}.",
                        ShapeRank(input.kRotation.shape), SafeShapeDim(input.kRotation.shape, 0),
                        SafeShapeDim(input.kRotation.shape, 1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape1D(input.vScale, input.numVHeads),
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: vScale shape rank=%u dim0=%lu.",
                        ShapeRank(input.vScale.shape), SafeShapeDim(input.vScale.shape, 0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape4D(input.kCache, input.blockNum, input.numKHeads, input.blockSize, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache shape rank=%u "
                        "dims={%lu,%lu,%lu,%lu}, expected={%lu,%lu,%lu,%lu}.",
                        ShapeRank(input.kCache.shape), SafeShapeDim(input.kCache.shape, 0),
                        SafeShapeDim(input.kCache.shape, 1), SafeShapeDim(input.kCache.shape, 2),
                        SafeShapeDim(input.kCache.shape, 3), input.blockNum, input.numKHeads, input.blockSize,
                        input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape4D(input.vCache, input.blockNum, input.numVHeads, input.blockSize, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: vCache shape rank=%u "
                        "dims={%lu,%lu,%lu,%lu}, expected={%lu,%lu,%lu,%lu}.",
                        ShapeRank(input.vCache.shape), SafeShapeDim(input.vCache.shape, 0),
                        SafeShapeDim(input.vCache.shape, 1), SafeShapeDim(input.vCache.shape, 2),
                        SafeShapeDim(input.vCache.shape, 3), input.blockNum, input.numVHeads, input.blockSize,
                        input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape4D(input.kScaleCache, input.blockNum, input.numKHeads, input.blockSize, 1),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache shape rank=%u "
                        "dims={%lu,%lu,%lu,%lu}.",
                        ShapeRank(input.kScaleCache.shape), SafeShapeDim(input.kScaleCache.shape, 0),
                        SafeShapeDim(input.kScaleCache.shape, 1), SafeShapeDim(input.kScaleCache.shape, 2),
                        SafeShapeDim(input.kScaleCache.shape, 3)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateStrides(const ContractInput &input,
                                                                     const std::string &opName)
{
    OP_CHECK_IF(!IsCacheStrideValid(input.kCache.stride, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache stride present=%u rank=%u "
                        "strides={%lu,%lu,%lu,%lu}, headDim=%lu.",
                        input.kCache.stride.present ? 1U : 0U, input.kCache.stride.rank, input.kCache.stride.stride0,
                        input.kCache.stride.stride1, input.kCache.stride.stride2, input.kCache.stride.stride3,
                        input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsCacheStrideValid(input.vCache.stride, input.headDim),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: vCache stride present=%u rank=%u "
                        "strides={%lu,%lu,%lu,%lu}, headDim=%lu.",
                        input.vCache.stride.present ? 1U : 0U, input.vCache.stride.rank, input.vCache.stride.stride0,
                        input.vCache.stride.stride1, input.vCache.stride.stride2, input.vCache.stride.stride3,
                        input.headDim),
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateMinimumTokenTileFeasible(
    const ContractInput &input, const std::string &opName)
{
    const uint64_t numQkHeads = input.numQHeads + input.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input.numVHeads;
    const uint64_t qkAlignedRows = CalcQkAlignedRows(1, input.numQHeads, input.numKHeads);
    OP_CHECK_IF(numQkHeads > QK_OUTPUT_ROWS_PER_AIV,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: minimum token tile exceeds Q/K output rows, "
                        "Nq+Nk=%lu limit=%lu.",
                        numQkHeads, QK_OUTPUT_ROWS_PER_AIV),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numQkvHeads > QKV_INPUT_ROWS_PER_AIV,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: minimum token tile exceeds QKV input rows, "
                        "Nq+Nk+Nv=%lu limit=%lu.",
                        numQkvHeads, QKV_INPUT_ROWS_PER_AIV),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.numVHeads > V_OUTPUT_ROWS_PER_AIV,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: minimum token tile exceeds V output rows, "
                        "Nv=%lu limit=%lu.",
                        input.numVHeads, V_OUTPUT_ROWS_PER_AIV),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qkAlignedRows > L0C_MAX_ROWS,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: minimum token tile exceeds side-by-side L0C "
                        "rows, align(Nq,16)+align(Nk,16)=%lu limit=%lu.",
                        qkAlignedRows, L0C_MAX_ROWS),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QkvRmsNormRopeCacheWithKScaleTiling::TrySelectTokenTile(const ContractTiling &tiling, uint64_t tokenTile,
                                                             const std::string &opName,
                                                             TokenTileSelection &selection)
{
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tokenTile, AIV_PER_AIC);
    const uint64_t rowTile = tokenTile * tiling.numQkHeads;
    const uint64_t rowTileAligned = Ops::Base::CeilAlign(rowTile, 16UL);
    const uint64_t qkAlignedRows = CalcQkAlignedRows(tokenTile, tiling.numQHeads, tiling.numKHeads);
    const bool fitsUb = tokenTilePerAiv * tiling.numQkHeads <= QK_OUTPUT_ROWS_PER_AIV &&
                        tokenTilePerAiv * tiling.numQkvHeads <= QKV_INPUT_ROWS_PER_AIV &&
                        tokenTilePerAiv * tiling.numVHeads <= V_OUTPUT_ROWS_PER_AIV;
    const bool fitsL0c = rowTile <= L0C_MAX_ROWS && rowTileAligned <= L0C_MAX_ROWS && qkAlignedRows <= L0C_MAX_ROWS;
    const bool accepted = fitsUb && fitsL0c;
    LogTokenTileCandidate(opName, tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb,
                          fitsL0c, accepted);
    if (accepted) {
        selection.tokenTile = tokenTile;
        selection.tokenTilePerAiv = tokenTilePerAiv;
        return true;
    }
    return false;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::SelectTokenTile(const ContractTiling &tiling,
                                                                     const std::string &opName,
                                                                     TokenTileSelection &selection)
{
    const uint64_t initialTokenTilePerAiv =
        std::min({MAX_TOKEN_TILE / 2, QK_OUTPUT_ROWS_PER_AIV / tiling.numQkHeads,
                  QKV_INPUT_ROWS_PER_AIV / tiling.numQkvHeads, V_OUTPUT_ROWS_PER_AIV / tiling.numVHeads});
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale token tile search start: initialTokenTilePerAiv=%lu "
            "qkOutputLimit=%lu qkvInputLimit=%lu vOutputLimit=%lu.",
            initialTokenTilePerAiv, QK_OUTPUT_ROWS_PER_AIV / tiling.numQkHeads,
            QKV_INPUT_ROWS_PER_AIV / tiling.numQkvHeads, V_OUTPUT_ROWS_PER_AIV / tiling.numVHeads);

    uint64_t searchTokenTilePerAiv = initialTokenTilePerAiv;
    while (searchTokenTilePerAiv > 0) {
        if (TrySelectTokenTile(tiling, 2 * searchTokenTilePerAiv, opName, selection)) {
            break;
        }
        --searchTokenTilePerAiv;
    }
    if (selection.tokenTile == 0) {
        (void)TrySelectTokenTile(tiling, 1, opName, selection);
    }
    OP_CHECK_IF(selection.tokenTile == 0,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: no token tile candidate fits."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillContractTiling(const ContractInput &input, uint64_t aicNum,
                                                             const TokenTileSelection &selection,
                                                             ContractTiling &tiling)
{
    tiling.coreTokenTile = Ops::Base::CeilDiv(input.totalTokens, aicNum);
    tiling.coreGroupNum = Ops::Base::CeilDiv(input.totalTokens, tiling.coreTokenTile);
    tiling.tokenTilePerAiv = selection.tokenTilePerAiv;
    tiling.tokenTile = selection.tokenTile;
    tiling.dimTile = DIM_TILE_FAST;
    tiling.rowTile = tiling.tokenTile * tiling.numQkHeads;
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

void QkvRmsNormRopeCacheWithKScaleTiling::LogContractTilingResult(const std::string &opName,
                                                                  const ContractTiling &tiling)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract tiling result: coreTokenTile=%lu coreGroupNum=%lu "
            "tokenTile=%lu tokenTilePerAiv=%lu Nq=%lu Nk=%lu Nv=%lu Nqk=%lu Nqkv=%lu "
            "rowTile=%lu rowTileAligned=%lu "
            "dimTile=%lu.",
            tiling.coreTokenTile, tiling.coreGroupNum, tiling.tokenTile, tiling.tokenTilePerAiv, tiling.numQHeads,
            tiling.numKHeads, tiling.numVHeads, tiling.numQkHeads, tiling.numQkvHeads, tiling.rowTile,
            tiling.rowTileAligned, tiling.dimTile);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract stride result: kCache={%lu,%lu,%lu} "
            "vCache={%lu,%lu,%lu} kScaleCache={%lu,%lu,%lu}.",
            tiling.kCacheStrideBlock, tiling.kCacheStrideHead, tiling.kCacheStrideToken, tiling.vCacheStrideBlock,
            tiling.vCacheStrideHead, tiling.vCacheStrideToken, tiling.kScaleCacheStrideBlock,
            tiling.kScaleCacheStrideHead, tiling.kScaleCacheStrideToken);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ValidateContractInput(const ContractInput &input,
                                                                           const std::string &opName)
{
    OP_LOGD(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract validation start.");
    ge::graphStatus status = ValidateHeadNums(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale head nums validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateScalarInputs(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale scalar validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateDtypes(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale dtype validation failed."),
                return ge::GRAPH_FAILED);
    status = ValidateShapes(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale shape validation failed."),
                return ge::GRAPH_FAILED);
    status = ValidateStrides(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale stride validation failed."),
                return ge::GRAPH_FAILED);
    status = ValidateMinimumTokenTileFeasible(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale resource boundary validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::BuildContractTiling(const ContractInput &input, uint64_t aicNum,
                                                                         ContractTiling &tiling,
                                                                         const std::string &opName)
{
    tiling = {};
    FillContractHeadNums(input, tiling);
    LogContractInputSummary(opName, input, aicNum, tiling);
    TokenTileSelection selection;
    OP_CHECK_IF(SelectTokenTile(tiling, opName, selection) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale token tile selection failed."),
                return ge::GRAPH_FAILED);

    FillContractTiling(input, aicNum, selection, tiling);
    LogContractTilingResult(opName, tiling);
    return ge::GRAPH_SUCCESS;
}

AivSlice QkvRmsNormRopeCacheWithKScaleTiling::MakeAivSlice(const TileParam &tile, uint64_t qkHeadTile,
                                                           uint64_t aivLocalId)
{
    AivSlice slice;
    const uint64_t cubeHalfTokenSize = Ops::Base::CeilDiv(tile.tokenSize, AIV_PER_AIC);
    const uint64_t tokenPrefix = aivLocalId == 0 ? 0 : cubeHalfTokenSize;
    const uint64_t tokenSize = aivLocalId == 0 ? cubeHalfTokenSize : tile.tokenSize - cubeHalfTokenSize;
    slice.tokenOffset = tile.tokenOffset + tokenPrefix;
    slice.tokenSize = tokenSize;
    slice.rowOffset = tokenPrefix * qkHeadTile;
    slice.rowSize = tokenSize * qkHeadTile;
    return slice;
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::MakeContiguousStrideInfo(const gert::Shape &shape,
                                                                              const std::string &opName,
                                                                              TensorStrideInfo &strideInfo)
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
                    OP_LOGE(opName.c_str(), "cache contiguous stride overflow: currentStride=%lld dim[%d]=%lld.",
                            static_cast<long long>(currentStride), dim, static_cast<long long>(dimValue)),
                    return ge::GRAPH_FAILED);
        currentStride *= dimValue;
    }
    strideInfo = ToStrideInfo(stride);
    return ge::GRAPH_SUCCESS;
}

const char *QkvRmsNormRopeCacheWithKScaleTiling::DtypeName(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16:
            return "bf16";
        case ge::DT_FLOAT:
            return "float";
        case ge::DT_INT32:
            return "int32";
        case ge::DT_FLOAT8_E4M3FN:
            return "float8_e4m3fn";
        case ge::DT_UNDEFINED:
            return "undefined";
        default:
            return "unknown";
    }
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ToTensorInfo(gert::TilingContext *context,
                                                                  const std::string &opName, const char *tensorName,
                                                                  size_t index, TensorContractInfo &info)
{
    const auto desc = context->GetInputDesc(index);
    if (desc == nullptr) {
        OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale required input %s index=%zu desc is nullptr.",
                tensorName, index);
        return ge::GRAPH_FAILED;
    }
    const auto shape = context->GetInputShape(index);
    if (shape == nullptr) {
        OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale required input %s index=%zu shape is nullptr.",
                tensorName, index);
        return ge::GRAPH_FAILED;
    }
    info.dtype = desc->GetDataType();
    info.shape = shape->GetStorageShape();
    info.shapePresent = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ToCacheTensorInfo(gert::TilingContext *context,
                                                                       const std::string &opName, size_t index,
                                                                       TensorContractInfo &info)
{
    const auto desc = context->GetInputDesc(index);
    if (desc == nullptr) {
        OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale cache input index=%zu desc is nullptr.", index);
        return ge::GRAPH_FAILED;
    }
    const auto storageShape = context->GetInputShape(index);
    if (storageShape == nullptr) {
        OP_LOGE(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu has no storage shape.", index);
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &storageShapeInfo = storageShape->GetStorageShape();
    const gert::Shape &logicalShapeInfo = storageShape->GetShape();
    const bool inputIsView = context->InputIsView(index);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale cache input index=%zu InputIsView=%u storageRank=%u "
            "storageShape={%lu,%lu,%lu,%lu} logicalRank=%u logicalShape={%lu,%lu,%lu,%lu}.",
            index, inputIsView ? 1U : 0U, ShapeRank(storageShapeInfo), SafeShapeDim(storageShapeInfo, 0),
            SafeShapeDim(storageShapeInfo, 1), SafeShapeDim(storageShapeInfo, 2), SafeShapeDim(storageShapeInfo, 3),
            ShapeRank(logicalShapeInfo), SafeShapeDim(logicalShapeInfo, 0), SafeShapeDim(logicalShapeInfo, 1),
            SafeShapeDim(logicalShapeInfo, 2), SafeShapeDim(logicalShapeInfo, 3));

    if (inputIsView) {
        const auto inputStride = context->GetInputStride(index);
        const uint32_t inputStrideRank = inputStride == nullptr ? 0U : static_cast<uint32_t>(inputStride->GetDimNum());
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu view stride query: stridePresent=%u "
                "strideRank=%u.",
                index, inputStride == nullptr ? 0U : 1U, inputStrideRank);
        if (inputStride != nullptr && inputStride->GetDimNum() != 0) {
            info.shape = logicalShapeInfo;
            info.shapePresent = true;
            info.stride = ToStrideInfo(*inputStride);
            info.dtype = desc->GetDataType();
            OP_LOGD(opName.c_str(),
                    "QkvRmsNormRopeCacheWithKScale cache input index=%zu uses view shape/stride: "
                    "shapeRank=%u shape={%lu,%lu,%lu,%lu} strideRank=%u stride={%lu,%lu,%lu,%lu}.",
                    index, ShapeRank(info.shape), SafeShapeDim(info.shape, 0), SafeShapeDim(info.shape, 1),
                    SafeShapeDim(info.shape, 2), SafeShapeDim(info.shape, 3), info.stride.rank, info.stride.stride0,
                    info.stride.stride1, info.stride.stride2, info.stride.stride3);
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale cache input index=%zu is view but stride is missing or empty; "
                "fall back to contiguous storage stride.",
                index);
    }

    const gert::Shape &shape = storageShape->GetStorageShape();
    info.shape = storageShapeInfo;
    info.shapePresent = true;
    OP_CHECK_IF(MakeContiguousStrideInfo(shape, opName, info.stride) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "make contiguous cache stride failed, input index=%zu.", index),
                return ge::GRAPH_FAILED);
    info.dtype = desc->GetDataType();
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale cache input index=%zu uses contiguous storage shape/stride: "
            "shapeRank=%u shape={%lu,%lu,%lu,%lu} strideRank=%u stride={%lu,%lu,%lu,%lu}.",
            index, ShapeRank(info.shape), SafeShapeDim(info.shape, 0), SafeShapeDim(info.shape, 1),
            SafeShapeDim(info.shape, 2), SafeShapeDim(info.shape, 3), info.stride.rank, info.stride.stride0,
            info.stride.stride1, info.stride.stride2, info.stride.stride3);
    return ge::GRAPH_SUCCESS;
}

TensorContractInfo QkvRmsNormRopeCacheWithKScaleTiling::ToOptionalTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    const auto desc = context->GetOptionalInputDesc(index);
    if (desc == nullptr) {
        return info;
    }
    info.dtype = desc->GetDataType();
    const auto shape = context->GetOptionalInputShape(index);
    if (shape == nullptr) {
        return info;
    }
    info.shape = shape->GetStorageShape();
    info.shapePresent = true;
    return info;
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogTensorInfo(const std::string &opName, const char *tensorName,
                                                        const TensorContractInfo &info)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale input tensor %s: dtype=%s(%u) shapePresent=%u shapeRank=%u "
            "shape={%lu,%lu,%lu,%lu} stridePresent=%u strideRank=%u stride={%lu,%lu,%lu,%lu}.",
            tensorName, DtypeName(info.dtype), static_cast<uint32_t>(info.dtype), info.shapePresent ? 1U : 0U,
            ShapeRank(info.shape), SafeShapeDim(info.shape, 0), SafeShapeDim(info.shape, 1),
            SafeShapeDim(info.shape, 2), SafeShapeDim(info.shape, 3), info.stride.present ? 1U : 0U,
            info.stride.rank, info.stride.stride0, info.stride.stride1, info.stride.stride2, info.stride.stride3);
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogContractInput(const std::string &opName, const ContractInput &input,
                                                           uint64_t aicNum, float epsilon)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale parsed attrs/input summary: head_nums={%lu,%lu,%lu} epsilon=%f "
            "T=%lu batch=%lu headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.numQHeads, input.numKHeads, input.numVHeads, epsilon, input.totalTokens, input.batch, input.headDim,
            input.maxSeqLen, input.blockNum, input.blockSize, aicNum);
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

ge::graphStatus
QkvRmsNormRopeCacheWithKScaleTiling::SetWorkspace(gert::TilingContext *context, const std::string &opName,
                                                  const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo,
                                                  size_t &workspaceSize)
{
    size_t *workspaces = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(opName.c_str(), "get workspace failed."), return ge::GRAPH_FAILED);
    const size_t opWorkspaceSize = static_cast<size_t>(compileInfo.opWorkspaceSize);
    workspaceSize = opWorkspaceSize + RESERVED_WORKSPACE_SIZE;
    workspaces[0] = workspaceSize;
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale workspace: opWorkspaceSize=%zu reservedWorkspaceSize=%zu total=%zu.",
            opWorkspaceSize, RESERVED_WORKSPACE_SIZE, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::LogTilingData(const std::string &opName,
                                                        const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
                                                        const ContractTiling &contractTiling, uint64_t tilingKey,
                                                        uint64_t blockDim, size_t dataSize, size_t workspaceSize)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final tilingData: totalTokens=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu "
            "headDim=%lu blockSize=%lu coreTokenTile=%lu coreGroupNum=%lu tokenTile=%lu epsilon=%f.",
            tilingData.totalTokens, tilingData.batch, tilingData.numQHeads, tilingData.numKHeads, tilingData.numVHeads,
            tilingData.headDim, tilingData.blockSize, tilingData.coreTokenTile, tilingData.coreGroupNum,
            tilingData.tokenTile, tilingData.epsilon);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final strides: kCache={%lu,%lu,%lu} vCache={%lu,%lu,%lu} "
            "kScaleCache={%lu,%lu,%lu}.",
            tilingData.kCacheStrideBlock, tilingData.kCacheStrideHead, tilingData.kCacheStrideToken,
            tilingData.vCacheStrideBlock, tilingData.vCacheStrideHead, tilingData.vCacheStrideToken,
            tilingData.kScaleCacheStrideBlock, tilingData.kScaleCacheStrideHead, tilingData.kScaleCacheStrideToken);
    OP_LOGD(
        opName.c_str(),
        "QkvRmsNormRopeCacheWithKScale final launch: tilingKey=%lu blockDim=%lu tilingDataSize=%zu workspaceSize=%zu "
        "Nqk=%lu Nv=%lu rowTile=%lu rowTileAligned=%lu tokenTilePerAiv=%lu dimTile=%lu.",
        tilingKey, blockDim, dataSize, workspaceSize, contractTiling.numQkHeads, contractTiling.numVHeads,
        contractTiling.rowTile, contractTiling.rowTileAligned, contractTiling.tokenTilePerAiv, contractTiling.dimTile);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::ParseHeadNumsAttr(gert::TilingContext *context,
                                                                       const std::string &opName, ContractInput &input)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(HEAD_NUMS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    OP_CHECK_IF(headNums->GetSize() != 3, OP_LOGE(opName.c_str(), "head_nums must have exactly 3 values."),
                return ge::GRAPH_FAILED);
    const auto *headNumsData = headNums->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, headNumsData);
    for (size_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(headNumsData[i] <= 0, OP_LOGE(opName.c_str(), "head_nums[%zu] must be greater than 0.", i),
                    return ge::GRAPH_FAILED);
    }

    input.numQHeads = static_cast<uint64_t>(headNumsData[0]);
    input.numKHeads = static_cast<uint64_t>(headNumsData[1]);
    input.numVHeads = static_cast<uint64_t>(headNumsData[2]);
    return ge::GRAPH_SUCCESS;
}

float QkvRmsNormRopeCacheWithKScaleTiling::ParseEpsilonAttr(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const float *epsilonAttr = attrs == nullptr ? nullptr : attrs->GetFloat(EPSILON_ATTR_INDEX);
    return epsilonAttr == nullptr ? DEFAULT_EPSILON : *epsilonAttr;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillShapeDerivedFields(const std::string &opName, ContractInput &input)
{
    const auto &qkvShape = input.qkv.shape;
    const auto &cosSinShape = input.cosSin.shape;
    const auto &cuSeqLenShape = input.cuSeqLen.shape;
    const auto &kCacheShape = input.kCache.shape;
    input.totalTokens = ShapeDim(qkvShape, 1);
    input.headDim = ShapeDim(qkvShape, 2);
    input.batch = ShapeDim(cuSeqLenShape, 0) - 1;
    input.maxSeqLen = ShapeDim(cosSinShape, 0);
    input.blockNum = ShapeDim(kCacheShape, 0);
    input.blockSize = ShapeDim(kCacheShape, 2);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale derived shape fields: totalTokens=%lu batch=%lu headDim=%lu "
            "maxSeqLen=%lu blockNum=%lu blockSize=%lu.",
            input.totalTokens, input.batch, input.headDim, input.maxSeqLen, input.blockNum, input.blockSize);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::FillRequiredTensorInputs(gert::TilingContext *context,
                                                                              const std::string &opName,
                                                                              ContractInput &input)
{
    OP_CHECK_IF(ToTensorInfo(context, opName, "qkv", QKV_INDEX, input.qkv) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse qkv tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context, opName, "qGamma", Q_GAMMA_INDEX, input.qGamma) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse qGamma tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context, opName, "kGamma", K_GAMMA_INDEX, input.kGamma) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse kGamma tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context, opName, "cosSin", COS_SIN_INDEX, input.cosSin) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse cosSin tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context, opName, "slotMapping", SLOT_MAPPING_INDEX, input.slotMapping) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse slotMapping tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context, opName, K_CACHE_INDEX, input.kCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse kCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context, opName, V_CACHE_INDEX, input.vCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse vCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context, opName, K_SCALE_CACHE_INDEX, input.kScaleCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse kScaleCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context, opName, "cuSeqLen", CU_SEQ_LEN_INDEX, input.cuSeqLen) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse cuSeqLen tensor info."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillOptionalTensorInputs(gert::TilingContext *context, ContractInput &input)
{
    input.qRotation = ToOptionalTensorInfo(context, Q_ROTATION_INDEX);
    input.kRotation = ToOptionalTensorInfo(context, K_ROTATION_INDEX);
    input.vScale = ToOptionalTensorInfo(context, V_SCALE_INDEX);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::BuildContractInput(gert::TilingContext *context,
                                                                        const std::string &opName, ContractInput &input,
                                                                        float &epsilon)
{
    OP_CHECK_IF(ParseHeadNumsAttr(context, opName, input) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse head_nums attr."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(FillRequiredTensorInputs(context, opName, input) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "failed to parse required tensor inputs."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateShapeInputsForDerivedFields(input, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "invalid shape inputs for derived fields."), return ge::GRAPH_FAILED);
    FillShapeDerivedFields(opName, input);
    FillOptionalTensorInputs(context, input);
    epsilon = ParseEpsilonAttr(context);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleTiling::FillTilingData(const ContractInput &input,
                                                         const ContractTiling &contractTiling, float epsilon,
                                                         QkvRmsNormRopeCacheWithKScaleTilingData &tilingData)
{
    tilingData.totalTokens = input.totalTokens;
    tilingData.batch = input.batch;
    tilingData.numQHeads = contractTiling.numQHeads;
    tilingData.numKHeads = contractTiling.numKHeads;
    tilingData.numVHeads = contractTiling.numVHeads;
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

ge::graphStatus
QkvRmsNormRopeCacheWithKScaleTiling::ValidateCompileInfo(const std::string &opName,
                                                         const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale compile info: aicNum=%u aivNum=%u ubSize=%lu l1Size=%lu l0cSize=%lu "
            "opWorkspaceSize=%lu.",
            compileInfo.aicNum, compileInfo.aivNum, compileInfo.ubSize, compileInfo.l1Size, compileInfo.l0cSize,
            compileInfo.opWorkspaceSize);
    OP_CHECK_IF(compileInfo.aicNum == 0, OP_LOGE(opName.c_str(), "aicNum in compile info must be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo.aivNum != compileInfo.aicNum * AIV_PER_AIC,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale requires aic:aiv ratio 1:%lu, got aicNum=%u aivNum=%u.",
                        AIV_PER_AIC, compileInfo.aicNum, compileInfo.aivNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
QkvRmsNormRopeCacheWithKScaleTiling::CopyRawTilingData(gert::TilingContext *context, const std::string &opName,
                                                       const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
                                                       size_t &tilingDataSize)
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
                OP_LOGE(opName.c_str(), "tiling data size[%zu] exceeds capacity[%zu].", tilingDataSize, tilingCapacity),
                return ge::GRAPH_FAILED);
    const auto ret =
        memcpy_s(rawTilingData->GetData(), tilingCapacity, reinterpret_cast<const void *>(&tilingData), tilingDataSize);
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
    aicNum_ = 0;
    epsilon_ = DEFAULT_EPSILON;
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
    OP_CHECK_IF(ValidateContractInput(input_, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "invalid QkvRmsNormRopeCacheWithKScale contract input."),
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
    aicNum_ = compileInfo_->aicNum;
    LogContractInput(opName, input_, aicNum_, epsilon_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleTiling::DoOpTiling()
{
    const std::string opName = context_->GetNodeName();
    OP_CHECK_IF(BuildContractTiling(input_, aicNum_, contractTiling_, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract tiling build failed."),
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

REGISTER_OPS_TILING_TEMPLATE(QkvRmsNormRopeCacheWithKScale, QkvRmsNormRopeCacheWithKScaleTiling,
                             TILING_TEMPLATE_PRIORITY);

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling
