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

#include "../../../op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h"
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

uint64_t QkvRmsNormRopeCacheWithKScaleBaseTiling::CalcQkAlignedRows(uint64_t tokenTile, uint64_t numQHeads,
                                                                    uint64_t numKHeads)
{
    return Ops::Base::CeilAlign(tokenTile * numQHeads, 16UL) + Ops::Base::CeilAlign(tokenTile * numKHeads, 16UL);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateHeadNums(const ContractInput &input,
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateScalarInputs(const ContractInput &input,
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateDtype(const std::string &opName,
                                                                       const char *tensorName,
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateDtypes(const ContractInput &input,
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateShapeInputsForDerivedFields(const ContractInput &input,
                                                                                             const std::string &opName)
{
    OP_CHECK_IF(!input.qkv.shapePresent || input.qkv.shape.GetDimNum() != 3,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: qkv present=%u rank=%u, expected rank=3.",
                        input.qkv.shapePresent ? 1U : 0U, static_cast<uint32_t>(input.qkv.shape.GetDimNum())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.cosSin.shapePresent || input.cosSin.shape.GetDimNum() != 2,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cosSin present=%u rank=%u, expected rank=2.",
                        input.cosSin.shapePresent ? 1U : 0U, static_cast<uint32_t>(input.cosSin.shape.GetDimNum())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.cuSeqLen.shapePresent || input.cuSeqLen.shape.GetDimNum() != 1,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen present=%u rank=%u, "
                        "expected rank=1.",
                        input.cuSeqLen.shapePresent ? 1U : 0U, static_cast<uint32_t>(input.cuSeqLen.shape.GetDimNum())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input.cuSeqLen.shape.GetDim(0) == 0,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen dim0=%ld, expected dim0>0.",
                        input.cuSeqLen.shape.GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.kCache.shapePresent || input.kCache.shape.GetDimNum() != 4,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache present=%u rank=%u, expected rank=4.",
                        input.kCache.shapePresent ? 1U : 0U, static_cast<uint32_t>(input.kCache.shape.GetDimNum())),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateShapeRanks(const ContractInput &input,
                                                                            const std::string &opName)
{
    struct RankRule {
        const char *tensorName;
        const TensorContractInfo *tensor;
        size_t expectedRank;
    };
    const RankRule rankRules[] = {
        {"qkv", &input.qkv, 3},
        {"qGamma", &input.qGamma, 1},
        {"kGamma", &input.kGamma, 1},
        {"cosSin", &input.cosSin, 2},
        {"slotMapping", &input.slotMapping, 1},
        {"cuSeqLen", &input.cuSeqLen, 1},
        {"qRotation", &input.qRotation, 2},
        {"kRotation", &input.kRotation, 2},
        {"vScale", &input.vScale, 1},
        {"kCache", &input.kCache, 4},
        {"vCache", &input.vCache, 4},
        {"kScaleCache", &input.kScaleCache, 4},
    };
    for (const auto &rule : rankRules) {
        OP_CHECK_IF(!rule.tensor->shapePresent || rule.tensor->shape.GetDimNum() != rule.expectedRank,
                    OP_LOGE(opName.c_str(),
                            "QkvRmsNormRopeCacheWithKScale contract reject: %s present=%u rank=%u, "
                            "expected rank=%zu.",
                            rule.tensorName, rule.tensor->shapePresent ? 1U : 0U,
                            static_cast<uint32_t>(rule.tensor->shape.GetDimNum()), rule.expectedRank),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateNonCacheShapes(const ContractInput &input,
                                                                                const std::string &opName)
{
    const uint64_t numQkHeads = input.numQHeads + input.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input.numVHeads;
    OP_CHECK_IF(static_cast<uint64_t>(input.qkv.shape.GetDim(0)) != numQkvHeads ||
                    static_cast<uint64_t>(input.qkv.shape.GetDim(1)) != input.totalTokens ||
                    static_cast<uint64_t>(input.qkv.shape.GetDim(2)) != input.headDim,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: qkv dims={%ld,%ld,%ld}.",
                        input.qkv.shape.GetDim(0), input.qkv.shape.GetDim(1), input.qkv.shape.GetDim(2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.qGamma.shape.GetDim(0)) != input.headDim ||
                    static_cast<uint64_t>(input.kGamma.shape.GetDim(0)) != input.headDim,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: gamma dims qGamma={%ld} kGamma={%ld}.",
                        input.qGamma.shape.GetDim(0), input.kGamma.shape.GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.cosSin.shape.GetDim(0)) != input.maxSeqLen ||
                    static_cast<uint64_t>(input.cosSin.shape.GetDim(1)) != input.headDim,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: cosSin dims={%ld,%ld}.",
                        input.cosSin.shape.GetDim(0), input.cosSin.shape.GetDim(1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.slotMapping.shape.GetDim(0)) != input.totalTokens,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: slotMapping dim0=%ld.",
                        input.slotMapping.shape.GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.cuSeqLen.shape.GetDim(0)) != input.batch + 1,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen dim0=%ld.",
                        input.cuSeqLen.shape.GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.qRotation.shape.GetDim(0)) != input.headDim ||
                    static_cast<uint64_t>(input.qRotation.shape.GetDim(1)) != input.headDim,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: qRotation dims={%ld,%ld}.",
                        input.qRotation.shape.GetDim(0), input.qRotation.shape.GetDim(1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.kRotation.shape.GetDim(0)) != input.headDim ||
                    static_cast<uint64_t>(input.kRotation.shape.GetDim(1)) != input.headDim,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: kRotation dims={%ld,%ld}.",
                        input.kRotation.shape.GetDim(0), input.kRotation.shape.GetDim(1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.vScale.shape.GetDim(0)) != input.numVHeads,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract reject: vScale dim0=%ld.",
                        input.vScale.shape.GetDim(0)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateCacheShapes(const ContractInput &input,
                                                                             const std::string &opName)
{
    OP_CHECK_IF(static_cast<uint64_t>(input.kCache.shape.GetDim(0)) != input.blockNum ||
                    static_cast<uint64_t>(input.kCache.shape.GetDim(1)) != input.numKHeads ||
                    static_cast<uint64_t>(input.kCache.shape.GetDim(2)) != input.blockSize ||
                    static_cast<uint64_t>(input.kCache.shape.GetDim(3)) != input.headDim,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kCache dims={%ld,%ld,%ld,%ld}, "
                        "expected={%lu,%lu,%lu,%lu}.",
                        input.kCache.shape.GetDim(0), input.kCache.shape.GetDim(1), input.kCache.shape.GetDim(2),
                        input.kCache.shape.GetDim(3), input.blockNum, input.numKHeads, input.blockSize, input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.vCache.shape.GetDim(0)) != input.blockNum ||
                    static_cast<uint64_t>(input.vCache.shape.GetDim(1)) != input.numVHeads ||
                    static_cast<uint64_t>(input.vCache.shape.GetDim(2)) != input.blockSize ||
                    static_cast<uint64_t>(input.vCache.shape.GetDim(3)) != input.headDim,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: vCache dims={%ld,%ld,%ld,%ld}, "
                        "expected={%lu,%lu,%lu,%lu}.",
                        input.vCache.shape.GetDim(0), input.vCache.shape.GetDim(1), input.vCache.shape.GetDim(2),
                        input.vCache.shape.GetDim(3), input.blockNum, input.numVHeads, input.blockSize, input.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input.kScaleCache.shape.GetDim(0)) != input.blockNum ||
                    static_cast<uint64_t>(input.kScaleCache.shape.GetDim(1)) != input.numKHeads ||
                    static_cast<uint64_t>(input.kScaleCache.shape.GetDim(2)) != input.blockSize ||
                    input.kScaleCache.shape.GetDim(3) != 1,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache dims={%ld,%ld,%ld,%ld}.",
                        input.kScaleCache.shape.GetDim(0), input.kScaleCache.shape.GetDim(1),
                        input.kScaleCache.shape.GetDim(2), input.kScaleCache.shape.GetDim(3)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateShapes(const ContractInput &input,
                                                                        const std::string &opName)
{
    ge::graphStatus status = ValidateShapeRanks(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale shape rank validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateNonCacheShapes(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale non-cache shape validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateCacheShapes(input, opName);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale cache shape validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateStrides(const ContractInput &input,
                                                                         const std::string &opName)
{
    const auto strideDimOrZero = [](const gert::Stride &stride, size_t index) -> int64_t {
        return index < stride.GetDimNum() ? stride.GetStride(index) : 0;
    };
    const int64_t headDimValue = static_cast<int64_t>(input.headDim);
    const gert::Stride &kStride = input.kCache.stride;
    const gert::Stride &vStride = input.vCache.stride;
    const gert::Stride &kScaleStride = input.kScaleCache.stride;
    OP_CHECK_IF(
        !input.kCache.stridePresent || kStride.GetDimNum() != 4 || kStride.GetStride(0) <= 0 ||
            kStride.GetStride(1) < headDimValue || kStride.GetStride(2) < headDimValue || kStride.GetStride(3) != 1,
        OP_LOGE(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale contract reject: kCache stride present=%u rank=%u "
                "strides={%lld,%lld,%lld,%lld}, headDim=%lu.",
                input.kCache.stridePresent ? 1U : 0U, static_cast<uint32_t>(input.kCache.stride.GetDimNum()),
                strideDimOrZero(input.kCache.stride, 0), strideDimOrZero(input.kCache.stride, 1),
                strideDimOrZero(input.kCache.stride, 2), strideDimOrZero(input.kCache.stride, 3), input.headDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !input.vCache.stridePresent || vStride.GetDimNum() != 4 || vStride.GetStride(0) <= 0 ||
            vStride.GetStride(1) < headDimValue || vStride.GetStride(2) < headDimValue || vStride.GetStride(3) != 1,
        OP_LOGE(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale contract reject: vCache stride present=%u rank=%u "
                "strides={%lld,%lld,%lld,%lld}, headDim=%lu.",
                input.vCache.stridePresent ? 1U : 0U, static_cast<uint32_t>(input.vCache.stride.GetDimNum()),
                strideDimOrZero(input.vCache.stride, 0), strideDimOrZero(input.vCache.stride, 1),
                strideDimOrZero(input.vCache.stride, 2), strideDimOrZero(input.vCache.stride, 3), input.headDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStride.GetStride(0) != vStride.GetStride(0) || kStride.GetStride(1) != vStride.GetStride(1) ||
                    kStride.GetStride(2) != vStride.GetStride(2),
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: k/v cache non-innermost strides mismatch "
                        "k={%lld,%lld,%lld} v={%lld,%lld,%lld}.",
                        input.kCache.stride.GetStride(0), input.kCache.stride.GetStride(1),
                        input.kCache.stride.GetStride(2), input.vCache.stride.GetStride(0),
                        input.vCache.stride.GetStride(1), input.vCache.stride.GetStride(2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input.kScaleCache.stridePresent || kScaleStride.GetDimNum() != 4 || kScaleStride.GetStride(0) <= 0 ||
                    kScaleStride.GetStride(1) < 1 || kScaleStride.GetStride(2) < 1 || kScaleStride.GetStride(3) != 1,
                OP_LOGE(opName.c_str(),
                        "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache stride present=%u rank=%u "
                        "strides={%lld,%lld,%lld,%lld}.",
                        input.kScaleCache.stridePresent ? 1U : 0U,
                        static_cast<uint32_t>(input.kScaleCache.stride.GetDimNum()),
                        strideDimOrZero(input.kScaleCache.stride, 0), strideDimOrZero(input.kScaleCache.stride, 1),
                        strideDimOrZero(input.kScaleCache.stride, 2), strideDimOrZero(input.kScaleCache.stride, 3)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateMinimumTokenTileFeasible(const ContractInput &input,
                                                                                          const std::string &opName)
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

bool QkvRmsNormRopeCacheWithKScaleBaseTiling::TrySelectTokenTile(const ContractTiling &tiling, uint64_t tokenTile,
                                                                 const std::string &opName,
                                                                 TokenTileSelection &selection)
{
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tokenTile, AIV_PER_AIC);
    const uint64_t numQkHeads = tiling.numQHeads + tiling.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + tiling.numVHeads;
    const uint64_t rowTile = tokenTile * numQkHeads;
    const uint64_t rowTileAligned = Ops::Base::CeilAlign(rowTile, 16UL);
    const uint64_t qkAlignedRows = CalcQkAlignedRows(tokenTile, tiling.numQHeads, tiling.numKHeads);
    const bool fitsUb = tokenTilePerAiv * numQkHeads <= QK_OUTPUT_ROWS_PER_AIV &&
                        tokenTilePerAiv * numQkvHeads <= QKV_INPUT_ROWS_PER_AIV &&
                        tokenTilePerAiv * tiling.numVHeads <= V_OUTPUT_ROWS_PER_AIV;
    const bool fitsL0c = rowTile <= L0C_MAX_ROWS && rowTileAligned <= L0C_MAX_ROWS && qkAlignedRows <= L0C_MAX_ROWS;
    const bool accepted = fitsUb && fitsL0c;
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale token tile candidate: tokenTilePerAiv=%lu tokenTile=%lu rowTile=%lu "
            "rowTileAligned=%lu qkAlignedRows=%lu fitsUb=%u fitsSideBySideL0c=%u accepted=%u.",
            tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb ? 1U : 0U, fitsL0c ? 1U : 0U,
            accepted ? 1U : 0U);
    if (accepted) {
        selection.tokenTile = tokenTile;
        selection.tokenTilePerAiv = tokenTilePerAiv;
        return true;
    }
    return false;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::SelectTokenTile(const ContractTiling &tiling,
                                                                         const std::string &opName,
                                                                         TokenTileSelection &selection)
{
    const uint64_t numQkHeads = tiling.numQHeads + tiling.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + tiling.numVHeads;
    const uint64_t initialTokenTilePerAiv =
        std::min({MAX_TOKEN_TILE / 2, QK_OUTPUT_ROWS_PER_AIV / numQkHeads, QKV_INPUT_ROWS_PER_AIV / numQkvHeads,
                  V_OUTPUT_ROWS_PER_AIV / tiling.numVHeads});
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale token tile search start: initialTokenTilePerAiv=%lu "
            "qkOutputLimit=%lu qkvInputLimit=%lu vOutputLimit=%lu.",
            initialTokenTilePerAiv, QK_OUTPUT_ROWS_PER_AIV / numQkHeads, QKV_INPUT_ROWS_PER_AIV / numQkvHeads,
            V_OUTPUT_ROWS_PER_AIV / tiling.numVHeads);

    uint64_t searchTokenTilePerAiv = initialTokenTilePerAiv;
    while (searchTokenTilePerAiv > 0) {
        if (TrySelectTokenTile(tiling, 2 * searchTokenTilePerAiv, opName, selection)) {
            break;
        }
        --searchTokenTilePerAiv;
    }
    if (selection.tokenTile == 0) {
        selection.tokenTile = 1;
        selection.tokenTilePerAiv = 1;
        OP_LOGD(opName.c_str(),
                "QkvRmsNormRopeCacheWithKScale token tile search fallback: use prevalidated minimum tokenTile=1.");
    }
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillContractTiling(const ContractInput &input, uint64_t aicNum,
                                                                 const TokenTileSelection &selection,
                                                                 ContractTiling &tiling)
{
    const uint64_t numQkHeads = tiling.numQHeads + tiling.numKHeads;
    tiling.coreTokenTile = Ops::Base::CeilDiv(input.totalTokens, aicNum);
    tiling.coreGroupNum = Ops::Base::CeilDiv(input.totalTokens, tiling.coreTokenTile);
    tiling.tokenTilePerAiv = selection.tokenTilePerAiv;
    tiling.tokenTile = selection.tokenTile;
    tiling.dimTile = DIM_TILE_FAST;
    tiling.rowTile = tiling.tokenTile * numQkHeads;
    tiling.rowTileAligned = Ops::Base::CeilAlign(tiling.rowTile, 16UL);
    tiling.kvCacheStrideBlock = static_cast<uint64_t>(input.kCache.stride.GetStride(0));
    tiling.kvCacheStrideHead = static_cast<uint64_t>(input.kCache.stride.GetStride(1));
    tiling.kvCacheStrideToken = static_cast<uint64_t>(input.kCache.stride.GetStride(2));
    tiling.kScaleCacheStrideBlock = static_cast<uint64_t>(input.kScaleCache.stride.GetStride(0));
    tiling.kScaleCacheStrideHead = static_cast<uint64_t>(input.kScaleCache.stride.GetStride(1));
    tiling.kScaleCacheStrideToken = static_cast<uint64_t>(input.kScaleCache.stride.GetStride(2));
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateContractInput(const ContractInput &input,
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::BuildContractTiling(const ContractInput &input,
                                                                             uint64_t aicNum, ContractTiling &tiling,
                                                                             const std::string &opName)
{
    tiling = {};
    tiling.numQHeads = input.numQHeads;
    tiling.numKHeads = input.numKHeads;
    tiling.numVHeads = input.numVHeads;
    const uint64_t numQkHeads = tiling.numQHeads + tiling.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + tiling.numVHeads;
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract input: T=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu Nqk=%lu Nqkv=%lu "
            "headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.totalTokens, input.batch, input.numQHeads, input.numKHeads, input.numVHeads, numQkHeads, numQkvHeads,
            input.headDim, input.maxSeqLen, input.blockNum, input.blockSize, aicNum);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale resource constants: maxTokenTile=%lu qkvInputRowsPerAiv=%lu "
            "qkOutputRowsPerAiv=%lu vOutputRowsPerAiv=%lu l0aFullDimSegmentRows=%lu l0cMaxRows=%lu "
            "aivPerAic=%lu.",
            MAX_TOKEN_TILE, QKV_INPUT_ROWS_PER_AIV, QK_OUTPUT_ROWS_PER_AIV, V_OUTPUT_ROWS_PER_AIV,
            L0A_FULL_DIM_SEGMENT_ROWS, L0C_MAX_ROWS, AIV_PER_AIC);
    TokenTileSelection selection;
    OP_CHECK_IF(SelectTokenTile(tiling, opName, selection) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale token tile selection failed."),
                return ge::GRAPH_FAILED);

    FillContractTiling(input, aicNum, selection, tiling);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract tiling result: coreTokenTile=%lu coreGroupNum=%lu "
            "tokenTile=%lu tokenTilePerAiv=%lu Nq=%lu Nk=%lu Nv=%lu Nqk=%lu Nqkv=%lu "
            "rowTile=%lu rowTileAligned=%lu "
            "dimTile=%lu.",
            tiling.coreTokenTile, tiling.coreGroupNum, tiling.tokenTile, tiling.tokenTilePerAiv, tiling.numQHeads,
            tiling.numKHeads, tiling.numVHeads, numQkHeads, numQkvHeads, tiling.rowTile, tiling.rowTileAligned,
            tiling.dimTile);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale contract stride result: kvCache={%lu,%lu,%lu} "
            "kScaleCache={%lu,%lu,%lu}.",
            tiling.kvCacheStrideBlock, tiling.kvCacheStrideHead, tiling.kvCacheStrideToken,
            tiling.kScaleCacheStrideBlock, tiling.kScaleCacheStrideHead, tiling.kScaleCacheStrideToken);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::MakeContiguousStrideInfo(const gert::Shape &shape,
                                                                                  const std::string &opName,
                                                                                  gert::Stride &strideInfo)
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
    strideInfo = stride;
    return ge::GRAPH_SUCCESS;
}

const char *QkvRmsNormRopeCacheWithKScaleBaseTiling::DtypeName(ge::DataType dtype)
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ToTensorInfo(gert::TilingContext *context,
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ToCacheTensorInfo(gert::TilingContext *context,
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
        OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale cache input index=%zu has no storage shape.", index);
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &storageShapeInfo = storageShape->GetStorageShape();
    const gert::Shape &logicalShapeInfo = storageShape->GetShape();
    const bool inputIsView = context->InputIsView(index);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale cache input index=%zu InputIsView=%u storageRank=%u "
            "logicalRank=%u.",
            index, inputIsView ? 1U : 0U, static_cast<uint32_t>(storageShapeInfo.GetDimNum()),
            static_cast<uint32_t>(logicalShapeInfo.GetDimNum()));

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
            info.stride = *inputStride;
            info.stridePresent = true;
            info.dtype = desc->GetDataType();
            OP_LOGD(opName.c_str(),
                    "QkvRmsNormRopeCacheWithKScale cache input index=%zu uses view shape/stride: "
                    "shapeRank=%u strideRank=%u stride={%lld,%lld,%lld,%lld}.",
                    index, static_cast<uint32_t>(info.shape.GetDimNum()),
                    static_cast<uint32_t>(info.stride.GetDimNum()), info.stride.GetStride(0), info.stride.GetStride(1),
                    info.stride.GetStride(2), info.stride.GetStride(3));
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
    info.stridePresent = true;
    info.dtype = desc->GetDataType();
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale cache input index=%zu uses contiguous storage shape/stride: "
            "shapeRank=%u strideRank=%u stride={%lld,%lld,%lld,%lld}.",
            index, static_cast<uint32_t>(info.shape.GetDimNum()), static_cast<uint32_t>(info.stride.GetDimNum()),
            info.stride.GetStride(0), info.stride.GetStride(1), info.stride.GetStride(2), info.stride.GetStride(3));
    return ge::GRAPH_SUCCESS;
}

TensorContractInfo QkvRmsNormRopeCacheWithKScaleBaseTiling::ToOptionalTensorInfo(gert::TilingContext *context,
                                                                                 size_t index)
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

void QkvRmsNormRopeCacheWithKScaleBaseTiling::LogTensorInfo(const std::string &opName, const char *tensorName,
                                                            const TensorContractInfo &info)
{
    const auto strideDimOrZero = [](const gert::Stride &stride, size_t index) -> int64_t {
        return index < stride.GetDimNum() ? stride.GetStride(index) : 0;
    };
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale input tensor %s: dtype=%s(%u) shapePresent=%u shapeRank=%u "
            "stridePresent=%u strideRank=%u stride={%lld,%lld,%lld,%lld}.",
            tensorName, DtypeName(info.dtype), static_cast<uint32_t>(info.dtype), info.shapePresent ? 1U : 0U,
            static_cast<uint32_t>(info.shape.GetDimNum()), info.stridePresent ? 1U : 0U,
            static_cast<uint32_t>(info.stride.GetDimNum()), strideDimOrZero(info.stride, 0),
            strideDimOrZero(info.stride, 1), strideDimOrZero(info.stride, 2), strideDimOrZero(info.stride, 3));
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::LogContractInput(const std::string &opName, const ContractInput &input,
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
QkvRmsNormRopeCacheWithKScaleBaseTiling::SetWorkspace(gert::TilingContext *context, const std::string &opName,
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

void QkvRmsNormRopeCacheWithKScaleBaseTiling::LogTilingData(const std::string &opName,
                                                            const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
                                                            const ContractTiling &contractTiling, uint64_t tilingKey,
                                                            uint64_t blockDim, size_t dataSize, size_t workspaceSize)
{
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final tilingData: totalTokens=%lu batch=%lu qHeadNum=%lu kvHeadNum=%lu "
            "headDim=%lu blockSize=%lu coreTokenTile=%lu coreGroupNum=%lu tokenTile=%lu epsilon=%f.",
            tilingData.totalTokens, tilingData.batch, tilingData.qHeadNum, tilingData.kvHeadNum, tilingData.headDim,
            tilingData.blockSize, tilingData.coreTokenTile, tilingData.coreGroupNum, tilingData.tokenTile,
            tilingData.epsilon);
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale final strides: kvCache={%lu,%lu,%lu} kScaleCache={%lu,%lu,%lu}.",
            tilingData.kvCacheStrideBlock, tilingData.kvCacheStrideHead, tilingData.kvCacheStrideToken,
            tilingData.kScaleCacheStrideBlock, tilingData.kScaleCacheStrideHead, tilingData.kScaleCacheStrideToken);
    OP_LOGD(
        opName.c_str(),
        "QkvRmsNormRopeCacheWithKScale final launch: tilingKey=%lu blockDim=%lu tilingDataSize=%zu workspaceSize=%zu "
        "Nqk=%lu Nv=%lu rowTile=%lu rowTileAligned=%lu tokenTilePerAiv=%lu dimTile=%lu.",
        tilingKey, blockDim, dataSize, workspaceSize, contractTiling.numQHeads + contractTiling.numKHeads,
        contractTiling.numVHeads, contractTiling.rowTile, contractTiling.rowTileAligned, contractTiling.tokenTilePerAiv,
        contractTiling.dimTile);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseHeadNumsAttr(gert::TilingContext *context,
                                                                           const std::string &opName,
                                                                           ContractInput &input)
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

float QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseEpsilonAttr(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const float *epsilonAttr = attrs == nullptr ? nullptr : attrs->GetFloat(EPSILON_ATTR_INDEX);
    return epsilonAttr == nullptr ? DEFAULT_EPSILON : *epsilonAttr;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillShapeDerivedFields(const std::string &opName, ContractInput &input)
{
    const auto &qkvShape = input.qkv.shape;
    const auto &cosSinShape = input.cosSin.shape;
    const auto &cuSeqLenShape = input.cuSeqLen.shape;
    const auto &kCacheShape = input.kCache.shape;
    input.totalTokens = static_cast<uint64_t>(qkvShape.GetDim(1));
    input.headDim = static_cast<uint64_t>(qkvShape.GetDim(2));
    input.batch = static_cast<uint64_t>(cuSeqLenShape.GetDim(0)) - 1;
    input.maxSeqLen = static_cast<uint64_t>(cosSinShape.GetDim(0));
    input.blockNum = static_cast<uint64_t>(kCacheShape.GetDim(0));
    input.blockSize = static_cast<uint64_t>(kCacheShape.GetDim(2));
    OP_LOGD(opName.c_str(),
            "QkvRmsNormRopeCacheWithKScale derived shape fields: totalTokens=%lu batch=%lu headDim=%lu "
            "maxSeqLen=%lu blockNum=%lu blockSize=%lu.",
            input.totalTokens, input.batch, input.headDim, input.maxSeqLen, input.blockNum, input.blockSize);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::FillRequiredTensorInputs(gert::TilingContext *context,
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
    OP_CHECK_IF(ToTensorInfo(context, opName, "slotMapping", SLOT_MAPPING_INDEX, input.slotMapping) !=
                    ge::GRAPH_SUCCESS,
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

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillOptionalTensorInputs(gert::TilingContext *context,
                                                                       ContractInput &input)
{
    input.qRotation = ToOptionalTensorInfo(context, Q_ROTATION_INDEX);
    input.kRotation = ToOptionalTensorInfo(context, K_ROTATION_INDEX);
    input.vScale = ToOptionalTensorInfo(context, V_SCALE_INDEX);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::BuildContractInput(gert::TilingContext *context,
                                                                            const std::string &opName,
                                                                            ContractInput &input, float &epsilon)
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

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillTilingData(const ContractInput &input,
                                                             const ContractTiling &contractTiling, float epsilon,
                                                             QkvRmsNormRopeCacheWithKScaleTilingData &tilingData)
{
    tilingData.totalTokens = input.totalTokens;
    tilingData.batch = input.batch;
    tilingData.qHeadNum = contractTiling.numQHeads;
    tilingData.kvHeadNum = contractTiling.numKHeads;
    tilingData.headDim = input.headDim;
    tilingData.blockSize = input.blockSize;
    tilingData.coreTokenTile = contractTiling.coreTokenTile;
    tilingData.coreGroupNum = contractTiling.coreGroupNum;
    tilingData.kvCacheStrideBlock = contractTiling.kvCacheStrideBlock;
    tilingData.kvCacheStrideHead = contractTiling.kvCacheStrideHead;
    tilingData.kvCacheStrideToken = contractTiling.kvCacheStrideToken;
    tilingData.kScaleCacheStrideBlock = contractTiling.kScaleCacheStrideBlock;
    tilingData.kScaleCacheStrideHead = contractTiling.kScaleCacheStrideHead;
    tilingData.kScaleCacheStrideToken = contractTiling.kScaleCacheStrideToken;
    tilingData.tokenTile = contractTiling.tokenTile;
    tilingData.epsilon = epsilon;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateCompileInfo(
    const std::string &opName, const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo)
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
QkvRmsNormRopeCacheWithKScaleBaseTiling::CopyRawTilingData(gert::TilingContext *context, const std::string &opName,
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

void QkvRmsNormRopeCacheWithKScaleBaseTiling::Reset(gert::TilingContext *context)
{
    TilingBaseClass::Reset(context);
    Reset();
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::Reset()
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

bool QkvRmsNormRopeCacheWithKScaleBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::GetShapeAttrsInfo()
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::GetPlatformInfo()
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

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::DoOpTiling()
{
    const std::string opName = context_->GetNodeName();
    OP_CHECK_IF(BuildContractTiling(input_, aicNum_, contractTiling_, opName) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName.c_str(), "QkvRmsNormRopeCacheWithKScale contract tiling build failed."),
                return ge::GRAPH_FAILED);
    FillTilingData(input_, contractTiling_, epsilon_, tilingData_);
    numBlocks_ = static_cast<uint32_t>(contractTiling_.coreGroupNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t QkvRmsNormRopeCacheWithKScaleBaseTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::GetWorkspaceSize()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);
    OP_CHECK_IF(SetWorkspace(context_, context_->GetNodeName(), *compileInfo_, workspaceSize_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "set QkvRmsNormRopeCacheWithKScale workspace failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::PostTiling()
{
    OP_CHECK_IF(CopyRawTilingData(context_, context_->GetNodeName(), tilingData_, tilingDataSize_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "copy QkvRmsNormRopeCacheWithKScale raw tiling data failed."),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(numBlocks_);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::DumpTilingInfo()
{
    LogTilingData(context_->GetNodeName(), tilingData_, contractTiling_, context_->GetTilingKey(), numBlocks_,
                  tilingDataSize_, workspaceSize_);
}

REGISTER_OPS_TILING_TEMPLATE(QkvRmsNormRopeCacheWithKScale, QkvRmsNormRopeCacheWithKScaleBaseTiling,
                             TILING_TEMPLATE_PRIORITY);

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling
