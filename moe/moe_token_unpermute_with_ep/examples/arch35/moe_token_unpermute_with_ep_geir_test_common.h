/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_TOKEN_UNPERMUTE_WITH_EP_GEIR_TEST_COMMON_H
#define MOE_TOKEN_UNPERMUTE_WITH_EP_GEIR_TEST_COMMON_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "moe_token_unpermute_with_ep_proto.h"

namespace moe_token_unpermute_with_ep_geir_test {

using Shape = std::vector<int64_t>;

struct CaseSpec {
    std::string name;
    int64_t tokenNum;
    int64_t hidden;
    int64_t topK;
    std::vector<int64_t> range;
    bool hasProbs;
    ge::DataType tokenDtype;
    ge::DataType probsDtype;
};

struct DeclaredShapes {
    Shape tokens;
    Shape indices;
    Shape probs;
};

inline bool IsEp(const CaseSpec &spec)
{
    return spec.range.size() == 2U && !(spec.range[0] == 0 && spec.range[1] == 0);
}

inline int64_t IndicesCount(const CaseSpec &spec)
{
    return spec.hasProbs || IsEp(spec) ? spec.tokenNum * spec.topK : spec.tokenNum;
}

inline int64_t TokenRows(const CaseSpec &spec)
{
    return IsEp(spec) ? spec.range[1] - spec.range[0] : IndicesCount(spec);
}

inline Shape TokensShape(const CaseSpec &spec)
{
    return {TokenRows(spec), spec.hidden};
}

inline Shape IndicesShape(const CaseSpec &spec)
{
    return {IndicesCount(spec)};
}

inline Shape ProbsShape(const CaseSpec &spec)
{
    return spec.hasProbs ? Shape{spec.tokenNum, spec.topK} : Shape{};
}

inline Shape OutputShape(const CaseSpec &spec)
{
    return {spec.tokenNum, spec.hidden};
}

inline int64_t ElementCount(const Shape &shape)
{
    int64_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    return count;
}

inline uint16_t FloatToHalfBits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    uint32_t mantissa = bits & 0x7FFFFFU;
    const uint32_t sourceExponent = (bits >> 23U) & 0xFFU;
    if (sourceExponent == 0xFFU) {
        return static_cast<uint16_t>(sign | 0x7C00U | (mantissa == 0U ? 0U : 0x0200U));
    }
    int32_t exponent = static_cast<int32_t>(sourceExponent) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        const uint32_t rounded = mantissa + ((1U << (shift - 1U)) - 1U) + ((mantissa >> shift) & 1U);
        return static_cast<uint16_t>(sign | (rounded >> shift));
    }
    mantissa += 0xFFFU + ((mantissa >> 13U) & 1U);
    if ((mantissa & 0x800000U) != 0U) {
        mantissa = 0U;
        ++exponent;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
}

inline float HalfBitsToFloat(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    uint32_t exponent = (bits >> 10U) & 0x1FU;
    uint32_t mantissa = bits & 0x03FFU;
    uint32_t value;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            value = sign;
        } else {
            int32_t normalizedExponent = -14;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --normalizedExponent;
            }
            mantissa &= 0x03FFU;
            value = sign | (static_cast<uint32_t>(normalizedExponent + 127) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        value = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        value = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline uint16_t FloatToBfloat16Bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

inline float Bfloat16BitsToFloat(uint16_t bits)
{
    uint32_t value = static_cast<uint32_t>(bits) << 16U;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline size_t DtypeSize(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return sizeof(float);
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return sizeof(uint16_t);
    }
    if (dtype == ge::DT_INT32) {
        return sizeof(int32_t);
    }
    return 0U;
}

inline std::vector<uint8_t> EncodeFloats(const std::vector<float> &values, ge::DataType dtype)
{
    std::vector<uint8_t> bytes(values.size() * DtypeSize(dtype));
    if (dtype == ge::DT_FLOAT) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }
    auto *data = reinterpret_cast<uint16_t *>(bytes.data());
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = dtype == ge::DT_FLOAT16 ? FloatToHalfBits(values[i]) : FloatToBfloat16Bits(values[i]);
    }
    return bytes;
}

inline std::vector<float> DecodeFloats(const uint8_t *data, size_t count, ge::DataType dtype)
{
    std::vector<float> result(count);
    if (dtype == ge::DT_FLOAT) {
        std::memcpy(result.data(), data, count * sizeof(float));
        return result;
    }
    const auto *values = reinterpret_cast<const uint16_t *>(data);
    for (size_t i = 0; i < count; ++i) {
        result[i] = dtype == ge::DT_FLOAT16 ? HalfBitsToFloat(values[i]) : Bfloat16BitsToFloat(values[i]);
    }
    return result;
}

inline float Quantize(float value, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return HalfBitsToFloat(FloatToHalfBits(value));
    }
    if (dtype == ge::DT_BF16) {
        return Bfloat16BitsToFloat(FloatToBfloat16Bits(value));
    }
    return value;
}

inline std::string ShapeJson(const Shape &shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += (i == 0U ? "" : ",") + std::to_string(shape[i]);
    }
    return result + "]";
}

inline std::string ShapeBundleJson(const Shape &tokens, const Shape &indices, const Shape &probs)
{
    return "[" + ShapeJson(tokens) + "," + ShapeJson(indices) + "," + ShapeJson(probs) + "]";
}

inline ge::TensorDesc MakeDesc(const Shape &shape, ge::DataType dtype)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(static_cast<int64_t>(shape.size()));
    return desc;
}

inline ge::Tensor MakeFloatTensor(const Shape &shape, ge::DataType dtype, const std::vector<float> &values)
{
    return ge::Tensor(MakeDesc(shape, dtype), EncodeFloats(values, dtype));
}

inline ge::Tensor MakeIntTensor(const Shape &shape, const std::vector<int32_t> &values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(int32_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return ge::Tensor(MakeDesc(shape, ge::DT_INT32), bytes);
}

inline std::vector<float> MakeTokenValues(const CaseSpec &spec)
{
    std::vector<float> values(static_cast<size_t>(TokenRows(spec) * spec.hidden));
    for (int64_t row = 0; row < TokenRows(spec); ++row) {
        for (int64_t col = 0; col < spec.hidden; ++col) {
            const float value = static_cast<float>(((row + 1) * 2 + (col % 5)) - 4) / 4.0F;
            values[static_cast<size_t>(row * spec.hidden + col)] = Quantize(value, spec.tokenDtype);
        }
    }
    return values;
}

inline std::vector<int32_t> MakeIndices(const CaseSpec &spec)
{
    const int64_t count = IndicesCount(spec);
    std::vector<int32_t> values(static_cast<size_t>(count));
    if (!IsEp(spec)) {
        for (int64_t i = 0; i < count; ++i) {
            values[static_cast<size_t>(i)] = static_cast<int32_t>(count - i - 1);
        }
        return values;
    }
    std::vector<int32_t> candidates = {
        static_cast<int32_t>(spec.range[0]),
        static_cast<int32_t>(spec.range[1] - 1),
    };
    if (spec.range[0] > 0) {
        candidates.push_back(static_cast<int32_t>(spec.range[0] - 1));
    }
    if (spec.range[1] < count) {
        candidates.push_back(static_cast<int32_t>(spec.range[1]));
    }
    for (int64_t i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = candidates[static_cast<size_t>(i) % candidates.size()];
    }
    return values;
}

inline std::vector<float> MakeProbValues(const CaseSpec &spec)
{
    std::vector<float> values(static_cast<size_t>(spec.tokenNum * spec.topK));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = Quantize(static_cast<float>((i % 3U) + 1U) / 4.0F, spec.probsDtype);
    }
    return values;
}

inline std::vector<float> Golden(const CaseSpec &spec, const std::vector<float> &tokens,
                                 const std::vector<int32_t> &indices, const std::vector<float> &probs)
{
    std::vector<float> output(static_cast<size_t>(spec.tokenNum * spec.hidden), 0.0F);
    const int64_t effectiveTopK = spec.hasProbs || IsEp(spec) ? spec.topK : 1;
    for (int64_t token = 0; token < spec.tokenNum; ++token) {
        for (int64_t k = 0; k < effectiveTopK; ++k) {
            const int64_t slot = token * effectiveTopK + k;
            const int64_t globalIndex = indices[static_cast<size_t>(slot)];
            if (IsEp(spec) && (globalIndex < spec.range[0] || globalIndex >= spec.range[1])) {
                continue;
            }
            const int64_t localIndex = IsEp(spec) ? globalIndex - spec.range[0] : globalIndex;
            const float weight = spec.hasProbs ? probs[static_cast<size_t>(slot)] : 1.0F;
            for (int64_t hidden = 0; hidden < spec.hidden; ++hidden) {
                const size_t outIndex = static_cast<size_t>(token * spec.hidden + hidden);
                const size_t tokenIndex = static_cast<size_t>(localIndex * spec.hidden + hidden);
                output[outIndex] += tokens[tokenIndex] * weight;
            }
        }
    }
    for (auto &value : output) {
        value = Quantize(value, spec.tokenDtype);
    }
    return output;
}

inline bool BuildGraph(const std::string &scenario, const CaseSpec &spec, const DeclaredShapes &declared,
                       ge::Graph &graph)
{
    auto tokensDesc = MakeDesc(declared.tokens, spec.tokenDtype);
    auto indicesDesc = MakeDesc(declared.indices, ge::DT_INT32);
    auto probsDesc = MakeDesc(declared.probs, spec.probsDtype);
    auto tokensData = ge::op::Data((scenario + "_tokens").c_str()).set_attr_index(0);
    auto indicesData = ge::op::Data((scenario + "_indices").c_str()).set_attr_index(1);
    tokensData.update_input_desc_x(tokensDesc);
    tokensData.update_output_desc_y(tokensDesc);
    indicesData.update_input_desc_x(indicesDesc);
    indicesData.update_output_desc_y(indicesDesc);
    graph.AddOp(tokensData);
    graph.AddOp(indicesData);
    std::vector<ge::Operator> inputs = {tokensData, indicesData};

    auto node = ge::op::MoeTokenUnpermuteWithEp(("MoeTokenUnpermuteWithEp_" + scenario).c_str());
    node.set_input_permuted_tokens(tokensData);
    node.set_input_sorted_indices(indicesData);
    node.update_input_desc_permuted_tokens(tokensDesc);
    node.update_input_desc_sorted_indices(indicesDesc);
    if (spec.hasProbs) {
        auto probsData = ge::op::Data((scenario + "_probs").c_str()).set_attr_index(2);
        probsData.update_input_desc_x(probsDesc);
        probsData.update_output_desc_y(probsDesc);
        graph.AddOp(probsData);
        inputs.push_back(probsData);
        node.set_input_probs(probsData);
        node.update_input_desc_probs(probsDesc);
    }
    node.set_attr_num_topk(spec.topK);
    node.set_attr_range(spec.range);
    node.set_attr_padded_mode(false);
    node.set_attr_restore_shape(std::vector<int64_t>{});

    Shape declaredOutput;
    if (declared.tokens == Shape{-2}) {
        declaredOutput = {-2};
    } else if (declared.tokens.size() == 2U &&
               (declared.tokens[0] < 0 || declared.tokens[1] < 0)) {
        declaredOutput = {-1, -1};
    } else {
        declaredOutput = OutputShape(spec);
    }
    node.update_output_desc_unpermuted_tokens(MakeDesc(declaredOutput, spec.tokenDtype));
    graph.SetInputs(inputs).SetOutputs({node});
    return true;
}

inline bool RunAndVerify(ge::Session &session, uint32_t graphId, const CaseSpec &spec, double &durationUs)
{
    const auto tokens = MakeTokenValues(spec);
    const auto indices = MakeIndices(spec);
    const auto probs = spec.hasProbs ? MakeProbValues(spec) : std::vector<float>{};
    std::vector<ge::Tensor> inputs;
    inputs.push_back(MakeFloatTensor(TokensShape(spec), spec.tokenDtype, tokens));
    inputs.push_back(MakeIntTensor(IndicesShape(spec), indices));
    if (spec.hasProbs) {
        inputs.push_back(MakeFloatTensor(ProbsShape(spec), spec.probsDtype, probs));
    }

    std::vector<ge::Tensor> outputs;
    const auto start = std::chrono::steady_clock::now();
    const auto status = session.RunGraph(graphId, inputs, outputs);
    const auto end = std::chrono::steady_clock::now();
    durationUs = std::chrono::duration<double, std::micro>(end - start).count();
    if (status != ge::SUCCESS || outputs.size() != 1U) {
        std::cerr << "ERROR: RunGraph status=" << status << ", output_count=" << outputs.size() << std::endl;
        return false;
    }

    const auto outputDesc = outputs[0].GetTensorDesc();
    if (outputDesc.GetShape().GetDims() != OutputShape(spec) || outputDesc.GetDataType() != spec.tokenDtype) {
        std::cerr << "ERROR: output shape or dtype mismatch" << std::endl;
        return false;
    }
    const auto expected = Golden(spec, tokens, indices, probs);
    const auto actual = DecodeFloats(outputs[0].GetData(), expected.size(), spec.tokenDtype);
    const float tolerance = spec.tokenDtype == ge::DT_FLOAT ? 1.0e-4F : 1.0e-3F;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::cerr << "ERROR: value mismatch at " << i << ", actual=" << actual[i]
                      << ", expected=" << expected[i] << std::endl;
            return false;
        }
    }
    const auto shapeBundle = ShapeBundleJson(TokensShape(spec), IndicesShape(spec), ProbsShape(spec));
    std::cout << "Shape, dtype and values PASSED for " << shapeBundle << std::endl;
    std::cout << "GEIR_PERF case=" << spec.name << " end_to_end_us=" << durationUs << std::endl;
    return true;
}

inline DeclaredShapes ConcreteDeclaredShapes(const CaseSpec &spec)
{
    return {TokensShape(spec), IndicesShape(spec), ProbsShape(spec)};
}

inline int32_t DeviceId()
{
    const char *text = std::getenv("ASCEND_DEVICE_ID");
    return text == nullptr ? 0 : static_cast<int32_t>(std::strtol(text, nullptr, 10));
}

inline bool InitializeGe()
{
    const std::string device = std::to_string(DeviceId());
    std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", device.c_str()},
        {"ge.graphRunMode", "1"},
        {"ge.exec.dynamicGraphExecuteMode", "dynamic_execute"},
    };
    const auto status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        std::cerr << "ERROR: GEInitialize status=" << status << std::endl;
        return false;
    }
    return true;
}

}  // namespace moe_token_unpermute_with_ep_geir_test

#endif  // MOE_TOKEN_UNPERMUTE_WITH_EP_GEIR_TEST_COMMON_H
