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
 * \file test_geir_moe_tutel_combine_gates.cpp
 * \brief 用 GE 图模式构建 MoeTutelCombineGates 单算子图并执行 RunGraph。
 *
 * 用法：test_geir_moe_tutel_combine_gates [fp32|fp16]
 * 用例含合法路由与越界路由各若干，越界位置期望输出 0。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
// REG_OP(MoeTutelCombineGates) 定义在本仓公共 proto 头中，本算子目录不重复声明。
#include "../../../common/include/op_graph/op_transformer_proto_extend.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t SAMPLES = 8;  // S
constexpr int64_t HIDDEN = 64;  // H
constexpr int64_t EXPERTS = 2;  // E
constexpr int64_t CAPACITY = 4; // C
constexpr int64_t ROUTES = 1;   // K; Top-1 demonstrates that K may be less than E.
constexpr size_t TIME_BUFFER_SIZE = 64;
constexpr size_t DATA_PATTERN_MULTIPLIER = 7;
constexpr size_t DATA_PATTERN_OFFSET = 3;
constexpr size_t DATA_PATTERN_MODULUS = 100;
constexpr size_t ROUTE_PATTERN_PERIOD = 4;
constexpr size_t EXPERT_UPPER_BOUND_PATTERN = 1;
constexpr size_t LOCATION_UPPER_BOUND_PATTERN = 2;
constexpr int64_t MAX_PRINT_COUNT = 8;
constexpr uint16_t FP16_SIGN_MASK = 0x8000;
constexpr int32_t FP16_EXPONENT_BIAS = 15;
constexpr int32_t FP16_MAX_EXPONENT = 30;
constexpr uint32_t FP16_MANTISSA_BITS = 10;
constexpr uint32_t FP16_MANTISSA_SCALE = 1U << FP16_MANTISSA_BITS;
constexpr float FP16_ROUND_HALFWAY = 0.5F;
constexpr float FP16_MIN_NORMAL = 0x1.0p-14F;
constexpr float FP16_MAX_FINITE = 65504.0F;

string GetTime()
{
    time_t timep;
    if (time(&timep) == static_cast<time_t>(-1)) {
        return {};
    }
    struct tm localTime = {};
    if (localtime_r(&timep, &localTime) == nullptr) {
        return {};
    }
    char tmp[TIME_BUFFER_SIZE] = {};
    if (strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", &localTime) == 0) {
        return {};
    }
    return string(tmp);
}

int32_t FloatToFp16Bits(float value, uint16_t &result)
{
    if (!std::isfinite(value)) {
        return FAILED;
    }
    const uint16_t sign = std::signbit(value) ? FP16_SIGN_MASK : 0;
    const float magnitude = std::fabs(value);
    if (magnitude == 0.0F) {
        result = sign;
        return SUCCESS;
    }
    if (magnitude < FP16_MIN_NORMAL || magnitude > FP16_MAX_FINITE) {
        return FAILED;
    }

    int exponent = 0;
    const float fraction = std::frexp(magnitude, &exponent);
    int32_t halfExponent = exponent - 1 + FP16_EXPONENT_BIAS;
    if (halfExponent <= 0 || halfExponent > FP16_MAX_EXPONENT) {
        return FAILED;
    }
    const float scaledMantissa = (fraction * 2.0F - 1.0F) * static_cast<float>(FP16_MANTISSA_SCALE);
    const float flooredMantissa = std::floor(scaledMantissa);
    uint32_t halfMantissa = static_cast<uint32_t>(flooredMantissa);
    const float remainder = scaledMantissa - flooredMantissa;
    if (remainder > FP16_ROUND_HALFWAY || (remainder == FP16_ROUND_HALFWAY && (halfMantissa & 1U) != 0U)) {
        ++halfMantissa;
    }
    if (halfMantissa == FP16_MANTISSA_SCALE) {
        halfMantissa = 0;
        ++halfExponent;
        if (halfExponent > FP16_MAX_EXPONENT) {
            return FAILED;
        }
    }
    result = static_cast<uint16_t>(sign | static_cast<uint16_t>(halfExponent << FP16_MANTISSA_BITS) |
                                   static_cast<uint16_t>(halfMantissa));
    return SUCCESS;
}

int32_t MakeFloatTensor(const vector<int64_t> &shape, DataType dtype, Format format, Tensor &tensor, TensorDesc &desc)
{
    desc = TensorDesc(ge::Shape(shape), format, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        size *= static_cast<size_t>(shape[i]);
    }
    if (dtype == ge::DT_FLOAT) {
        vector<float> data(size, 0.0F);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<float>((i * DATA_PATTERN_MULTIPLIER + DATA_PATTERN_OFFSET) % DATA_PATTERN_MODULUS) /
                      static_cast<float>(DATA_PATTERN_MODULUS);
        }
        tensor = Tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(float));
        return SUCCESS;
    }
    if (dtype != ge::DT_FLOAT16) {
        return FAILED;
    }
    vector<uint16_t> data(size, 0);
    for (size_t i = 0; i < size; i++) {
        const float value =
            static_cast<float>((i * DATA_PATTERN_MULTIPLIER + DATA_PATTERN_OFFSET) % DATA_PATTERN_MODULUS) /
            static_cast<float>(DATA_PATTERN_MODULUS);
        if (FloatToFp16Bits(value, data[i]) != SUCCESS) {
            return FAILED;
        }
    }
    tensor = Tensor(desc, reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(uint16_t));
    return SUCCESS;
}

int32_t MakeInt32Tensor(const vector<int64_t> &shape, const vector<int32_t> &values, Tensor &tensor, TensorDesc &desc)
{
    desc = TensorDesc(ge::Shape(shape), FORMAT_ND, DT_INT32);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    tensor = Tensor(desc, reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(int32_t));
    return SUCCESS;
}
} // namespace

int main(int argc, char *argv[])
{
    const char *graphName = "tc_ge_irrun_moe_tutel_combine_gates";
    Graph graph(graphName);

    DataType floatDtype = DT_FLOAT;
    if (argc > 2 || (argc == 2 && argv[1] == nullptr)) {
        printf("Usage: %s [fp32|fp16]\n", argv[0]);
        return FAILED;
    }
    if (argc == 2) {
        const string dtypeArg(argv[1]);
        if (dtypeArg == "fp16") {
            floatDtype = DT_FLOAT16;
        } else if (dtypeArg != "fp32") {
            printf("Usage: %s [fp32|fp16]\n", argv[0]);
            return FAILED;
        }
    }

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    Tensor tensorX;
    Tensor tensorY;
    Tensor tensorIndices;
    Tensor tensorLocations;
    TensorDesc descX;
    TensorDesc descY;
    TensorDesc descIndices;
    TensorDesc descLocations;

    if (MakeFloatTensor({SAMPLES, HIDDEN}, floatDtype, FORMAT_ND, tensorX, descX) != SUCCESS ||
        MakeFloatTensor({EXPERTS, CAPACITY, HIDDEN}, floatDtype, FORMAT_ND, tensorY, descY) != SUCCESS) {
        printf("%s - ERROR - [XIR]: MakeFloatTensor failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    // 路由值刻意混入越界项（-1 与 EXPERTS / CAPACITY），这些位置期望输出 0。
    vector<int32_t> indicesValues(static_cast<size_t>(ROUTES * SAMPLES), 0);
    vector<int32_t> locationsValues(static_cast<size_t>(ROUTES * SAMPLES), 0);
    for (size_t i = 0; i < indicesValues.size(); i++) {
        if (i % ROUTE_PATTERN_PERIOD == 0) {
            indicesValues[i] = -1; // index 负越界
            locationsValues[i] = static_cast<int32_t>(i % CAPACITY);
        } else if (i % ROUTE_PATTERN_PERIOD == EXPERT_UPPER_BOUND_PATTERN) {
            indicesValues[i] = static_cast<int32_t>(EXPERTS); // index 上越界
            locationsValues[i] = static_cast<int32_t>(i % CAPACITY);
        } else if (i % ROUTE_PATTERN_PERIOD == LOCATION_UPPER_BOUND_PATTERN) {
            indicesValues[i] = static_cast<int32_t>(i % EXPERTS);
            locationsValues[i] = static_cast<int32_t>(CAPACITY); // location 上越界
        } else {
            indicesValues[i] = static_cast<int32_t>(i % EXPERTS); // 合法路由
            locationsValues[i] = static_cast<int32_t>(i % CAPACITY);
        }
    }
    if (MakeInt32Tensor({ROUTES, SAMPLES}, indicesValues, tensorIndices, descIndices) != SUCCESS ||
        MakeInt32Tensor({ROUTES, SAMPLES}, locationsValues, tensorLocations, descLocations) != SUCCESS) {
        printf("%s - ERROR - [XIR]: MakeInt32Tensor failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    auto xData = op::Data("placeholder_x").set_attr_index(0);
    xData.update_input_desc_x(descX);
    xData.update_output_desc_y(descX);
    auto yData = op::Data("placeholder_y_grad").set_attr_index(1);
    yData.update_input_desc_x(descY);
    yData.update_output_desc_y(descY);
    auto indicesData = op::Data("placeholder_indices").set_attr_index(2);
    indicesData.update_input_desc_x(descIndices);
    indicesData.update_output_desc_y(descIndices);
    auto locationsData = op::Data("placeholder_locations").set_attr_index(3);
    locationsData.update_input_desc_x(descLocations);
    locationsData.update_output_desc_y(descLocations);

    auto combineGates = op::MoeTutelCombineGates("moe_tutel_combine_gates");
    combineGates.set_input_x(xData);
    combineGates.set_input_y_grad(yData);
    combineGates.set_input_indices(indicesData);
    combineGates.set_input_locations(locationsData);

    vector<Operator> inputsOp = {xData, yData, indicesData, locationsData};
    vector<Operator> outputsOp = {combineGates};
    graph.SetInputs(inputsOp).SetOutputs(outputsOp);

    map<AscendString, AscendString> buildOptions = {};
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(buildOptions));
    if (session == nullptr) {
        printf("Create session failed\n");
        (void)ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graphOptions = {};
    const uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed ret=%u\n", GetTime().c_str(), ret);
        session.reset();
        (void)ge::GEFinalize();
        return FAILED;
    }

    vector<ge::Tensor> input = {tensorX, tensorY, tensorIndices, tensorLocations};
    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed ret=%u\n", GetTime().c_str(), ret);
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: error: %s\n", GetTime().c_str(), errorMsg.GetString());
        session.reset();
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    for (size_t i = 0; i < output.size(); i++) {
        const int64_t outSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%zu] dtype=%d shapeSize=%ld\n", i, output[i].GetTensorDesc().GetDataType(), outSize);
        if (output[i].GetTensorDesc().GetDataType() == ge::DT_FLOAT) {
            const float *p = reinterpret_cast<const float *>(output[i].GetData());
            const int64_t printCnt = outSize < MAX_PRINT_COUNT ? outSize : MAX_PRINT_COUNT;
            for (int64_t j = 0; j < printCnt; j++) {
                printf("  result[%ld] = %f\n", j, p[j]);
            }
        }
    }

    session.reset();
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("GEFinalize failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());
    return SUCCESS;
}
