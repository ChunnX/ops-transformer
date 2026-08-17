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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
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
#include "op_transformer_proto_extend.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t SAMPLES = 8;  // S
constexpr int64_t HIDDEN = 64;  // H
constexpr int64_t EXPERTS = 2;  // E（同时是路由轴长度）
constexpr int64_t CAPACITY = 4; // C

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return string(tmp);
}

uint16_t FloatToFp16Bits(float val)
{
    uint32_t bits = 0;
    memcpy(&bits, &val, sizeof(uint32_t));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    const uint16_t exponent = static_cast<uint16_t>((bits >> 23) & 0xff);
    const uint32_t mantissa = bits & 0x7fffff;
    if (exponent == 0) {
        return sign;
    }
    if (exponent >= 113 && exponent <= 142) {
        return static_cast<uint16_t>(sign | ((exponent - 112) << 10) | (mantissa >> 13));
    }
    if (exponent < 113) {
        if (exponent >= 103) {
            const uint32_t shifted = (mantissa | 0x800000) >> (113 - exponent);
            return static_cast<uint16_t>(sign | static_cast<uint16_t>(shifted));
        }
        return sign;
    }
    return static_cast<uint16_t>(sign | 0x7c00);
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
    const uint32_t byteSize = (dtype == ge::DT_FLOAT) ? 4U : 2U;
    vector<uint8_t> data(size * byteSize, 0);
    for (size_t i = 0; i < size; i++) {
        const float val = static_cast<float>((i * 7 + 3) % 100) / 100.0f;
        if (dtype == ge::DT_FLOAT) {
            memcpy(data.data() + i * 4, &val, sizeof(float));
        } else {
            const uint16_t half = FloatToFp16Bits(val);
            memcpy(data.data() + i * 2, &half, sizeof(uint16_t));
        }
    }
    tensor = Tensor(desc, data.data(), data.size());
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
    if (argc > 1 && string(argv[1]) == "fp16") {
        floatDtype = DT_FLOAT16;
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

    (void)MakeFloatTensor({SAMPLES, HIDDEN}, floatDtype, FORMAT_ND, tensorX, descX);
    (void)MakeFloatTensor({EXPERTS, CAPACITY, HIDDEN}, floatDtype, FORMAT_ND, tensorY, descY);

    // 路由值刻意混入越界项（-1 与 EXPERTS / CAPACITY），这些位置期望输出 0。
    vector<int32_t> indicesValues(static_cast<size_t>(EXPERTS * SAMPLES), 0);
    vector<int32_t> locationsValues(static_cast<size_t>(EXPERTS * SAMPLES), 0);
    for (size_t i = 0; i < indicesValues.size(); i++) {
        if (i % 4 == 0) {
            indicesValues[i] = -1; // index 负越界
            locationsValues[i] = static_cast<int32_t>(i % CAPACITY);
        } else if (i % 4 == 1) {
            indicesValues[i] = static_cast<int32_t>(EXPERTS); // index 上越界
            locationsValues[i] = static_cast<int32_t>(i % CAPACITY);
        } else if (i % 4 == 2) {
            indicesValues[i] = static_cast<int32_t>(i % EXPERTS);
            locationsValues[i] = static_cast<int32_t>(CAPACITY); // location 上越界
        } else {
            indicesValues[i] = static_cast<int32_t>(i % EXPERTS); // 合法路由
            locationsValues[i] = static_cast<int32_t>(i % CAPACITY);
        }
    }
    (void)MakeInt32Tensor({EXPERTS, SAMPLES}, indicesValues, tensorIndices, descIndices);
    (void)MakeInt32Tensor({EXPERTS, SAMPLES}, locationsValues, tensorLocations, descLocations);

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
    ge::Session *session = new (std::nothrow) Session(buildOptions);
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
        delete session;
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
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    for (size_t i = 0; i < output.size(); i++) {
        const int64_t outSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%zu] dtype=%d shapeSize=%ld\n", i, output[i].GetTensorDesc().GetDataType(), outSize);
        if (output[i].GetTensorDesc().GetDataType() == ge::DT_FLOAT) {
            const float *p = reinterpret_cast<const float *>(output[i].GetData());
            const int64_t printCnt = outSize < 8 ? outSize : 8;
            for (int64_t j = 0; j < printCnt; j++) {
                printf("  result[%ld] = %f\n", j, p[j]);
            }
        }
    }

    delete session;
    session = nullptr;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("GEFinalize failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());
    return SUCCESS;
}
