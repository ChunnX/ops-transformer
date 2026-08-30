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
 * \file test_geir_moe_tutel_combine_x.cpp
 * \brief MoeTutelCombineX GE 图模式调用样例（Ascend950）。
 */

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "op_graph/op_transformer_proto_extend.h"
#include "tensor.h"
#include "types.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t EXPERT_NUM = 4;
constexpr int64_t CAPACITY = 8;
constexpr int64_t HIDDEN = 64;
constexpr int64_t TOP_K = 2;
constexpr int64_t SAMPLES = 16;
constexpr int64_t OUTPUT_PREVIEW_COUNT = 6;
constexpr int64_t FLOAT_PATTERN_MULTIPLIER = 7;
constexpr int64_t FLOAT_PATTERN_OFFSET = 3;
constexpr int64_t FLOAT_PATTERN_MODULUS = 100;
constexpr float FLOAT_PATTERN_NORMALIZER = 100.0F;
constexpr int64_t INVALID_ROUTE_PERIOD = 4;
constexpr int64_t INVALID_ROUTE_SLOT = 3;

string GetTime()
{
    std::time_t timep = 0;
    if (std::time(&timep) == static_cast<std::time_t>(-1)) {
        return "unknown-time";
    }
    std::tm timeInfo = {};
    if (localtime_r(&timep, &timeInfo) == nullptr) {
        return "unknown-time";
    }
    char tmp[64] = {0};
    if (std::strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", &timeInfo) == 0) {
        return "unknown-time";
    }
    return string(tmp);
}

int64_t ShapeSize(const vector<int64_t> &shape)
{
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    return size;
}

Tensor MakeFloatTensor(const vector<int64_t> &shape, TensorDesc &desc)
{
    desc.SetRealDimCnt(shape.size());
    int64_t size = ShapeSize(shape);
    vector<float> data(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; i++) {
        data[static_cast<size_t>(i)] =
            static_cast<float>((i * FLOAT_PATTERN_MULTIPLIER + FLOAT_PATTERN_OFFSET) % FLOAT_PATTERN_MODULUS) /
            FLOAT_PATTERN_NORMALIZER;
    }
    return Tensor(desc, reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float));
}

// 每四条路由中构造一条无效路由（indices=-1 / locations=CAPACITY），
// 用来在图模式下同时打通正常分支与drop/no-expert分支。
Tensor MakeRouteTensor(const vector<int64_t> &shape, TensorDesc &desc, bool isIndices)
{
    desc.SetRealDimCnt(shape.size());
    int64_t size = ShapeSize(shape);
    vector<int32_t> data(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; i++) {
        bool invalid = (i % INVALID_ROUTE_PERIOD == INVALID_ROUTE_SLOT);
        if (isIndices) {
            data[static_cast<size_t>(i)] = invalid ? -1 : static_cast<int32_t>(i % EXPERT_NUM);
        } else {
            data[static_cast<size_t>(i)] =
                invalid ? static_cast<int32_t>(CAPACITY) : static_cast<int32_t>(i % CAPACITY);
        }
    }
    return Tensor(desc, reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(int32_t));
}

bool AddOpChecked(Graph &graph, const Operator &op, const char *opName)
{
    const graphStatus status = graph.AddOp(op);
    if (status != GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: AddOp %s failed ret=%u\n", GetTime().c_str(), opName, status);
        return false;
    }
    return true;
}
} // namespace

int main()
{
    const char *graphName = "tc_ge_irrun_moe_tutel_combine_x";
    Graph graph(graphName);
    vector<Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<int64_t> yGradShape = {EXPERT_NUM, CAPACITY, HIDDEN};
    vector<int64_t> routeShape = {TOP_K, SAMPLES};

    auto yGradData = op::Data("placeholder_y_grad").set_attr_index(0);
    TensorDesc yGradDesc(Shape(yGradShape), FORMAT_ND, DT_FLOAT);
    yGradDesc.SetPlacement(kPlacementHost);
    input.push_back(MakeFloatTensor(yGradShape, yGradDesc));
    yGradData.update_input_desc_x(yGradDesc);
    yGradData.update_output_desc_y(yGradDesc);
    if (!AddOpChecked(graph, yGradData, "placeholder_y_grad")) {
        static_cast<void>(GEFinalize());
        return FAILED;
    }

    auto gatesData = op::Data("placeholder_gates").set_attr_index(1);
    TensorDesc gatesDesc(Shape(routeShape), FORMAT_ND, DT_FLOAT);
    gatesDesc.SetPlacement(kPlacementHost);
    input.push_back(MakeFloatTensor(routeShape, gatesDesc));
    gatesData.update_input_desc_x(gatesDesc);
    gatesData.update_output_desc_y(gatesDesc);
    if (!AddOpChecked(graph, gatesData, "placeholder_gates")) {
        static_cast<void>(GEFinalize());
        return FAILED;
    }

    auto indicesData = op::Data("placeholder_indices").set_attr_index(2);
    TensorDesc indicesDesc(Shape(routeShape), FORMAT_ND, DT_INT32);
    indicesDesc.SetPlacement(kPlacementHost);
    input.push_back(MakeRouteTensor(routeShape, indicesDesc, true));
    indicesData.update_input_desc_x(indicesDesc);
    indicesData.update_output_desc_y(indicesDesc);
    if (!AddOpChecked(graph, indicesData, "placeholder_indices")) {
        static_cast<void>(GEFinalize());
        return FAILED;
    }

    auto locationsData = op::Data("placeholder_locations").set_attr_index(3);
    TensorDesc locationsDesc(Shape(routeShape), FORMAT_ND, DT_INT32);
    locationsDesc.SetPlacement(kPlacementHost);
    input.push_back(MakeRouteTensor(routeShape, locationsDesc, false));
    locationsData.update_input_desc_x(locationsDesc);
    locationsData.update_output_desc_y(locationsDesc);
    if (!AddOpChecked(graph, locationsData, "placeholder_locations")) {
        static_cast<void>(GEFinalize());
        return FAILED;
    }

    auto combineOp = op::MoeTutelCombineX("moe_tutel_combine_x");
    combineOp.set_input_y_grad(yGradData);
    combineOp.set_input_gates(gatesData);
    combineOp.set_input_indices(indicesData);
    combineOp.set_input_locations(locationsData);

    vector<Operator> inputs = {yGradData, gatesData, indicesData, locationsData};
    vector<Operator> outputs = {combineOp};
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> buildOptions = {};
    Session *session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("Create session failed\n");
        static_cast<void>(GEFinalize());
        return FAILED;
    }

    map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed ret=%u\n", GetTime().c_str(), ret);
        delete session;
        static_cast<void>(GEFinalize());
        return FAILED;
    }

    vector<Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed ret=%u\n", GetTime().c_str(), ret);
        AscendString errorMsg = GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: error: %s\n", GetTime().c_str(), errorMsg.GetString());
        delete session;
        static_cast<void>(GEFinalize());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success, output_num=%zu\n", GetTime().c_str(), output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t outSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%zu] shapeSize=%ld\n", i, outSize);
        const uint8_t *rawData = output[i].GetData();
        if (outSize > 0 && rawData == nullptr) {
            printf("%s - ERROR - [XIR]: output[%zu] data is null for %ld elements\n", GetTime().c_str(), i, outSize);
            delete session;
            static_cast<void>(GEFinalize());
            return FAILED;
        }
        const float *p = reinterpret_cast<const float *>(rawData);
        int64_t printCnt = outSize < OUTPUT_PREVIEW_COUNT ? outSize : OUTPUT_PREVIEW_COUNT;
        for (int64_t j = 0; j < printCnt; j++) {
            printf("  x_grad[%ld] = %f\n", j, p[j]);
        }
    }

    delete session;
    if (GEFinalize() != SUCCESS) {
        printf("GEFinalize failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());
    return SUCCESS;
}
