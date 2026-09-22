/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// GE graph-mode (GEIR) example for MoeTutelCombineX. Builds a single-op graph
// from the operator IR (op_graph/moe_tutel_combine_x_proto.h) with the micro
// numerical case documented in README.md (功能说明), runs it through a ge
// Session and checks the output against the reference formula:
//   x_grad[t, :] = gates[t] * y_grad[indices[t] * C + locations[t], :]
// with E = 2 experts, C = EC / E = 2 (EC = 4 rows), H = 2, S = 4, so the
// expected result is [0.2, 2, 0.6, 6, 1.2, 12, 0.4, 4].
//
// Prerequisites: the operator package built by `build.sh --pkg` has been
// installed into the runtime environment — the ge runtime discovers the
// custom operator from the installed vendors package.
//
// Run from the repository root:
//   bash build.sh --run_example moe_tutel_combine_x graph --soc=ascend950
// =============================================================================

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/moe_tutel_combine_x_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// Documented micro case (README.md 功能说明, no-batch form): E=2 experts,
// capacity C = EC / E = 2 with EC = 4 expert-buffer rows, hidden H = 2 and
// S = 4 token copies.
const int64_t EC = 4; // y_grad rows of the expert buffer (E * C)
const int64_t H = 2;  // hidden dim (last dim of y_grad)
const int64_t S = 4;  // token copies (last dim of gates/indices/locations)

// Input data of the micro case. y_grad rows are kept distinct so the routing
// (indices/locations) is visible in the result; gates are 0.1/0.2/0.3/0.4.
const vector<int64_t> yGradShape = {EC, H};
const vector<float> yGradData = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f};
const vector<int64_t> gatesShape = {S};
const vector<float> gatesData = {0.1f, 0.2f, 0.3f, 0.4f};
const vector<int32_t> indicesData = {0, 1, 1, 0};
const vector<int32_t> locationsData = {1, 0, 1, 0};
const vector<int64_t> xGradShape = {S, H};

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        return twoByte;
    } else if (dt == ge::DT_BF16) {
        return twoByte;
    } else if (dt == ge::DT_INT32) {
        return fourByte;
    } else if (dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64) {
        return eightByte;
    } else if (dt == ge::DT_UINT64) {
        return eightByte;
    } else if (dt == ge::DT_INT8) {
        return oneByte;
    } else if (dt == ge::DT_UINT8) {
        return oneByte;
    }
    return fourByte;
}

// Fills a host Tensor with the given typed data. The generic examples use a
// constant fill; this operator needs routing metadata, so real data is used.
template <typename T>
int32_t GenTensorData(const vector<int64_t> &shapes, Tensor &inputTensor, TensorDesc &inputTensorDesc,
                      DataType dataType, const vector<T> &hostData)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (hostData.size() != size) {
        printf("%s - ERROR - [XIR]: host data size %zu does not match tensor size %zu\n", GetTime().c_str(),
               hostData.size(), size);
        return FAILED;
    }
    uint32_t dataLen = size * GetDataTypeSize(dataType);
    T *pData = new (std::nothrow) T[size];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Alloc host data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = hostData[i];
    }
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<uint8_t *>(pData), dataLen);
    return SUCCESS;
}

#define ADD_INPUT(intputName, intputDtype, inputShape, hostData) \
    vector<int64_t> intputName##_shape = inputShape; \
    auto intputName##_placeholder = op::Data("placeholder_" #intputName).set_attr_index(0); \
    TensorDesc intputName##_desc = TensorDesc(ge::Shape(intputName##_shape), FORMAT_ND, intputDtype); \
    intputName##_desc.SetPlacement(ge::kPlacementHost); \
    intputName##_desc.SetFormat(FORMAT_ND); \
    Tensor tensor_##intputName; \
    ret = GenTensorData(intputName##_shape, tensor_##intputName, intputName##_desc, intputDtype, hostData); \
    if (ret != SUCCESS) { \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str()); \
        return FAILED; \
    } \
    intputName##_placeholder.update_input_desc_x(intputName##_desc); \
    input.push_back(tensor_##intputName); \
    graph.AddOp(intputName##_placeholder); \
    mtcxOp.set_input_##intputName(intputName##_placeholder); \
    inputs.push_back(intputName##_placeholder);

#define ADD_OUTPUT(outputName, outputDtype, outputShape) \
    TensorDesc outputName##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    mtcxOp.update_output_desc_##outputName(outputName##_desc);

int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs, std::vector<Operator> &outputs,
                     Graph &graph)
{
    int32_t ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto mtcxOp = op::MoeTutelCombineX("test_geir_moe_tutel_combine_x");

    // 添加输入（顺序严格匹配 proto.h：y_grad、gates、indices、locations）
    ADD_INPUT(y_grad, DT_FLOAT, yGradShape, yGradData);
    ADD_INPUT(gates, DT_FLOAT, gatesShape, gatesData);
    ADD_INPUT(indices, DT_INT32, gatesShape, indicesData);
    ADD_INPUT(locations, DT_INT32, gatesShape, locationsData);

    // 添加输出 x_grad: [S, H]，dtype 跟随 y_grad（本例为 float32）
    ADD_OUTPUT(x_grad, DT_FLOAT, xGradShape);

    outputs.push_back(mtcxOp);
    // 添加完毕
    return SUCCESS;
}

// Reference math transcribed from README.md (功能说明): E = max(indices) + 1,
// C = floor(EC / E) (C = EC when E = 1), srcRow = indices[t] * C +
// locations[t]; entries with indices[t] < 0, locations[t] outside [0, C) or
// srcRow outside [0, EC) are invalid and produce all-zero output rows.
vector<float> GenExpected()
{
    int32_t maxIdx = -1;
    for (size_t i = 0; i < indicesData.size(); i++) {
        maxIdx = std::max(maxIdx, indicesData[i]);
    }
    const int64_t expertNum = static_cast<int64_t>(maxIdx) + 1;
    const int64_t capacity = (expertNum >= 1) ? (EC / expertNum) : 0;
    vector<float> expected(static_cast<size_t>(S * H), 0.0f);
    for (int64_t t = 0; t < S; t++) {
        const int64_t srcRow = static_cast<int64_t>(indicesData[t]) * capacity + locationsData[t];
        const bool valid =
            indicesData[t] >= 0 && locationsData[t] >= 0 && locationsData[t] < capacity && srcRow >= 0 && srcRow < EC;
        for (int64_t h = 0; h < H; h++) {
            expected[static_cast<size_t>(t * H + h)] = valid ? gatesData[t] * yGradData[srcRow * H + h] : 0.0f;
        }
    }
    return expected;
}

int main()
{
    const char *graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to add compute graph\n", GetTime().c_str());
    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        std::cout << "GE error: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    // 1. print the device result
    if (output.size() != 1) {
        printf("%s - ERROR - [XIR]: output tensor count %zu != 1\n", GetTime().c_str(), output.size());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    int64_t outSize = output[0].GetTensorDesc().GetShape().GetShapeSize();
    if (outSize != S * H || output[0].GetTensorDesc().GetDataType() != DT_FLOAT) {
        printf("%s - ERROR - [XIR]: unexpected output desc (size %ld, dtype %d)\n", GetTime().c_str(),
               static_cast<long>(outSize), static_cast<int>(output[0].GetTensorDesc().GetDataType()));
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    const float *result = reinterpret_cast<const float *>(output[0].GetData());
    for (int64_t i = 0; i < outSize; i++) {
        LOG_PRINT("xGrad[%ld] is: %f\n", static_cast<long>(i), result[i]);
    }

    // 2. check against the reference formula (README.md 功能说明)
    const vector<float> expected = GenExpected();
    bool checkPass = true;
    for (int64_t i = 0; i < outSize; i++) {
        const float expect = expected[static_cast<size_t>(i)];
        const float diff = std::fabs(result[i] - expect);
        if (diff > 1e-5f * std::max(1.0f, std::fabs(expect))) {
            printf("%s - ERROR - [XIR]: xGrad[%ld] mismatch: got %f, expect %f\n", GetTime().c_str(),
                   static_cast<long>(i), result[i], expect);
            checkPass = false;
        }
    }
    if (!checkPass) {
        printf("%s - ERROR - [XIR]: x_grad check FAILED\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: x_grad check PASS (matches the README formula)\n", GetTime().c_str());

    delete session;

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
