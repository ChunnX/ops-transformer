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
 * \file test_geir_moe_tutel_dispatch.cpp
 * \brief MoeTutelDispatch 图模式（GE IR）调用样例
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"

// MoeTutelDispatch 的 REG_OP 由 transformer 仓公共 proto 头持有（本算子不另建 op_graph/*_proto.h）
#include "../../../common/include/op_graph/op_transformer_proto_extend.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr int64_t SAMPLES = 8;  // S：token 数
constexpr int64_t HIDDEN = 16;  // H：隐藏维
constexpr int64_t EXPERTS = 2;  // B：路由条数 / 专家维，等于 gates.dim(0)
constexpr int64_t CAPACITY = 4; // C：每专家容量

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == DT_FLOAT || dt == DT_INT32) {
        return 4U;
    }
    if (dt == DT_FLOAT16 || dt == DT_BF16) {
        return 2U;
    }
    return 0U;
}

int32_t GenTensor(const vector<int64_t> &shape, DataType dtype, Tensor &tensor, TensorDesc &desc, int32_t value)
{
    desc.SetRealDimCnt(shape.size());
    size_t count = 1;
    for (auto d : shape) {
        count *= static_cast<size_t>(d);
    }
    const uint32_t typeSize = GetDataTypeSize(dtype);
    if (typeSize == 0U) {
        printf("%s - ERROR - [XIR]: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    const size_t dataLen = count * typeSize;
    auto *buffer = new (std::nothrow) uint8_t[dataLen];
    if (buffer == nullptr) {
        return FAILED;
    }
    if (typeSize == 4U) {
        auto *p = reinterpret_cast<int32_t *>(buffer);
        for (size_t i = 0; i < count; ++i) {
            p[i] = value;
        }
    } else {
        auto *p = reinterpret_cast<int16_t *>(buffer);
        for (size_t i = 0; i < count; ++i) {
            p[i] = static_cast<int16_t>(value);
        }
    }
    tensor = Tensor(desc, buffer, dataLen);
    delete[] buffer;
    return SUCCESS;
}

/*
 * 构图：x[S,H] + gates[B,S] + indices[B,S] + locations[B,S] --(capacity)--> y[B,C,H]
 * indices / locations 这里都填 1，即所有路由都写到 y[1][1][:]；真实场景由 topk 路由产生。
 */
int32_t BuildGraph(Graph &graph, vector<Tensor> &inputs)
{
    auto op = op::MoeTutelDispatch("moe_tutel_dispatch");
    op.set_attr_capacity(CAPACITY);

    vector<Operator> dataNodes;
    struct InputSpec {
        const char *name = nullptr;
        vector<int64_t> shape{};
        DataType dtype = DT_FLOAT;
        int32_t value = 0;
    };
    const vector<InputSpec> specs = {
        {"x", {SAMPLES, HIDDEN}, DT_FLOAT, 1},
        {"gates", {EXPERTS, SAMPLES}, DT_FLOAT, 1},
        {"indices", {EXPERTS, SAMPLES}, DT_INT32, 1},
        {"locations", {EXPERTS, SAMPLES}, DT_INT32, 1},
    };

    for (size_t i = 0; i < specs.size(); ++i) {
        auto data = op::Data(string("placeholder_") + specs[i].name).set_attr_index(static_cast<int64_t>(i));
        TensorDesc desc(ge::Shape(specs[i].shape), FORMAT_ND, specs[i].dtype);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        if (GenTensor(specs[i].shape, specs[i].dtype, tensor, desc, specs[i].value) != SUCCESS) {
            return FAILED;
        }
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        inputs.push_back(tensor);
        graph.AddOp(data);
        dataNodes.push_back(data);
    }

    op.set_input_x(dataNodes[0]);
    op.set_input_gates(dataNodes[1]);
    op.set_input_indices(dataNodes[2]);
    op.set_input_locations(dataNodes[3]);

    TensorDesc yDesc(ge::Shape({EXPERTS, CAPACITY, HIDDEN}), FORMAT_ND, DT_FLOAT);
    op.update_output_desc_y(yDesc);

    vector<Operator> outputs{op};
    graph.SetInputs(dataNodes).SetOutputs(outputs);
    return SUCCESS;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    Graph graph("moe_tutel_dispatch_geir");
    vector<Tensor> inputs;
    if (BuildGraph(graph, inputs) != SUCCESS) {
        (void)ge::GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> buildOptions;
    auto *session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        (void)ge::GEFinalize();
        return FAILED;
    }

    const uint32_t graphId = 0U;
    if (session->AddGraph(graphId, graph, {}) != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }

    vector<Tensor> outputs;
    if (session->RunGraph(graphId, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const int64_t shapeSize = outputs[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %zu shape size = %ld\n", GetTime().c_str(), i, shapeSize);
    }

    delete session;
    if (ge::GEFinalize() != SUCCESS) {
        return FAILED;
    }
    printf("%s - INFO - [XIR]: run test_geir_moe_tutel_dispatch success\n", GetTime().c_str());
    return SUCCESS;
}
