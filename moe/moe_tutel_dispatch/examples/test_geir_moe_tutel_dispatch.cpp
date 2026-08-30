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
#include <ctime>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "graph/error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "securec.h"
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
constexpr int64_t ROUTE_K = 2;  // K：每个 token 的路由条数，来自 gates.dim(0)
constexpr int64_t EXPERT_E = 5; // E：输出专家数，由 caller 的 y descriptor 独立提供
constexpr int64_t CAPACITY = 4; // C：每专家容量

string GetTime()
{
    const time_t timep = time(nullptr);
    if (timep == static_cast<time_t>(-1)) {
        return "time-unavailable";
    }
    struct tm timeValue {};
    if (localtime_r(&timep, &timeValue) == nullptr) {
        return "time-unavailable";
    }
    char tmp[64] = {0};
    const size_t length = strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", &timeValue);
    if (length == 0) {
        return "time-unavailable";
    }
    return string(tmp, length);
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

bool SafeMulSize(size_t lhs, size_t rhs, size_t &result)
{
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }
    if (lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

int32_t GenTensor(const vector<int64_t> &shape, DataType dtype, int32_t value, Tensor &tensor, TensorDesc &desc)
{
    desc.SetRealDimCnt(shape.size());
    size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim < 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() ||
            !SafeMulSize(count, static_cast<size_t>(dim), count)) {
            printf("%s - ERROR - [XIR]: invalid tensor shape\n", GetTime().c_str());
            return FAILED;
        }
    }
    const uint32_t typeSize = GetDataTypeSize(dtype);
    if (typeSize == 0U) {
        printf("%s - ERROR - [XIR]: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    size_t dataLen = 0;
    if (!SafeMulSize(count, static_cast<size_t>(typeSize), dataLen)) {
        printf("%s - ERROR - [XIR]: tensor byte size overflow\n", GetTime().c_str());
        return FAILED;
    }
    vector<uint8_t> buffer(dataLen, 0U);
    if (typeSize == 4U) {
        const int32_t typedValue = value;
        for (size_t i = 0; i < count; ++i) {
            const size_t offset = i * sizeof(typedValue);
            if (memcpy_s(buffer.data() + offset, dataLen - offset, &typedValue, sizeof(typedValue)) != EOK) {
                return FAILED;
            }
        }
    } else {
        const int16_t typedValue = static_cast<int16_t>(value);
        for (size_t i = 0; i < count; ++i) {
            const size_t offset = i * sizeof(typedValue);
            if (memcpy_s(buffer.data() + offset, dataLen - offset, &typedValue, sizeof(typedValue)) != EOK) {
                return FAILED;
            }
        }
    }
    tensor = Tensor(desc, buffer);
    return SUCCESS;
}

/*
 * 构图：x[S,H] + gates[K,S] + indices[K,S] + locations[K,S] --(capacity)--> y[E,C,H]
 * 本样例故意令 E>K，用于演示 Ascend 950 上 E 只能来自 caller output descriptor。
 * indices / locations 这里都填 1；样例只演示构图和输出 descriptor，不作为重复目的地的数值样例。
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
        {"gates", {ROUTE_K, SAMPLES}, DT_FLOAT, 1},
        {"indices", {ROUTE_K, SAMPLES}, DT_INT32, 1},
        {"locations", {ROUTE_K, SAMPLES}, DT_INT32, 1},
    };

    for (size_t i = 0; i < specs.size(); ++i) {
        const string dataName = string("placeholder_") + specs[i].name;
        auto data = op::Data(dataName.c_str()).set_attr_index(static_cast<int64_t>(i));
        TensorDesc desc(ge::Shape(specs[i].shape), FORMAT_ND, specs[i].dtype);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        if (GenTensor(specs[i].shape, specs[i].dtype, specs[i].value, tensor, desc) != SUCCESS) {
            return FAILED;
        }
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        inputs.push_back(tensor);
        if (graph.AddOp(data) != ge::GRAPH_SUCCESS) {
            return FAILED;
        }
        dataNodes.push_back(data);
    }

    op.set_input_x(dataNodes[0]);
    op.set_input_gates(dataNodes[1]);
    op.set_input_indices(dataNodes[2]);
    op.set_input_locations(dataNodes[3]);

    TensorDesc yDesc(ge::Shape({EXPERT_E, CAPACITY, HIDDEN}), FORMAT_ND, DT_FLOAT);
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
    std::unique_ptr<Session> session(new (std::nothrow) Session(buildOptions));
    if (session == nullptr) {
        (void)ge::GEFinalize();
        return FAILED;
    }

    const uint32_t graphId = 0U;
    const std::map<AscendString, AscendString> graphOptions;
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    vector<Tensor> outputs;
    if (session->RunGraph(graphId, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const int64_t shapeSize = outputs[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %zu shape size = %lld\n", GetTime().c_str(), i,
               static_cast<long long>(shapeSize));
    }

    session.reset();
    if (ge::GEFinalize() != SUCCESS) {
        return FAILED;
    }
    printf("%s - INFO - [XIR]: run test_geir_moe_tutel_dispatch success\n", GetTime().c_str());
    return SUCCESS;
}
