/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "../op_graph/lrn_proto.h"
#include "tensor.h"
#include "types.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

uint32_t GetDTypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1U;
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2U;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4U;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8U;
        default:
            return 0U;
    }
}

int WriteDataToFile(const std::string& file, uint64_t size, const uint8_t* data)
{
    std::ofstream ofs(file, std::ios::binary);
    if (!ofs.is_open()) {
        return kFailed;
    }
    ofs.write(reinterpret_cast<const char*>(data), size);
    return kSuccess;
}

void ProcessOutputData(const std::vector<ge::Tensor>& output)
{
    for (size_t i = 0; i < output.size(); i++) {
        std::string outputFile = "./tc_ge_irrun_test_lrn_output_" + std::to_string(i) + ".bin";
        const uint8_t* data = output[i].GetData();
        int64_t shapeSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t typeSize = GetDTypeSize(output[i].GetTensorDesc().GetDataType());
        if (typeSize == 0U) {
            std::cerr << "ERROR: output " << i << " has unsupported dtype" << std::endl;
            continue;
        }
        uint64_t dataSize = static_cast<uint64_t>(shapeSize) * typeSize;
        WriteDataToFile(outputFile, dataSize, data);
    }
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<ge::Operator>& graphInputs,
                     std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    // FLOAT32 测试: x shape [1, 4, 2, 3]，depth_radius=2 使通道窗口在
    // c=0 和 c=3 两侧均发生边界裁剪（win(0)=[0,2], win(3)=[1,3]）。
    const ge::DataType dtype = ge::DT_FLOAT;
    const std::vector<int64_t> inputShape = {1, 4, 2, 3};
    const int64_t elemCount = 24;

    // 生成输入数据: 1.0, 2.0, ..., 24.0（按 NCHW 排布，每通道 6 个元素）
    // 期望输出抽检 (n=0, c=0, h=0, w=0):
    //   norm = bias + alpha * (1^2 + 7^2 + 13^2) = 220
    //   y = 1 / 220^0.5 ≈ 0.067420
    uint8_t* buf = new (std::nothrow) uint8_t[elemCount * sizeof(float)];
    if (buf == nullptr) {
        return kFailed;
    }
    float* p = reinterpret_cast<float*>(buf);
    for (int64_t i = 0; i < elemCount; i++) {
        p[i] = static_cast<float>(i + 1);
    }
    ge::TensorDesc inputDesc(ge::Shape(inputShape), ge::FORMAT_NCHW, dtype);
    inputDesc.SetPlacement(ge::kPlacementHost);
    inputDesc.SetRealDimCnt(inputShape.size());
    ge::Tensor inputTensor(inputDesc, buf, elemCount * sizeof(float));
    delete[] buf;

    auto data = ge::op::Data("x").set_attr_index(0);
    data.update_input_desc_x(inputDesc);
    data.update_output_desc_y(inputDesc);
    graph.AddOp(data);

    auto lrn = ge::op::LRN("lrn");
    lrn.set_input_x(data);
    lrn.set_attr_depth_radius(2);
    lrn.set_attr_bias(1.0f);
    lrn.set_attr_alpha(1.0f);
    lrn.set_attr_beta(0.5f);
    lrn.set_attr_norm_region("ACROSS_CHANNELS");

    ge::TensorDesc yDesc(ge::Shape(inputShape), ge::FORMAT_NCHW, dtype);
    lrn.update_output_desc_y(yDesc);

    input.push_back(inputTensor);
    graphInputs.push_back(data);
    graphOutputs.push_back(lrn);
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start LRN GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("lrn_graph");
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Operator> graphOutputs;
    if (CreateOppInGraph(inputs, graphInputs, graphOutputs, graph) != kSuccess) {
        std::cerr << "CreateGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    // session 选项：ge.jit_compile=0 二进制复用（直接加载算子包内预编译 kernel）。
    std::map<ge::AscendString, ge::AscendString> sessionOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.jit_compile", "0"},
    };
    ge::Session* session = new (std::nothrow) ge::Session(sessionOptions);
    if (session == nullptr) {
        std::cerr << "Create session failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    if (session->AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed" << std::endl;
        delete session;
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session->RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed" << std::endl;
        delete session;
        ge::GEFinalize();
        return kFailed;
    }

    ProcessOutputData(outputs);

    delete session;
    if (ge::GEFinalize() != ge::SUCCESS) {
        std::cerr << "GEFinalize failed" << std::endl;
        return kFailed;
    }
    std::cout << GetTime() << " - INFO - LRN GEIR example success" << std::endl;
    return kSuccess;
}
