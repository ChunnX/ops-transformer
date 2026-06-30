/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ctime>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/mhc_pre_sinkhorn_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using ge::AscendString;
using ge::DataType;
using ge::FORMAT_ND;
using ge::Graph;
using ge::Operator;
using ge::Session;
using ge::Status;
using ge::Tensor;
using ge::TensorDesc;

std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return sizeof(float);
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return sizeof(uint16_t);
    }
    if (dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) {
        return sizeof(uint32_t);
    }
    if (dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
        return sizeof(uint64_t);
    }
    return sizeof(uint8_t);
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int GenTensorData(const std::vector<int64_t> &shape, Tensor &tensor, TensorDesc &desc, DataType dtype)
{
    desc.SetRealDimCnt(shape.size());
    const int64_t elem_num = GetShapeSize(shape);
    const uint32_t dtype_size = GetDataTypeSize(dtype);
    const uint32_t byte_len = static_cast<uint32_t>(elem_num * dtype_size);
    uint8_t *data = new (std::nothrow) uint8_t[byte_len];
    if (data == nullptr) {
        return FAILED;
    }

    if (dtype == ge::DT_FLOAT) {
        auto *float_data = reinterpret_cast<float *>(data);
        for (int64_t i = 0; i < elem_num; ++i) {
            float_data[i] = 0.01f * static_cast<float>((i % 17) - 8);
        }
    } else {
        auto *half_data = reinterpret_cast<uint16_t *>(data);
        for (int64_t i = 0; i < elem_num; ++i) {
            half_data[i] = static_cast<uint16_t>(0x3f80 + (i % 7));
        }
    }

    tensor = Tensor(desc, data, byte_len);
    return SUCCESS;
}

int AddDataInput(const std::string &name, uint32_t index, DataType dtype, const std::vector<int64_t> &shape,
                 Graph &graph, std::vector<Tensor> &input_tensors, std::vector<Operator> &input_ops,
                 ge::op::MhcPreSinkhorn &op)
{
    auto data = ge::op::Data(name).set_attr_index(index);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);

    Tensor tensor;
    if (GenTensorData(shape, tensor, desc, dtype) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input %s failed\n", GetTime().c_str(), name.c_str());
        return FAILED;
    }

    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    input_tensors.push_back(tensor);
    graph.AddOp(data);
    input_ops.push_back(data);

    if (name == "x") {
        op.set_input_x(data);
    } else if (name == "phi") {
        op.set_input_phi(data);
    } else if (name == "alpha") {
        op.set_input_alpha(data);
    } else if (name == "bias") {
        op.set_input_bias(data);
    }
    return SUCCESS;
}

void SetOutputDesc(ge::op::MhcPreSinkhorn &op)
{
    constexpr int64_t batch = 1;
    constexpr int64_t seqLen = 4;
    constexpr int64_t hcMult = 4;
    constexpr int64_t hiddenDim = 256;
    constexpr int64_t hcMix = hcMult * hcMult + 2 * hcMult;
    constexpr int64_t numIters = 20;

    op.update_output_desc_hin(TensorDesc(ge::Shape({batch, seqLen, hiddenDim}), FORMAT_ND, ge::DT_BF16));
    op.update_output_desc_hPost(TensorDesc(ge::Shape({batch, seqLen, hcMult}), FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_hRes(TensorDesc(ge::Shape({batch, seqLen, hcMult * hcMult}), FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_hPre(TensorDesc(ge::Shape({batch, seqLen, hcMult}), FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_hcBeforeNorm(TensorDesc(ge::Shape({batch, seqLen, hcMix}), FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_invRms(TensorDesc(ge::Shape({batch, seqLen, 1}), FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_sumOut(TensorDesc(ge::Shape({numIters * 2, batch, seqLen, hcMult}), FORMAT_ND,
                                           ge::DT_FLOAT));
    op.update_output_desc_normOut(TensorDesc(ge::Shape({numIters * 2, batch, seqLen, hcMult, hcMult}), FORMAT_ND,
                                            ge::DT_FLOAT));
}

int CreateGraph(std::vector<Tensor> &input_tensors, std::vector<Operator> &input_ops, std::vector<Operator> &outputs,
                Graph &graph)
{
    auto mhc_pre_sinkhorn = ge::op::MhcPreSinkhorn("test_geir_mhc_pre_sinkhorn");

    if (AddDataInput("x", 0, ge::DT_BF16, {1, 4, 4, 256}, graph, input_tensors, input_ops,
                     mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("phi", 1, ge::DT_FLOAT, {24, 1024}, graph, input_tensors, input_ops,
                     mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("alpha", 2, ge::DT_FLOAT, {3}, graph, input_tensors, input_ops, mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("bias", 3, ge::DT_FLOAT, {24}, graph, input_tensors, input_ops, mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }

    mhc_pre_sinkhorn.set_attr_hc_mult(4);
    mhc_pre_sinkhorn.set_attr_num_iters(20);
    mhc_pre_sinkhorn.set_attr_hc_eps(1e-6f);
    mhc_pre_sinkhorn.set_attr_norm_eps(1e-6f);
    mhc_pre_sinkhorn.set_attr_need_backward(true);
    SetOutputDesc(mhc_pre_sinkhorn);

    outputs.push_back(mhc_pre_sinkhorn);
    return SUCCESS;
}

int main()
{
    Graph graph("mhc_pre_sinkhorn_geir_graph");
    std::vector<Tensor> input_tensors;
    std::vector<Operator> input_ops;
    std::vector<Operator> output_ops;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    ret = CreateGraph(input_tensors, input_ops, output_ops, graph);
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }
    graph.SetInputs(input_ops).SetOutputs(output_ops);

    std::map<AscendString, AscendString> build_options = {};
    auto session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        ge::GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    constexpr uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    std::vector<Tensor> output_tensors;
    ret = session->RunGraph(graph_id, input_tensors, output_tensors);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
        std::cout << "Error message: " << error_msg.GetString() << std::endl;
        std::cout << "Warning message: " << warning_msg.GetString() << std::endl;
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Run graph success, output num: %zu\n", GetTime().c_str(), output_tensors.size());
    if (output_tensors.size() != 8) {
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
