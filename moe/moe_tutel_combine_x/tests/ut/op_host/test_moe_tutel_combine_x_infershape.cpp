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
 * \file test_moe_tutel_combine_x_infershape.cpp
 * \brief InferShape/InferDataType UT。x_grad = [gates.dim(1), y_grad.dim(2)] = [S, H]，
 *        x_grad dtype 与 y_grad 相同。
 */

#include <gtest/gtest.h>
#include <iostream>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

class MoeTutelCombineXProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelCombineXProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTutelCombineXProto TearDown" << std::endl;
    }
};

// 主路径：y_grad [E=4, C=8, H=64]、路由 [K=2, S=16] -> x_grad [16, 64]。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_float32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineX",
        {
            {{{4, 8, 64}, {4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y_grad
            {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},       // gates
            {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},       // indices
            {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},       // locations
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x_grad
        },
        {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{16, 64}});
}

// bf16 与 fp16 走同一条 shape 推导。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_bfloat16)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{8, 32, 1024}, {8, 32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{4, 256}, {4, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{4, 256}, {4, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4, 256}, {4, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{256, 1024}});
}

// K=0：gates 第 0 维为 0，输出仍是 [S, H]，不报错。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_k_zero)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{4, 8, 64}, {4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0, 16}, {0, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0, 16}, {0, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{0, 16}, {0, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{16, 64}});
}

// S=0 空 tensor：输出 [0, H]，不报错。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_empty_samples)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{4, 8, 64}, {4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{0, 64}});
}

// E/C/K/S/H 同时为 0：InferShape 仍按 [S,H] 生成 [0,0]。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_all_zero_axes)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0, 0}, {0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0, 0}, {0, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{0, 0}, {0, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{0, 0}});
}

// 固定 rank 的动态维（-1）允许在 InferShape 阶段透传；运行时 Tiling 仍要求实际维度非负。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_dynamic_dims)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{4, 8, -1}, {4, 8, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{-1, -1}});
}

// y_grad rank 不是 3：拒绝。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_invalid_y_grad_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{4, 64}, {4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// gates rank 不是 2：拒绝。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_invalid_gates_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{4, 8, 64}, {4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 16, 1}, {2, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 16, 1}, {2, 16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 16, 1}, {2, 16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// UNKNOWN_RANK(-2)：H 与 S 取不到具体轴，明确拒绝。
TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{-2}, {4, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineXProto, moe_tutel_combine_x_inferdatatype_follows_y_grad)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto *opImpl = spaceRegistry->GetOpImpl("MoeTutelCombineX");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataType = opImpl->infer_datatype;
    ASSERT_NE(inferDataType, nullptr);

    for (const ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        ge::DataType inputDtype = dtype;
        ge::DataType indicesDtype = ge::DT_INT32;
        ge::DataType outputDtype = ge::DT_UNDEFINED;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .SetOpType("MoeTutelCombineX")
                                 .NodeIoNum(4, 1)
                                 .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(3, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&inputDtype, &inputDtype, &indicesDtype, &indicesDtype})
                                 .OutputDataTypes({&outputDtype})
                                 .Build();
        auto *context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(inferDataType(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), inputDtype);
    }
}
