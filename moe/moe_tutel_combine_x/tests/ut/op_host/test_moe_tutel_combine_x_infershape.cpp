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
 * \brief Unit tests of the MoeTutelCombineX graph-mode shape inference.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTutelCombineX : public testing::Test {
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

// x_grad.shape = gates.shape + (y_grad.shape[-1],), no-batch form
TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_shape_no_batch)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// batched form: rank(y_grad)=3, rank(gates)=2
TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_shape_batch)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{2, 4, 2}, {2, 4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown (dynamic) dims are propagated without an error
TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_shape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// batch form with mismatched micro-batch dims (y_grad.dim0 != gates.dim0)
// must be rejected — the shape contract of the operator.
TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_shape_batch_dim_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{3, 4, 2}, {3, 4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// batch form with an unknown batch dim on either side stays accepted (the
// unknown-dim tolerance of the metadata shape check extends to the batch dim)
TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_shape_batch_dim_unknown)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{-1, 4, 2}, {-1, 4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// empty tensors (0-sized dims) are legal degenerate shapes and propagate
TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_shape_empty)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineX",
                                                      {
                                                          {{{0, 2}, {0, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType: the output dtype follows y_grad for every dtype registered in
// the OpDef (FLOAT / FLOAT16 / BFLOAT16), gates must share it and
// indices/locations must be INT32.
static void RunInferDataTypeCase(ge::DataType ygDtype, ge::DataType gatesDtype, ge::DataType indicesDtype,
                                 ge::DataType locationsDtype, ge::graphStatus expectResult,
                                 ge::DataType expectOutputDtype = ge::DT_UNDEFINED)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTutelCombineX");
    ASSERT_NE(opImpl, nullptr);
    auto data_type_func = opImpl->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputRefs[4] = {ygDtype, gatesDtype, indicesDtype, locationsDtype};
    ge::DataType outputRef = expectOutputDtype;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(4, 1)
                              .IrInstanceNum({1, 1, 1, 1})
                              .NodeInputTd(0, ygDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, gatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, locationsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputRefs[0], &inputRefs[1], &inputRefs[2], &inputRefs[3]})
                              .OutputDataTypes({&outputRef})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(data_type_func(context), expectResult);
    if (expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(0), expectOutputDtype);
    }
}

TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_data_type_fp32)
{
    RunInferDataTypeCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::GRAPH_SUCCESS, ge::DT_FLOAT);
}

TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_data_type_fp16)
{
    RunInferDataTypeCase(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::GRAPH_SUCCESS, ge::DT_FLOAT16);
}

TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_data_type_bf16)
{
    RunInferDataTypeCase(ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::GRAPH_SUCCESS, ge::DT_BF16);
}

TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_data_type_y_grad_int32_rejected)
{
    RunInferDataTypeCase(ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_data_type_gates_mismatch_rejected)
{
    RunInferDataTypeCase(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineX, moe_tutel_combine_x_infer_data_type_indices_int64_rejected)
{
    RunInferDataTypeCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT32, ge::GRAPH_FAILED);
}
