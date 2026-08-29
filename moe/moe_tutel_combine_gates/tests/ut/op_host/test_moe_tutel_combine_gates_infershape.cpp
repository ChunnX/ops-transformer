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
 * \file test_moe_tutel_combine_gates_infershape.cpp
 * \brief InferShape and InferDataType UT.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTutelCombineGatesProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelCombineGatesProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTutelCombineGatesProto TearDown" << std::endl;
    }
};

// K is independent of E: this is a Top-1 route tensor with E == 2.
TEST_F(MoeTutelCombineGatesProto, shape_infer_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineGates",
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // x     [S, H]
            {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y_grad[E, C, H]
            {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},         // indices  [K, S]
            {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},         // locations[K, S]
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// S == 0 propagates as [K, 0].
TEST_F(MoeTutelCombineGatesProto, shape_infer_empty_samples)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineGates",
                                                      {
                                                          {{{0, 64}, {0, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1, 0}, {1, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1, 0}, {1, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Unknown rank propagates without interpreting -2 as a physical dimension.
TEST_F(MoeTutelCombineGatesProto, shape_infer_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineGates",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Unknown dimensions propagate; concrete contract validation belongs to tiling.
TEST_F(MoeTutelCombineGatesProto, shape_infer_unknown_dimension)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineGates",
                                                      {
                                                          {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, 8}, {-1, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1, 8}, {-1, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// H == 0 does not affect route-shape inference; K may still be less than E.
TEST_F(MoeTutelCombineGatesProto, shape_infer_zero_hidden)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTutelCombineGates",
                                                      {
                                                          {{{8, 0}, {8, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{3, 5, 0}, {3, 5, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType follows x and is required by GE graph mode.
static void CheckInferDataType(ge::DataType floatType)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("MoeTutelCombineGates")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType intType = ge::DT_INT32;
    ge::DataType outType = ge::DT_UNDEFINED;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(4)
                             .NodeIoNum(4, 1)
                             .NodeInputTd(0, floatType, ge::FORMAT_ND, ge::FORMAT_ND)    // x
                             .NodeInputTd(1, floatType, ge::FORMAT_ND, ge::FORMAT_ND)    // y_grad
                             .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND) // indices
                             .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND) // locations
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&floatType, &floatType, &intType, &intType})
                             .OutputDataTypes({&outType})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), floatType);
}

TEST_F(MoeTutelCombineGatesProto, infer_datatype_float32)
{
    CheckInferDataType(ge::DT_FLOAT);
}

TEST_F(MoeTutelCombineGatesProto, infer_datatype_float16)
{
    CheckInferDataType(ge::DT_FLOAT16);
}

TEST_F(MoeTutelCombineGatesProto, infer_datatype_bfloat16)
{
    CheckInferDataType(ge::DT_BF16);
}

TEST_F(MoeTutelCombineGatesProto, infer_shape_null_context_fails)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto shapeFunc = spaceRegistry->GetOpImpl("MoeTutelCombineGates")->infer_shape;
    ASSERT_NE(shapeFunc, nullptr);
    EXPECT_EQ(shapeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(MoeTutelCombineGatesProto, infer_datatype_null_context_fails)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("MoeTutelCombineGates")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    EXPECT_EQ(dataTypeFunc(nullptr), ge::GRAPH_FAILED);
}
