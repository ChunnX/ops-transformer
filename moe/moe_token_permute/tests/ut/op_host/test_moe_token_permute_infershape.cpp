/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenPermute : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermute SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermute TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermute, moe_token_permute_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermute",
                                                      {
                                                        {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{256, 4}, {256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 7168}, {1024}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermute, MoeTokenPermute_infershape_scale)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermute",
                                                      {
                                                        {{{36, 385}, {36, 385}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{36, 3}, {36, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{108, 385}, {108}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermute, MoeTokenPermute_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermute",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, -1}, {1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermute, MoeTokenPermute_infershape_unknownshape)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermute",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermute, MoeTokenPermute_inferdtype)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenPermute"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenPermute")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT32;
    ge::DataType output_0 = ge::DT_UNDEFINED;
    ge::DataType output_1 = ge::DT_UNDEFINED;
    ge::DataType output_2 = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 3)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({
                                  {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                  {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                              })
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_1})
                              .OutputDataTypes({&output_0, &output_1, &output_2})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

TEST_F(MoeTokenPermute, MoeTokenPermute_infershape_mxfp8)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermute",
                                                      {
                                                        {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 4}, {8, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{20, 128}, {32}, {20, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeTokenPermute, MoeTokenPermute_infershape_mxfp4)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermute",
                                                      {
                                                        {{{8, 128}, {8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                        {{{8, 4}, {8, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"numOutTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"paddedMode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{20, 128}, {32}, {20, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
