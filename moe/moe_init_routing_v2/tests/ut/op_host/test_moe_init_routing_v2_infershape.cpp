/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeInitRoutingV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV2 TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_03)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_04)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_05)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_06)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_07)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_08)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 5}, {2, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_09)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_10)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_11)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(99)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_12)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_13)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_datatype_01)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType index_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 4)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &index_ref})
                                  .OutputDataTypes({&output_ref, &index_ref, &index_ref, &index_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), index_ref);
        EXPECT_EQ(context->GetOutputDataType(2), index_ref);
        EXPECT_EQ(context->GetOutputDataType(3), index_ref);
    }
}

// expertTokensCountOrCumsumFlag=2 (cumsum模式)
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_cumsum_flag_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// FLOAT32数据类型
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_float32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// FLOAT32 datatype推导
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_datatype_float32)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType index_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 4)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &index_ref})
                                  .OutputDataTypes({&output_ref, &index_ref, &index_ref, &index_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), index_ref);
        EXPECT_EQ(context->GetOutputDataType(2), index_ref);
        EXPECT_EQ(context->GetOutputDataType(3), index_ref);
    }
}

// BFLOAT16数据类型shape推导
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_bfloat16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Active场景 (activeNum > 0)
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_active)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Drop+expertTokensBeforeCapacityFlag=true
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_drop_with_capacity_flag)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BFLOAT16 datatype推导
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_datatype_bfloat16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        ge::DataType index_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 4)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &index_ref})
                                  .OutputDataTypes({&output_ref, &index_ref, &index_ref, &index_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), index_ref);
        EXPECT_EQ(context->GetOutputDataType(2), index_ref);
        EXPECT_EQ(context->GetOutputDataType(3), index_ref);
    }
}

// INT64 expertIdx测试
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_int64_idx)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 大规模数据shape推导测试
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_large_scale)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{1000, 256}, {1000, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1000, 8}, {1000, 8}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(100)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8000, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Drop模式大规模数据测试
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_drop_large)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{500, 128}, {500, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{500, 8}, {500, 8}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 200, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Active场景大规模数据测试
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_active_large)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{1000, 256}, {1000, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1000, 8}, {1000, 8}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(100)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5000, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
