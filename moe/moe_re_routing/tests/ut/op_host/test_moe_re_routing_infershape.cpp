/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MoeReRouting : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeReRoutingProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeReRoutingProto TearDown" << std::endl;
    }
};

TEST_F(MoeReRouting, moe_re_routing_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRouting",
                                                      {
                                                        {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 7168}, {256}, {256}, {16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRouting, MoeReRouting_infershape_scale)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRouting",
                                                      {
                                                        {{{36, 385}, {36, 385}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{3, 12}, {3, 12}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{36, 3}, {36, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{36, 385}, {36, 3}, {36}, {12}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeReRouting, MoeReRouting_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRouting",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}


TEST_F(MoeReRouting, MoeReRouting_infershape_unkownshape)
{
    gert::InfershapeContextPara infershapeContextPara("MoeReRouting",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}, {-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
