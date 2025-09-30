// /**
//  * This program is free software, you can redistribute it and/or modify.
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /* !
//  * \file test_moe_init_routing_v3_infershape.cpp
//  * \brief
//  */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MoeInitRoutingV3 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3 TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2, 64}, {2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 30})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
