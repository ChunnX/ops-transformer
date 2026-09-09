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
