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
 * \brief InferShape UT。
 *
 * 说明：InferDataType 实现落在 op_graph/moe_tutel_combine_gates_graph_infer.cpp，
 * 该目录只在 BUILD_OPEN_PROJECT 下随 graph plugin 编译，不参与 op_host UT 构建树，
 * 因此这里不写 InferDataType 用例（该部分做本地验证，不上库 UT）。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

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

// gates_grad 与 indices 同形：[E, S]。
TEST_F(MoeTutelCombineGatesProto, shape_infer_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineGates",
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // x     [S, H]
            {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y_grad[E, C, H]
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},         // indices  [E, S]
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},         // locations[E, S]
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// S == 0 的空 Tensor：输出同形 [E, 0]，空进空出。
TEST_F(MoeTutelCombineGatesProto, shape_infer_empty_samples)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineGates",
        {
            {{{0, 64}, {0, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 64}, {2, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 rank：indices 为 -2（UNKNOWN_RANK）时输出也必须是 UNKNOWN_RANK，
// 不能把 -2 当成真实维度推下去。def.cpp 声明了 DynamicRankSupportFlag(true)，此路必须真验。
TEST_F(MoeTutelCombineGatesProto, shape_infer_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineGates",
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

// E == 0 的空 Tensor：输出 [0, S]。
TEST_F(MoeTutelCombineGatesProto, shape_infer_empty_experts)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineGates",
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 4, 64}, {0, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// H == 0 的空内积：shape 推导不受影响。
TEST_F(MoeTutelCombineGatesProto, shape_infer_zero_hidden)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTutelCombineGates",
        {
            {{{8, 0}, {8, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 5, 0}, {3, 5, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
