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
 * \file test_moe_tutel_dispatch_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTutelDispatchInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelDispatchInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTutelDispatchInfershape TearDown" << std::endl;
    }
};

// y = [gates.dim(0), capacity, x.dim(1)]
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_basic)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x    [S, H]
                                         {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},   // gates     [B, S]
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},   // indices   [B, S]
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},   // locations [B, S]
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y [B, capacity, H]
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// capacity = 0：y 为空张量，仍要推出正确 rank 与各维
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_capacity_zero)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{8, 16}, {8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 8}, {3, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// S = 0：空 token，y 仍由 B / capacity / H 决定
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_empty_samples)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{0, 32}, {0, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{2, 0}, {2, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 5, 32}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// capacity 为负：拒收
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_negative_capacity_failed)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// x 不是 2 维：拒收
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_x_rank_failed)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{2, 8, 16}, {2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// 动态 rank（-2 / UNKNOWN_RANK）：本算子 rank 固定，明确拒收而非静默推错
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_unknown_rank_failed)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x 为 UNKNOWN_RANK
                                         {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// gates 为 UNKNOWN_RANK：同样拒收
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_gates_unknown_rank_failed)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// B = 0：空专家维，y 为空张量
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_empty_expert)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{0, 8}, {0, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 4, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// gates 不是 2 维：拒收
TEST_F(MoeTutelDispatchInfershape, moe_tutel_dispatch_infershape_gates_rank_failed)
{
    gert::InfershapeContextPara para("MoeTutelDispatch",
                                     {
                                         {{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 2, 8}, {2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}
