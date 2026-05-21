/**
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infershape_test_util.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../op_graph/swiglu_gated_mlp_proto.h"

class SwigluGatedMlpInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluGatedMlpInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluGatedMlpInferShape TearDown" << std::endl;
    }
};

TEST_F(SwigluGatedMlpInferShape, full_mlp_infershape)
{
    ge::op::SwigluGatedMlp op;
    op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gate_up_weight", create_desc({3, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("down_weight", create_desc({4, 2}, ge::DT_FLOAT16));
    op.SetAttr("cubeMathType", static_cast<int64_t>(1));

    Runtime2TestParam param{{"cubeMathType"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto yDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {2, 2};
    EXPECT_EQ(yDesc.GetShape().GetDims(), expectedShape);

    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    yDesc = op.GetOutputDesc(0);
    EXPECT_EQ(yDesc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(SwigluGatedMlpInferShape, swiglu_stage_infershape)
{
    ge::op::SwigluGatedMlp op;
    op.UpdateInputDesc("x", create_desc({2, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("gate_up_weight", create_desc({3, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("down_weight", create_desc({4, 2}, ge::DT_FLOAT));
    op.SetAttr("cubeMathType", static_cast<int64_t>(102));

    Runtime2TestParam param{{"cubeMathType"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto yDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {2, 4};
    EXPECT_EQ(yDesc.GetShape().GetDims(), expectedShape);

    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    yDesc = op.GetOutputDesc(0);
    EXPECT_EQ(yDesc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(SwigluGatedMlpInferShape, illegal_gate_up_shape)
{
    ge::op::SwigluGatedMlp op;
    op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gate_up_weight", create_desc({3, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("down_weight", create_desc({4, 2}, ge::DT_FLOAT16));
    op.SetAttr("cubeMathType", static_cast<int64_t>(1));

    Runtime2TestParam param{{"cubeMathType"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}
