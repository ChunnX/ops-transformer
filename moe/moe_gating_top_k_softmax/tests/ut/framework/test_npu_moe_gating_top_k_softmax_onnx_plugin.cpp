/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "../../../framework/npu_moe_gating_top_k_softmax_onnx_plugin.cpp"

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, CopiesK)
{
    for (int64_t k : {0, 1, 8}) {
        SCOPED_TRACE(k);
        ge::Operator src("src", "TestOp");
        ge::Operator dest("dest", "TestOp");
        src.SetAttr("k", k);

        ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::SUCCESS);
        int64_t value = -1;
        ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
        EXPECT_EQ(value, k);
    }
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, MissingKFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("k", int64_t{9});

    EXPECT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, WrongAttributeTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("k", 2.5f);
    dest.SetAttr("k", int64_t{9});

    EXPECT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, NegativeOneFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("k", int64_t{-1});
    dest.SetAttr("k", int64_t{9});

    EXPECT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}
