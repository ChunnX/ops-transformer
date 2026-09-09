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

#include "../../../framework/npu_moe_finalize_routing_v2_onnx_plugin.cpp"

TEST(OnnxMoeFinalizeRoutingV2PluginTest, MissingDropPadModeDefaultsToZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, CopiesDropPadMode)
{
    for (int64_t mode : {0, 1, 2}) {
        SCOPED_TRACE(mode);
        ge::Operator src("src", "TestOp");
        ge::Operator dest("dest", "TestOp");
        src.SetAttr("drop_pad_mode", mode);

        ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
        int64_t value = -1;
        ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
        EXPECT_EQ(value, mode);
    }
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, WrongAttributeTypeDefaultsToZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("drop_pad_mode", 1.5f);
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}
