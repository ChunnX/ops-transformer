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
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../op_plugin/scatter_pa_kv_cache_proto.h"

class ScatterPaKvCache : public testing::Test {
protected:
static void SetUpTestCase() {
    std::cout << "ScatterPaKvCache SetUp" << std::endl;
}

static void TearDownTestCase() {
    std::cout << "ScatterPaKvCache TearDown" << std::endl;
}
};

TEST_F(ScatterPaKvCache, template1)
{
    const int num_tokens = 102;
    const int num_head = 1;
    const int k_head_size = 128;
    const int v_head_size = 128;
    const int num_blocks = 306;
    const int block_size = 128;

    ge::op::ScatterPaKvCache op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    std::vector<int64_t> key_shape = {num_tokens, num_head, k_head_size};
    std::vector<int64_t> value_shape = {num_tokens, num_head, v_head_size};
    std::vector<int64_t> key_cache_shape = {num_blocks, block_size, num_head, k_head_size};
    std::vector<int64_t> value_cache_shape = {num_blocks, block_size, num_head, v_head_size};
    std::vector<int64_t> slot_mapping_shape = {num_tokens};

    auto input_key_cache = create_desc_shape_range(key_cache_shape, ge::DT_FLOAT, ge::FORMAT_ND, key_cache_shape, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("key_cache", input_key_cache);

    auto input_value_cache = create_desc_shape_range(value_cache_shape, ge::DT_FLOAT, ge::FORMAT_ND, value_cache_shape, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("value_cache", input_value_cache);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto key_cache_shape_cur = op.GetOutputDesc("key_cache");
    std::vector<int64_t> expected_output_key_cache_shape = key_cache_shape;
    EXPECT_EQ(key_cache_shape_cur.GetShape().GetDims(), expected_output_key_cache_shape);


    auto value_cache_shape_cur = op.GetOutputDesc("value_cache");
    std::vector<int64_t> expected_output_value_cache_shape = value_cache_shape;
    EXPECT_EQ(value_cache_shape_cur.GetShape().GetDims(), expected_output_value_cache_shape);
}