/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

using namespace std;

namespace MoeEpCombineUT {

class MoeEpCombineArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeEpCombineArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeEpCombineArch35TilingTest TearDown" << std::endl;
    }
};

// Basic success: without topk_weights
TEST_F(MoeEpCombineArch35TilingTest, BasicNoTopkWeights)
{
    struct MoeEpCombineCompileInfo {};
    MoeEpCombineCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpCombine",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 1024, 7168}, {8 * 1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 8}, {1024, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 1024, 3}, {8 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1024, 7168}, {1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// Basic success: with topk_weights
TEST_F(MoeEpCombineArch35TilingTest, BasicWithTopkWeights)
{
    struct MoeEpCombineCompileInfo {};
    MoeEpCombineCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpCombine",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 1024, 7168}, {8 * 1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 8}, {1024, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 1024, 3}, {8 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 8}, {1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 7168}, {1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 8}, {1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 1);
}

// Small shape with topk_weights
TEST_F(MoeEpCombineArch35TilingTest, SmallWithTopkWeights)
{
    struct MoeEpCombineCompileInfo {};
    MoeEpCombineCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpCombine",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2 * 128, 2048}, {2 * 128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2 * 128, 3}, {2 * 128, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4}, {128, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 4}, {128, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 1);
}

// Invalid: epWorldSize too small
TEST_F(MoeEpCombineArch35TilingTest, InvalidEpWorldSize)
{
    struct MoeEpCombineCompileInfo {};
    MoeEpCombineCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpCombine",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1 * 64, 4096}, {1 * 64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1 * 64, 3}, {1 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 4096}, {64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: hidden out of range
TEST_F(MoeEpCombineArch35TilingTest, InvalidHiddenSize)
{
    struct MoeEpCombineCompileInfo {};
    MoeEpCombineCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpCombine",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 64, 512}, {8 * 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 64, 3}, {8 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 512}, {64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace MoeEpCombineUT
