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

namespace MoeEpDispatchEpilogueUT {

class MoeEpDispatchEpilogueArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeEpDispatchEpilogueArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeEpDispatchEpilogueArch35TilingTest TearDown" << std::endl;
    }
};

// Basic success: recvX + recvSrcMetadata + recvTopkWeights (no recvScales for non-fp8)
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, Basic)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 8}, {1024, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8 * 1024, 3}, {8 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8 * 1024, 7168}, {8 * 1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8 * 1024, 3}, {8 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 1024}, {8 * 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// Small shape
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, SmallShape)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2 * 128, 3}, {2 * 128, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2 * 128, 2048}, {2 * 128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2 * 128, 3}, {2 * 128, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2 * 128}, {2 * 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// No recvTopkWeights output (optional output absent)
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, NoTopkWeights)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{4 * 512, 4096}, {4 * 512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4 * 512, 3}, {4 * 512, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// Invalid epWorldSize (must be >= 2)
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, InvalidEpWorldSize)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{64, 4096}, {64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 3}, {64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid numExperts (not divisible by epWorldSize)
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, InvalidNumExperts)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4 * 64, 3}, {4 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4 * 64, 4096}, {4 * 64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4 * 64, 3}, {4 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 64}, {4 * 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Non-cached mode (4 inputs, no cachedRecvSrcMetadata)
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, NonCached)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{4 * 512, 4096}, {4 * 512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4 * 512, 3}, {4 * 512, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 512}, {4 * 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// fp16 recvX dtype
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, Fp16RecvX)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 4}, {256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{4 * 256, 2048}, {4 * 256, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4 * 256, 3}, {4 * 256, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 256}, {4 * 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

// ExpertAlignment > 1 should fail (must be exactly 1)
TEST_F(MoeEpDispatchEpilogueArch35TilingTest, ExpertAlignmentGT1)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{8 * 512, 4096}, {8 * 512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8 * 512, 3}, {8 * 512, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 512}, {8 * 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace MoeEpDispatchEpilogueUT
