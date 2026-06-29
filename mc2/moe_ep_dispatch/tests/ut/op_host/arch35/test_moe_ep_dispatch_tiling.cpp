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

namespace MoeEpDispatchUT {

class MoeEpDispatchArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeEpDispatchArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeEpDispatchArch35TilingTest TearDown" << std::endl;
    }
};

// Basic success: non-cached, doCpuSync=true, direct network
TEST_F(MoeEpDispatchArch35TilingTest, BasicNonCached)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
          {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
          {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
          {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
          {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
          {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
          {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
          {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 1);
}

// Cached mode
TEST_F(MoeEpDispatchArch35TilingTest, CachedMode)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 3072}, {512, 3072}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Hybrid network mode (epWorldSize > 128)
TEST_F(MoeEpDispatchArch35TilingTest, HybridNetworkMode)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    int64_t ep = 144;
    int64_t nmt = 512;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{ep}, {ep}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(72)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(144)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(nmt)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 17);
}

// Invalid: epWorldSize too small
TEST_F(MoeEpDispatchArch35TilingTest, InvalidEpWorldSize)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4096}, {64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: numExperts not divisible by epWorldSize
TEST_F(MoeEpDispatchArch35TilingTest, InvalidNumExperts)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4096}, {64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(63)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Small hidden valid
TEST_F(MoeEpDispatchArch35TilingTest, SmallHidden)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 1024}, {256, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 4}, {256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 4}, {256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{256, 4}, {256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 1);
}

// Non-cached with doCpuSync=false (no host-pinned counter write)
TEST_F(MoeEpDispatchArch35TilingTest, NonCachedNoCpuSync)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 1);
}

// ExpertAlignment > 1
TEST_F(MoeEpDispatchArch35TilingTest, ExpertAlignmentGT1)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"hp_Counter_AddrPtr", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 1);
}

} // namespace MoeEpDispatchUT
