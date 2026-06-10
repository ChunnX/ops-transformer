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

class MoeEpDispatchEpilogueArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeEpDispatchEpilogueArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeEpDispatchEpilogueArch22TilingTest TearDown" << std::endl;
    }
};

// Basic success: typical parameters
TEST_F(MoeEpDispatchEpilogueArch22TilingTest, Basic)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 8}, {2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 2048, 3}, {8 * 2048, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{8 * 2048, 7168}, {8 * 2048, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8 * 2048, 8}, {8 * 2048, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 2048, 8}, {8 * 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 2048, 3}, {8 * 2048, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Small shape
TEST_F(MoeEpDispatchEpilogueArch22TilingTest, SmallShape)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 4}, {256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2 * 256, 3}, {2 * 256, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{2 * 256, 2048}, {2 * 256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2 * 256, 4}, {2 * 256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2 * 256, 4}, {2 * 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2 * 256, 3}, {2 * 256, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: epWorldSize too small
TEST_F(MoeEpDispatchEpilogueArch22TilingTest, InvalidEpWorldSize)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1 * 64, 3}, {1 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1 * 64, 4096}, {1 * 64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1 * 64, 4}, {1 * 64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1 * 64, 4}, {1 * 64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1 * 64, 3}, {1 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: epRankId out of range
TEST_F(MoeEpDispatchEpilogueArch22TilingTest, InvalidEpRankId)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 64, 3}, {4 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{4 * 64, 4096}, {4 * 64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4 * 64, 4}, {4 * 64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 64, 4}, {4 * 64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4 * 64, 3}, {4 * 64, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 910B platform
TEST_F(MoeEpDispatchEpilogueArch22TilingTest, Ascend910BPlatform)
{
    struct MoeEpDispatchEpilogueCompileInfo {};
    MoeEpDispatchEpilogueCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatchEpilogue",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 512, 3}, {8 * 512, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{8 * 512, 7168}, {8 * 512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8 * 512, 8}, {8 * 512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 512, 8}, {8 * 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8 * 512, 3}, {8 * 512, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace MoeEpDispatchEpilogueUT
