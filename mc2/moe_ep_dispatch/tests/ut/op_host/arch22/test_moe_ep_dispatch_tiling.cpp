/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliancewith the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

using namespace std;

namespace MoeEpDispatchUT {

class MoeEpDispatchArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeEpDispatchArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeEpDispatchArch22TilingTest TearDown" << std::endl;
    }
};

// Basic success: non-cached, doCpuSync=true, direct network (epWorldSize <= 128)
TEST_F(MoeEpDispatchArch22TilingTest, BasicNonCached)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
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
            {{{8 * 2048, 3}, {8 * 2048, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Cached mode: cached_handle_dst_buffer_slot_idx present, topk_idx/topk_weights null
TEST_F(MoeEpDispatchArch22TilingTest, CachedMode)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            // topk_idx: nullptr (optional, not provided)
            // topk_weights: nullptr (optional, not provided)
            {{{1024, 4}, {1024, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1024, 4}, {1024, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 1024, 3}, {4 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// FP16 dtype
TEST_F(MoeEpDispatchArch22TilingTest, Fp16Dtype)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2 * 512, 3}, {2 * 512, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Large epWorldSize (hybrid network mode: >128)
TEST_F(MoeEpDispatchArch22TilingTest, HybridNetworkMode)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    int64_t ep = 256;
    int64_t nmt = 1024;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 4096}, {256, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 8}, {256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 8}, {256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{ep}, {ep}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{256, 8}, {256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{ep * nmt, 3}, {ep * nmt, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(nmt)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: epWorldSize < MIN_EP_WORLD_SIZE (2)
TEST_F(MoeEpDispatchArch22TilingTest, InvalidEpWorldSizeTooSmall)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
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
            {{{1 * 1024, 3}, {1 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: epRankId >= epWorldSize
TEST_F(MoeEpDispatchArch22TilingTest, InvalidEpRankId)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4096}, {64, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 1024, 3}, {4 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: hidden out of range
TEST_F(MoeEpDispatchArch22TilingTest, InvalidHiddenSize)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 512}, {64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4 * 1024, 3}, {4 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Invalid: numExperts not divisible by epWorldSize
TEST_F(MoeEpDispatchArch22TilingTest, InvalidNumExperts)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
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
            {{{8 * 1024, 3}, {8 * 1024, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(63)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 910B platform with larger core count
TEST_F(MoeEpDispatchArch22TilingTest, Ascend910BPlatform)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8 * 8, 7168}, {8 * 8 * 8, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 2048, 3}, {8 * 2048, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Large batch size (max tokens)
TEST_F(MoeEpDispatchArch22TilingTest, LargeNumTokens)
{
    struct MoeEpDispatchCompileInfo {};
    MoeEpDispatchCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeEpDispatch",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8192, 7168}, {8192, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8192, 8}, {8192, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8192, 8}, {8192, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8192, 8}, {8192, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8192, 3}, {8 * 8192, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8192)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(400 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace MoeEpDispatchUT
