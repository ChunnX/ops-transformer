/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../../../op_kernel/arch35/mega_moe_tiling.h"
#include "../../../../op_kernel/arch35/mega_moe_workspace_info.h"
#include "../../../../op_host/op_tiling/arch35/mega_moe_batch_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace MegaMoeUT {

class MegaMoeArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest TearDown" << std::endl;
    }
};

namespace {
constexpr uint64_t WORKSPACE_ALIGNMENT = 512U;
constexpr uint64_t READY_SLOT_BYTES = static_cast<uint64_t>(INT_CACHELINE) * sizeof(int32_t);

uint64_t AlignWorkspace(uint64_t value)
{
    return (value + WORKSPACE_ALIGNMENT - 1U) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

uint64_t WorkspaceOffset(GM_ADDR base, GM_ADDR ptr)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(base));
}

MegaMoeTilingData MakeMteGenericNoQuantTiling()
{
    MegaMoeTilingData tiling{};
    tiling.expertPerRank = 3U; // Avoid accidentally aligned flag sizes hiding a layout bug.
    tiling.bs = 512U;
    tiling.h = 1024U;
    tiling.hiddenDim = 2048U;
    tiling.epWorldSize = 8U;
    tiling.blockNumPerEP = 4U;
    tiling.maxOutputSize = 257U;
    tiling.topK = 8U;
    tiling.aicNum = 36U;
    tiling.blockAivNum = 72U;
    tiling.combineQuantMode = COMBINE_NO_QUANT;
    tiling.groupedMatmulMode = GROUPED_MATMUL_MODE_GENERAL;
    tiling.topoType = TOPO_TYPE_MTE;
    tiling.ubSize = 256U * 1024U;
    return tiling;
}
} // namespace

TEST_F(MegaMoeArch35TilingTest, MteGenericNoQuantWorkspaceOutputReadyLayout)
{
    MegaMoeTilingData tiling = MakeMteGenericNoQuantTiling();
    GM_ADDR base = reinterpret_cast<GM_ADDR>(uintptr_t{0x100000000ULL});
    WorkspaceInfo info(base, &tiling);
    ASSERT_NE(info.gmm2MmadResPtr, nullptr);
    ASSERT_NE(info.gmm2NoQuantReadyPtr, nullptr);

    uint64_t outputOffset = WorkspaceOffset(base, info.gmm2MmadResPtr);
    uint64_t readyOffset = WorkspaceOffset(base, info.gmm2NoQuantReadyPtr);
    uint64_t outputBytes =
        AlignWorkspace(2ULL * static_cast<uint64_t>(tiling.maxOutputSize) * static_cast<uint64_t>(tiling.h));
    uint64_t readyBytes = AlignWorkspace(static_cast<uint64_t>(tiling.expertPerRank) * READY_SLOT_BYTES);

    EXPECT_EQ(outputOffset % WORKSPACE_ALIGNMENT, 0U);
    EXPECT_EQ(readyOffset % WORKSPACE_ALIGNMENT, 0U);
    EXPECT_EQ(readyOffset, outputOffset + outputBytes);
    EXPECT_EQ(static_cast<uint64_t>(info.workspaceSize), readyOffset + readyBytes);
    for (uint32_t expertIdx = 0; expertIdx < tiling.expertPerRank; ++expertIdx) {
        EXPECT_EQ((readyOffset + static_cast<uint64_t>(expertIdx) * READY_SLOT_BYTES) % READY_SLOT_BYTES, 0U);
    }
}

TEST_F(MegaMoeArch35TilingTest, MteGenericNoQuantWorkspaceUses64BitOffsets)
{
    MegaMoeTilingData tiling = MakeMteGenericNoQuantTiling();
    tiling.h = 4096U;
    tiling.maxOutputSize = 524289U;
    GM_ADDR base = reinterpret_cast<GM_ADDR>(uintptr_t{0x100000000ULL});
    WorkspaceInfo info(base, &tiling);
    ASSERT_NE(info.gmm2MmadResPtr, nullptr);
    ASSERT_NE(info.gmm2NoQuantReadyPtr, nullptr);

    uint64_t outputOffset = WorkspaceOffset(base, info.gmm2MmadResPtr);
    uint64_t readyOffset = WorkspaceOffset(base, info.gmm2NoQuantReadyPtr);
    uint64_t outputBytes =
        AlignWorkspace(2ULL * static_cast<uint64_t>(tiling.maxOutputSize) * static_cast<uint64_t>(tiling.h));
    ASSERT_GT(outputBytes, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
    EXPECT_EQ(outputBytes, 0x100002000ULL);
    EXPECT_EQ(readyOffset - outputOffset, outputBytes);
    EXPECT_GT(static_cast<uint64_t>(info.workspaceSize),
              static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
}

TEST_F(MegaMoeArch35TilingTest, NewNoQuantReadyLayoutIsMteGenericOnly)
{
    MegaMoeTilingData tiling = MakeMteGenericNoQuantTiling();
    GM_ADDR base = reinterpret_cast<GM_ADDR>(uintptr_t{0x100000000ULL});
    tiling.topoType = TOPO_TYPE_URMA;
    WorkspaceInfo urmaInfo(base, &tiling, 1U);
    EXPECT_EQ(urmaInfo.gmm2MmadResPtr, nullptr);
    EXPECT_EQ(urmaInfo.gmm2NoQuantReadyPtr, nullptr);

    tiling = MakeMteGenericNoQuantTiling();
    tiling.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    WorkspaceInfo a8w4Info(base, &tiling);
    EXPECT_NE(a8w4Info.gmm2MmadResPtr, nullptr);
    EXPECT_EQ(a8w4Info.gmm2NoQuantReadyPtr, nullptr);

    tiling = MakeMteGenericNoQuantTiling();
    tiling.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W8_NZ;
    WorkspaceInfo nzInfo(base, &tiling);
    EXPECT_NE(nzInfo.gmm2MmadResPtr, nullptr);
    EXPECT_NE(nzInfo.gmm2NoQuantReadyPtr, nullptr);
}

TEST_F(MegaMoeArch35TilingTest, BalancedTokenOwnershipCoversEveryTokenOnce)
{
    constexpr uint32_t logicalAiv1Num = 36U;
    const std::vector<uint32_t> tokenCounts{0U, 1U, 35U, 36U, 255U, 256U, 257U};
    for (uint32_t tokenCount : tokenCounts) {
        std::vector<uint32_t> owners(tokenCount, 0U);
        uint32_t previousEnd = 0U;
        for (uint32_t coreIdx = 0; coreIdx < logicalAiv1Num; ++coreIdx) {
            MegaMoeImpl::TokenRange range =
                MegaMoeImpl::GetBalancedTokenRange(tokenCount, coreIdx, logicalAiv1Num);
            EXPECT_EQ(range.start, previousEnd) << "tokenCount=" << tokenCount << ", coreIdx=" << coreIdx;
            EXPECT_LE(range.start + range.count, tokenCount);
            for (uint32_t tokenIdx = range.start; tokenIdx < range.start + range.count; ++tokenIdx) {
                owners[tokenIdx]++;
            }
            previousEnd = range.start + range.count;
        }
        EXPECT_EQ(previousEnd, tokenCount);
        for (uint32_t ownerCount : owners) {
            EXPECT_EQ(ownerCount, 1U) << "tokenCount=" << tokenCount;
        }
    }

    for (uint32_t coreIdx = 0; coreIdx < logicalAiv1Num; ++coreIdx) {
        MegaMoeImpl::TokenRange range =
            MegaMoeImpl::GetBalancedTokenRange(256U, coreIdx, logicalAiv1Num);
        EXPECT_TRUE(range.count == 7U || range.count == 8U);
    }
    EXPECT_EQ(MegaMoeImpl::GetBalancedTokenRange(256U, 0U, 0U).count, 0U);
}

TEST_F(MegaMoeArch35TilingTest, ExpertsPerBatchUsesTileWorkAndAicCount)
{
    // Fixed EP8 case: 8 GMM1 tiles/expert and 16 GMM2 tiles/expert, so
    // ceil(36 / min(8, 16)) = 5 experts are needed to fill the AICs.
    EXPECT_EQ(optiling::MegaMoeBatchTiling::CalcExpertsPerBatch(512U, 8U, 16U, 4096U, 4096U, 36U), 5U);

    // The result follows available AICs and tile work rather than epWorldSize.
    EXPECT_EQ(optiling::MegaMoeBatchTiling::CalcExpertsPerBatch(512U, 8U, 16U, 4096U, 4096U, 64U), 8U);
    EXPECT_EQ(optiling::MegaMoeBatchTiling::CalcExpertsPerBatch(512U, 8U, 16U, 8192U, 4096U, 36U), 3U);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(12777472)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_clamp", Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::max())},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 16UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H5120_BS256_FP8E5M2)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 6}, {256, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 6}, {256, 6}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 3072, 5120}, {8, 3072, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8, 5120, 1536}, {8, 5120, 1536}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8, 3072, 80, 2}, {8, 3072, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{8, 5120, 24, 2}, {8, 5120, 24, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(23904256)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_clamp", Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::max())},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H7168_BS512)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 7168, 2048}, {2, 7168, 2048}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 4096, 112, 2}, {2, 4096, 112, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{2, 7168, 32, 2}, {2, 7168, 32, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(89060352)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_clamp", Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::max())},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 16UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, DifferentNConfig)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9598976)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_clamp", Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::max())},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // namespace MegaMoeUT
