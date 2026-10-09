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
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_kernel/arch35/common/mega_moe_workspace.h"
#include "../../../../op_kernel/arch35/mega_moe_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace MegaMoeUT {

constexpr int64_t ARCH35_CCL_BUFFER_SIZE = 1024LL * 1024LL * 1024LL;

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

static void ExpectCclWorkspaceSuffix(const TilingInfo &tilingInfo)
{
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    MegaMoeImpl::WorkspaceLayout workspaceLayout(&tilingData);

    ASSERT_GT(workspaceLayout.workspaceSize, 0);
    EXPECT_GT(tilingData.cclWorkspaceOffset, 0U);
    EXPECT_LE(tilingData.cclWorkspaceOffset + static_cast<uint64_t>(workspaceLayout.workspaceSize),
              static_cast<uint64_t>(ARCH35_CCL_BUFFER_SIZE));
    // Tiling now publishes only the framework LibAPI workspace; the operator workspace is in the CCL suffix above.
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_LT(static_cast<int64_t>(tilingInfo.workspaceSizes[0]), workspaceLayout.workspaceSize);
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
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 664612UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ExpectCclWorkspaceSuffix(tilingInfo);
}

TEST_F(MegaMoeArch35TilingTest, H5120_BS256_URMA_InvalidRankNumPerServer)
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
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H7168_BS512_MTE_RankNumIgnored)
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
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 664612UL;
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
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 8995UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

static void RunUrmaTilingCase(int64_t xBs, int64_t topkIdsBs, int64_t topkWeightsBs, int64_t numMaxTokensPerRank,
                              int64_t maxRecvTokenNum, int64_t hiddenDim, const std::string &activation,
                              const std::vector<float> &activationParams, ge::graphStatus expectedStatus)
{
    ASSERT_GT(hiddenDim, 0);
    ASSERT_EQ(hiddenDim % (MegaMoeImpl::ACTIVATION_N_HALF * MegaMoeImpl::MXFP_DIVISOR_SIZE), 0);

    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{xBs, 4096}, {xBs, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{topkIdsBs, 8}, {topkIdsBs, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{topkWeightsBs, 8}, {topkWeightsBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, hiddenDim, 4096}, {8, hiddenDim, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},
            {{{8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF},
              {8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_FRACTAL_NZ},
            {{{8, hiddenDim, 64, 2}, {8, hiddenDim, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF / MegaMoeImpl::MXFP_DIVISOR_SIZE, 2},
              {8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF / MegaMoeImpl::MXFP_DIVISOR_SIZE, 2}},
             ge::DT_FLOAT8_E8M0,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{xBs, 4096}, {xBs, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxRecvTokenNum)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numMaxTokensPerRank)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>(activation)},
            {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(activationParams)},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    const uint64_t expectedTilingKey = expectedStatus == ge::GRAPH_SUCCESS ? 42607652UL : 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);

    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ExpectCclWorkspaceSuffix(tilingInfo);

    if (activation != "situglu") {
        return;
    }

    ASSERT_TRUE(activationParams.size() == 1U || activationParams.size() == 2U);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingData.actMode, static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU));
    EXPECT_EQ(tilingData.actSubMode,
              static_cast<uint8_t>(activationParams.size() == 2U ? MegaMoeImpl::MegaMoeActSubMode::LINEAR :
                                                                   MegaMoeImpl::MegaMoeActSubMode::DEFAULT));
    EXPECT_FLOAT_EQ(tilingData.activationAlpha, activationParams.size() == 2U ? activationParams[1] : 0.0f);
    EXPECT_FLOAT_EQ(tilingData.activationBeta, activationParams[0]);
}

static void RunUrmaVariableBsInputShapeCase(int64_t xBs, int64_t topkIdsBs, int64_t topkWeightsBs,
                                            int64_t numMaxTokensPerRank, int64_t maxRecvTokenNum,
                                            ge::graphStatus expectedStatus)
{
    RunUrmaTilingCase(xBs, topkIdsBs, topkWeightsBs, numMaxTokensPerRank, maxRecvTokenNum, 2048, "swiglu",
                      {std::numeric_limits<float>::max()}, expectedStatus);
}

static void RunUrmaVariableBsCase(int64_t bs, int64_t numMaxTokensPerRank, int64_t maxRecvTokenNum,
                                  ge::graphStatus expectedStatus)
{
    RunUrmaVariableBsInputShapeCase(bs, bs, bs, numMaxTokensPerRank, maxRecvTokenNum, expectedStatus);
}

static void RunUrmaActivationCase(int64_t hiddenDim, const std::string &activation,
                                  const std::vector<float> &activationParams, ge::graphStatus expectedStatus)
{
    RunUrmaTilingCase(128, 128, 128, 256, 0, hiddenDim, activation, activationParams, expectedStatus);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokens256_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 256, 0, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_Default_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {1.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_Linear_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {1.0f, 2.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N768_SITU_URMA_2Servers)
{
    RunUrmaActivationCase(768, "situglu", {1.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N512_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(512, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N1536_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(1536, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N8192_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(8192, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N256_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(256, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N640_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(640, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N8448_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(8448, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SwiGluStep_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "swiglustep", {}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_MissingBeta_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokensZero_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 0, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokens64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 64, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS0_NumMaxTokens256_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(0, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_TopkIdsBS64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsInputShapeCase(128, 64, 128, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_TopkWeightsBS64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsInputShapeCase(128, 128, 64, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_RouteCapacityOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / 8 + 1;
    RunUrmaVariableBsCase(128, numMaxTokensPerRank, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_OutputCapacityOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / 16 + 1;
    RunUrmaVariableBsCase(128, numMaxTokensPerRank, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_MaxRecvTokenNumOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t maxRecvTokenNum = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    RunUrmaVariableBsCase(128, 256, maxRecvTokenNum, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_MaxRecvTokenNumAboveCapacity_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 256, 4097, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, BS128_NumMaxTokens256_URMA_WorkspaceUsesCapacityMaskAndAllServers)
{
    MegaMoeTilingData tilingData{};
    tilingData.bs = 128;
    tilingData.h = 4096;
    tilingData.hiddenDim = 2048;
    tilingData.epWorldSize = 2;
    tilingData.maxOutputSize = 4096;
    tilingData.topK = 8;
    tilingData.aicNum = 28;
    tilingData.moeExpertPerRank = 8;
    tilingData.combineSyncSlotCountPerExpert = 1;
    tilingData.topoType = MegaMoeImpl::TOPO_TYPE_URMA;
    tilingData.numMaxTokensPerRank = 256;
    tilingData.rankNumPerServer = 1;

    MegaMoeImpl::WorkspaceLayout workspaceLayout(&tilingData);
    const int64_t actualMaskWorkspaceBytes =
        workspaceLayout.dispatchRelaySendQueueOffset - workspaceLayout.maskSlotOffset;
    const int64_t capacityMaskSlotBytes = MegaMoeImpl::CalcDispatchMaskAlignSize(&tilingData) + MegaMoeImpl::ALIGN_32;
    const int64_t expectedMaskWorkspaceBytes = Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize * capacityMaskSlotBytes,
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
    const int64_t actualRelayQueueBytes =
        workspaceLayout.dispatchRemoteReadyFlagSnapshotOffset - workspaceLayout.dispatchRelaySendQueueOffset;
    const int64_t serverNum = Ops::Base::CeilDiv(tilingData.epWorldSize, tilingData.rankNumPerServer);
    const int64_t expectedRelayQueueBytes = Ops::Base::CeilAlign(
        serverNum * (MegaMoeImpl::ALIGN_32 + static_cast<int64_t>(tilingData.bs) * MegaMoeImpl::ALIGN_32),
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
    const int64_t localBsMaskSlotBytes =
        MegaMoeImpl::CalcDispatchMaskAlignSizeBy(tilingData.bs, tilingData.topK) + MegaMoeImpl::ALIGN_32;
    const int64_t localBsMaskWorkspaceBytes = Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize * localBsMaskSlotBytes,
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));

    EXPECT_EQ(actualMaskWorkspaceBytes, expectedMaskWorkspaceBytes);
    EXPECT_EQ(actualRelayQueueBytes, expectedRelayQueueBytes);
    EXPECT_NE(actualMaskWorkspaceBytes, localBsMaskWorkspaceBytes);
}

static void RunA8W4FormatCase(ge::Format weightOneFormat, ge::Format weightTwoFormat, ge::graphStatus expectedStatus,
                              uint64_t expectedTilingKey)
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
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, weightOneFormat},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, weightTwoFormat},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    const auto &layout = tilingData.a8w4L1Layout;
    EXPECT_EQ(layout.aBufferNum, 3U);
    EXPECT_EQ(layout.bBufferNum, 4U);
    EXPECT_EQ(layout.tileK, 256U);
    EXPECT_EQ(layout.scaleK, 2048U);
    const uint64_t expectedBOffsets[] = {192U * 1024, 320U * 1024, 256U * 1024, 384U * 1024};
    for (uint64_t slot = 0; slot < 4U; ++slot) {
        EXPECT_EQ(layout.bOffsets[slot], expectedBOffsets[slot]);
    }
    EXPECT_EQ(layout.aOffsets[2] + layout.aSlotBytes, layout.bOffsets[0]);
    EXPECT_EQ(layout.scaleAOffsets[0], 448U * 1024);
    EXPECT_EQ(layout.scaleBOffsets[1] + layout.scaleBSlotBytes, 512U * 1024);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A8W4_URMA)
{
    RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 46802984UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A8W4_URMA_InvalidWeight2Format)
{
    RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_ND, ge::GRAPH_FAILED, 0UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_URMA)
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
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ_C0_32},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 62138408UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

static void RunA4W4NzFormatCase(ge::Format weightOneFormat, ge::Format weightTwoFormat, ge::graphStatus expectedStatus,
                                uint64_t expectedTilingKey)
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
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, weightOneFormat},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, weightTwoFormat},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 62138408UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA_InvalidWeight1Format)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_FAILED, 0UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA_InvalidWeight2Format)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::GRAPH_FAILED, 0UL);
}

static void RunPerExpertTensorListCase(uint32_t weightTwoTensorCount, ge::graphStatus expectedStatus)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    using TensorDescription = gert::TilingContextPara::TensorDescription;
    std::vector<TensorDescription> inputs{
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < weightTwoTensorCount; ++tensorIdx) {
        inputs.push_back({{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{2048, 64, 2}, {2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{4096, 16, 2}, {4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    }

    const std::vector<TensorDescription> outputs{
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs{
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
        {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"dispatch_quant_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
        {"shared_expert_quant_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
        {"activation_params",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
        {"activation_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
    const std::vector<uint32_t> inputInstanceNum{
        1U, 1U, 1U, 1U, 4U, weightTwoTensorCount, 4U, 4U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    };
    const std::vector<uint32_t> outputInstanceNum{1U, 1U};
    gert::TilingContextPara tilingContextPara("MegaMoe", inputs, outputs, attrs, inputInstanceNum, outputInstanceNum,
                                              &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    const uint64_t expectedTilingKey = expectedStatus == ge::GRAPH_SUCCESS ? 42607652UL : 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN_URMA_PerExpertTensorLists)
{
    RunPerExpertTensorListCase(4U, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN_URMA_PerExpertTensorListCountMismatch)
{
    RunPerExpertTensorListCase(3U, ge::GRAPH_FAILED);
}

static MegaMoeTilingData MakeMteRingLayoutCase()
{
    MegaMoeTilingData tilingData{};
    tilingData.bs = 128;
    tilingData.h = 4096;
    tilingData.hiddenDim = 2048;
    tilingData.epWorldSize = 4;
    tilingData.numMaxTokensPerRank = 8192;
    tilingData.maxOutputSize = 32768;
    tilingData.topK = 8;
    tilingData.aicNum = 28;
    tilingData.moeExpertPerRank = 4;
    tilingData.topoType = MegaMoeImpl::TOPO_TYPE_MTE;
    tilingData.moeGmmMode = MegaMoeImpl::GMM_MODE_A8W4_NZ;
    tilingData.mGroupsPerWave = 14;
    return tilingData;
}

static void ExpectRoutedBufferRows(const MegaMoeTilingData &tilingData, int64_t rows)
{
    MegaMoeImpl::WorkspaceLayout layout(&tilingData);
    const int64_t scaleGroup = 64;
    const int64_t storageAlign = 512;
    const int64_t dispatchElementsPerByte = MegaMoeImpl::IsA4W4GmmMode(tilingData.moeGmmMode) ? 2LL : 1LL;
    const int64_t dispatchBytes = rows * tilingData.h / dispatchElementsPerByte;
    const int64_t dispatchScaleBytes = rows * Ops::Base::CeilDiv(static_cast<int64_t>(tilingData.h), scaleGroup) * 2LL;
    const int64_t activationBytes = rows * tilingData.hiddenDim / 2LL;
    const int64_t activationScaleBytes =
        rows * Ops::Base::CeilDiv(static_cast<int64_t>(tilingData.hiddenDim / 2U), scaleGroup) * 2LL;
    EXPECT_EQ(layout.dispatchRevScaleOffset - layout.dispatchRevDataOffset,
              Ops::Base::CeilAlign(dispatchBytes, storageAlign));
    const int64_t afterDispatchScale = layout.dispatchRevWeightsOffset >= 0 ? layout.dispatchRevWeightsOffset :
                                                                           layout.activationQuantDataOffset;
    EXPECT_EQ(afterDispatchScale - layout.dispatchRevScaleOffset,
              Ops::Base::CeilAlign(dispatchScaleBytes, storageAlign));
    EXPECT_EQ(layout.activationQuantScaleOffset - layout.activationQuantDataOffset,
              Ops::Base::CeilAlign(activationBytes, storageAlign));
    EXPECT_EQ(layout.expertRecvTokenCountOffset - layout.activationQuantScaleOffset,
              Ops::Base::CeilAlign(activationScaleBytes, storageAlign));
    EXPECT_LE(layout.dispatchRevScaleOffset + Ops::Base::CeilAlign(dispatchScaleBytes, storageAlign),
              layout.activationQuantDataOffset);
    const int64_t firstGmmResult = layout.gmm1MmadResOffset >= 0 ? layout.gmm1MmadResOffset : layout.gmm2MmadResOffset;
    EXPECT_LE(layout.activationQuantScaleOffset + Ops::Base::CeilAlign(activationScaleBytes, storageAlign),
              firstGmmResult);
    if (layout.gmm1MmadResOffset >= 0) {
        const int64_t afterGmm1 = layout.gmm1MmadResOffset + static_cast<int64_t>(2) * rows * tilingData.hiddenDim;
        EXPECT_EQ(layout.gmm2MmadResOffset, Ops::Base::CeilAlign(afterGmm1, storageAlign));
    }
    const int64_t afterGmm2 = layout.gmm1TileStatusOffset >= 0 ? layout.gmm1TileStatusOffset :
                             layout.dedupRowDescOffset >= 0 ? layout.dedupRowDescOffset : layout.workspaceSize;
    EXPECT_EQ(afterGmm2 - layout.gmm2MmadResOffset,
              Ops::Base::CeilAlign(static_cast<int64_t>(2) * rows * tilingData.h, storageAlign));
}

TEST_F(MegaMoeArch35TilingTest, MteRingsUseWaveBudgetAndRetainGlobalMetadata)
{
    auto tilingData = MakeMteRingLayoutCase();
    const int64_t ringRows = 3LL * 14LL * 256LL;
    ASSERT_TRUE(MegaMoeImpl::IsMteRingBufferEnabled(&tilingData));
    EXPECT_EQ(MegaMoeImpl::CalcMteRingRowCapacity(&tilingData), ringRows);
    ExpectRoutedBufferRows(tilingData, ringRows);
    MegaMoeImpl::WorkspaceLayout first(&tilingData);
    EXPECT_EQ(first.flagActivationToGmm2Offset - first.metaInfoOffset, 32768LL * 32LL);
    ASSERT_GE(first.ringProgressOffset, first.flagActivationToGmm2Offset);
    EXPECT_EQ(first.ringProgressOffset + 3LL * 28LL * 16LL * 4LL,
              first.flagActivationToGmm2Offset + first.flagResetElementCount * 4LL);

    tilingData.maxOutputSize *= 2U;
    MegaMoeImpl::WorkspaceLayout second(&tilingData);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingRowCapacity(&tilingData), ringRows);
    ExpectRoutedBufferRows(tilingData, ringRows);
    EXPECT_EQ(second.flagActivationToGmm2Offset - second.metaInfoOffset, 65536LL * 32LL);
    EXPECT_GT(second.workspaceSize, first.workspaceSize);
    EXPECT_LT(second.workspaceSize - first.workspaceSize, 32768LL * tilingData.h);

    tilingData.mGroupsPerWave = 7;
    EXPECT_EQ(MegaMoeImpl::CalcMteRingRowCapacity(&tilingData), ringRows / 2LL);
    ExpectRoutedBufferRows(tilingData, ringRows / 2LL);
}

TEST_F(MegaMoeArch35TilingTest, MteRingsOnlyAllocateActualGmm1GmResults)
{
    auto tilingData = MakeMteRingLayoutCase();
    for (const uint8_t mode : {MegaMoeImpl::GMM_MODE_A8W8_ND, MegaMoeImpl::GMM_MODE_A8W8_NZ,
                               MegaMoeImpl::GMM_MODE_A4W4_ND, MegaMoeImpl::GMM_MODE_A4W4_NZ}) {
        tilingData.moeGmmMode = mode;
        tilingData.topkWeightsPrefetch = 0;
        MegaMoeImpl::WorkspaceLayout noPrefetch(&tilingData);
        EXPECT_EQ(noPrefetch.gmm1MmadResOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);
        ExpectRoutedBufferRows(tilingData, 3LL * 14LL * 256LL);
        tilingData.topkWeightsPrefetch = 1;
        tilingData.maxTilesPerExpert = 1024;
        MegaMoeImpl::WorkspaceLayout prefetch(&tilingData);
        EXPECT_GE(prefetch.gmm1MmadResOffset, 0);
        EXPECT_EQ(prefetch.gmm1TileStatusElementCount, (4LL * 1024LL + 1LL) * 16LL);
        ExpectRoutedBufferRows(tilingData, 3LL * 14LL * 256LL);
    }
}

TEST_F(MegaMoeArch35TilingTest, MteRingsKeepDedupQuantizedAndUrmaLayouts)
{
    for (const int32_t dedupMode : {1, 2, 3}) {
        auto tilingData = MakeMteRingLayoutCase();
        tilingData.dedupMode = dedupMode;
        EXPECT_FALSE(MegaMoeImpl::IsMteRingBufferEnabled(&tilingData));
        EXPECT_EQ(MegaMoeImpl::CalcMteRingRowCapacity(&tilingData), 32768LL);
        ExpectRoutedBufferRows(tilingData, 32768LL);
        MegaMoeImpl::WorkspaceLayout layout(&tilingData);
        EXPECT_EQ(layout.ringProgressOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);
    }
    for (const int64_t combineQuantMode : {3LL, 4LL}) {
        auto tilingData = MakeMteRingLayoutCase();
        tilingData.combineQuantMode = combineQuantMode;
        EXPECT_FALSE(MegaMoeImpl::IsMteRingBufferEnabled(&tilingData));
        ExpectRoutedBufferRows(tilingData, 32768LL);
        MegaMoeImpl::WorkspaceLayout layout(&tilingData);
        EXPECT_EQ(layout.ringProgressOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);
    }
    auto urmaData = MakeMteRingLayoutCase();
    urmaData.topoType = MegaMoeImpl::TOPO_TYPE_URMA;
    urmaData.rankNumPerServer = 2;
    EXPECT_FALSE(MegaMoeImpl::IsMteRingBufferEnabled(&urmaData));
    EXPECT_EQ(MegaMoeImpl::CalcMteRingRowCapacity(&urmaData), 32768LL);
    MegaMoeImpl::WorkspaceLayout urma(&urmaData);
    EXPECT_EQ(urma.dispatchRevScaleOffset - urma.dispatchRevDataOffset, 32768LL * 4096LL);
    EXPECT_EQ(urma.ringProgressOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);

    auto unsetWave = MakeMteRingLayoutCase();
    unsetWave.mGroupsPerWave = 0;
    EXPECT_FALSE(MegaMoeImpl::IsMteRingBufferEnabled(&unsetWave));
    ExpectRoutedBufferRows(unsetWave, 32768LL);
}

TEST_F(MegaMoeArch35TilingTest, MteRingMappingPadsExpertTailsAndWrapsFullGroups)
{
    EXPECT_FALSE(MegaMoeImpl::IsMteRingBufferEnabled(nullptr));
    EXPECT_EQ(MegaMoeImpl::CalcMteRingRowCapacity(nullptr), 0);
    const uint64_t ringRows = 3ULL * 4ULL * 256ULL;
    uint64_t paddedRow = 0;
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(paddedRow, ringRows), 0U);
    paddedRow += MegaMoeImpl::CalcMtePaddedRowCount(100);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(paddedRow, ringRows), 256U);
    paddedRow += MegaMoeImpl::CalcMtePaddedRowCount(300);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(paddedRow, ringRows), 768U);
    paddedRow += MegaMoeImpl::CalcMtePaddedRowCount(50);
    EXPECT_EQ(paddedRow, 1024U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(0), 0U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(1), 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(255), 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(256), 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(257), 512U);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(ringRows - 256U, ringRows), ringRows - 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(ringRows, ringRows), 0U);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(5U * ringRows + 17U, ringRows), 17U);
    const uint64_t largeGroupRow = (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL) * 256ULL;
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(largeGroupRow, ringRows), largeGroupRow % ringRows);
}

} // namespace MegaMoeUT
