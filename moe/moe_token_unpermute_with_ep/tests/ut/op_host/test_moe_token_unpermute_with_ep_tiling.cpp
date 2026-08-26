/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_token_unpermute_with_ep_tiling.cpp
 * \brief
 */
#include <iostream>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/moe_token_unpermute_with_ep_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;

namespace {
void InitParsePlatformInfo(fe::PlatFormInfos &platformInfo)
{
    platformInfo.Init();
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileJson);
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    socInfos["ai_core_cnt"] = to_string(compileInfoJson["hardware_info"]["CORE_NUM"].get<uint32_t>());
    socInfos["l2_size"] = to_string(compileInfoJson["hardware_info"]["L2_SIZE"].get<uint32_t>());
    socInfos["core_type_list"] = "AICore";
    aicoreSpec["ub_size"] = to_string(compileInfoJson["hardware_info"]["UB_SIZE"].get<uint32_t>());
    aicoreSpec["l1_size"] = to_string(compileInfoJson["hardware_info"]["L1_SIZE"].get<uint32_t>());
    map<string, string> versions = {{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("version", versions);
}

#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
constexpr int64_t AlignUp32(const int64_t value)
{
    return (value + 31) / 32 * 32;
}

constexpr int64_t Fp16TokenBufferBytes(const int64_t tokenLength)
{
    return AlignUp32(tokenLength * static_cast<int64_t>(sizeof(int32_t))) +
           AlignUp32(tokenLength * static_cast<int64_t>(sizeof(int16_t))) +
           AlignUp32(tokenLength * static_cast<int64_t>(sizeof(float)));
}
#endif
} // namespace

struct MoeTokenUnpermuteWithEpCompileInfo {};

class MoeTokenUnpermuteWithEpTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    if (tiling_data == nullptr) {
        return "";
    }
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_bf16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{6143, 5120}, {6143, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
#else
            {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
#endif
            {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 3;
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
    string expectTilingData = "5120 0 8 1 6144 6143 5120 1 0 96 0 96 1 0 4 ";
#else
    string expectTilingData = "5120 0 8 1 6144 49152 5120 1 0 96 0 96 1 0 4 ";
#endif
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_bf16_2)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{6143, 20480}, {6143, 20480}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND},
            //{{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{768, 20480}, {768, 20480}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 0;
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
    string expectTilingData = "20480 1 8 1 6144 6143 8192 2 4096 12 0 12 1 0 2 ";
#else
    string expectTilingData = "20480 1 8 1 6144 6143 17920 1 2560 12 0 12 1 0 2 ";
#endif
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_not_none_bf16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{49150, 5120}, {49150, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "5120 0 8 1 49151 49150 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_fp16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{6143, 20480}, {6143, 20480}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND},
            //{{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{768, 20480}, {768, 20480}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 0;
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
    string expectTilingData = "20480 1 8 1 6144 6143 8192 2 4096 12 0 12 1 0 2 ";
#else
    string expectTilingData = "20480 1 8 1 6144 6143 17920 1 2560 12 0 12 1 0 2 ";
#endif
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_not_none_fp16)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{49150, 5120}, {49150, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "5120 0 8 1 49151 49150 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_none_fp32)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{6143, 20480}, {6143, 20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6144}, {6144}}, ge::DT_INT32, ge::FORMAT_ND},
            //{{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{768, 20480}, {768, 20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 6144})},
#else
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
#endif
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 0;
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
    string expectTilingData = "20480 1 8 1 6144 6143 8704 2 3072 12 0 12 1 0 2 ";
#else
    string expectTilingData = "20480 1 8 1 49151 6143 20480 1 0 12 0 12 1 0 2 ";
#endif
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_not_none_fp32)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{49150, 5120}, {49150, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 49151})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "5120 0 8 1 49151 49150 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_small_tokens_num_core)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_without_range_attr)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_default_range_ignores_num_topk)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_empty_range_uses_fallback)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_fallback_without_range_rejects_short_tokens)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{63, 128}, {63, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_fallback_empty_range_rejects_short_tokens)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{63, 128}, {63, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_fallback_default_range_rejects_short_tokens)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{63, 128}, {63, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_rejects_non_1d_sorted_indices)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 1}, {8, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_rejects_zero_probs_topk)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{0, 128}, {0, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 0}, {8, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_ep_range_rejects_negative_start)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({-1, 7})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_ep_range_rejects_non_increasing_bounds)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_ep_range_rejects_token_rows_mismatch)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_fp16_token_buffers_are_aligned_independently)
{
    constexpr int64_t ubSize = 253952;
    constexpr int64_t fixedBufferSpace = 512 * (2 * (2 + 4 - 1) + 4 * (2 + 4)) + 256;
    constexpr int64_t tokenBufferSpace = ubSize - fixedBufferSpace;
    constexpr int64_t expectedTokenTileLength = 23624;
    static_assert(Fp16TokenBufferBytes(expectedTokenTileLength) <= tokenBufferSpace);
    static_assert(Fp16TokenBufferBytes(expectedTokenTileLength + 1) > tokenBufferSpace);

    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{23628, 1}, {23628, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{23628}, {23628}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{23628, 1}, {23628, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{23628, 1}, {23628, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 23628})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo, "Ascend950", 1, ubSize);
    const string expectTilingData = "1 0 1 0 23628 23628 1 1 0 23628 0 23624 1 4 4 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2, expectTilingData, {16777216});
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_rejects_ub_that_cannot_hold_one_aligned_token_group)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo, "Ascend950", 1, 20510);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
#endif

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_prob_small_ub_token_split)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{767, 8192}, {767, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
#else
            {{{49152, 8192}, {49152, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
#endif
            {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{768, 64}, {768, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{768, 8192}, {768, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 768})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo, "Ascend910B", 64, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_probs_invalid_dtype)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{7, 128}, {7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#else
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#endif
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_input_topk_invalid)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_tokens_unknown_dtype)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{7, 128}, {7, 128}}, static_cast<ge::DataType>(100), ge::FORMAT_ND},
#else
            {{{64, 128}, {64, 128}}, static_cast<ge::DataType>(100), ge::FORMAT_ND},
#endif
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, static_cast<ge::DataType>(100), ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_indices_int64_dtype)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{7, 128}, {7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#else
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#endif
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_tokens_not_2d)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_probs_topk_mismatch)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{7, 128}, {7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#else
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#endif
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_probs_dim_mismatch)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
            {{{7, 128}, {7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#else
            {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
#endif
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 128}, {8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_probs_topk_too_large)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithEp",
        {
            {{{513, 128}, {513, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{513}, {513}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 513}, {1, 513}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 128}, {1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(513)},
#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 513})},
#else
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
#endif
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({})}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenUnpermuteWithEpTiling, test_tiling_parse)
{
    MoeTokenUnpermuteWithEpCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    InitParsePlatformInfo(platformInfo);
    const char *compileJson =
        R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "BT_SIZE": 0, "load3d_constraints": "1", "socVersion":"Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenUnpermuteWithEp"))
                      .OpName(ge::AscendString("MoeTokenUnpermuteWithEp"))
                      .IOInstanceNum({1, 1, 1}, {1})
                      .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .Build();
    auto *parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenUnpermuteWithEp");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
