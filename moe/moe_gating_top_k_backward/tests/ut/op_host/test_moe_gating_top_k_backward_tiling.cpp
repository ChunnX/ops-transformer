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
 * \file test_moe_gating_top_k_backward_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/moe_gating_top_k_backward_tiling.h"

using namespace std;

class MoeGatingTopKBackwardTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKBackwardTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

// ============================================
// Success cases
// ============================================

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_succ_01) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_succ_02) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 64}, {16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_succ_03) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{32, 128}, {32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{32, 4}, {32, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{32, 4}, {32, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{32, 128}, {32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-10f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================
// Fail cases - mapped to boundary.txt
// ============================================

// Boundary case 1: empty tensor
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_01) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 2: x_norm dim != 2
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_02) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 3: grad_y dim != 2
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_03) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 4: expert_idx dim != 2
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_04) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 5: M mismatch - grad_y M != x_norm M
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_05) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 5: M mismatch - expert_idx M != grad_y M
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_06) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 6: K mismatch - expert_idx K != grad_y K
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_07) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 7: k <= 0
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_08) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 0}, {16, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 0}, {16, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 8: k > expertNum
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_09) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 9: expertNum > 2048
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_10) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 2049}, {16, 2049}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 2049}, {16, 2049}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 10: renorm != 0
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_11) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 11: normType != 1
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_12) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 12: x_norm dtype != FP32
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_13) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 13: grad_y dtype invalid
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_14) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 14: expert_idx dtype != INT32
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_15) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 256}, {16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Boundary case 15: UB space insufficient (maxRows <= 0)
TEST_F(MoeGatingTopKBackwardTiling, moe_gating_top_k_backward_tiling_fail_16) {
    optiling::MoeGatingTopKBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKBackward",
                                              {
                                                {{{16, 2048}, {16, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 2048}, {16, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 2048}, {16, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 2048}, {16, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"renorm",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"norm_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"routed_scaling_factor",Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                {"eps",Ops::Transformer::AnyValue::CreateFrom<float>(1e-20f)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "TODO ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
