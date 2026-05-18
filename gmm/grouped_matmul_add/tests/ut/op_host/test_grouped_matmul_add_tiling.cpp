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
 * \file test_grouped_matmul_add.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/grouped_matmul_add_tiling.h"
#include "../../../../grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class GroupedMatmulAddTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulAddTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulAddTiling TearDown" << std::endl;
    }
};

namespace {

optiling::GMMCompileInfo CreateArch35CompileInfo()
{
    return {
        32,         // aicNum
        64,         // aivNum
        262144,     // ubSize
        524288,     // l1Size
        134217728,  // l2Size
        262144,     // l0CSize
        65536,      // l0ASize
        65536,      // l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
}

} // namespace

// ========== arch35 path abnormal tests ==========
// EZ0013: x or weight has wrong dim (xDimNum != 2)
TEST_F(GroupedMatmulAddTiling, test_abnormal_wrong_x_dim_arch35)
{
    auto compileInfo = CreateArch35CompileInfo();
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd",
        {{{{1280}, {1280}}, ge::DT_BF16, ge::FORMAT_ND},          // x is 1D -> dim mismatch
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "3510", 32, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// EZ0026: wrong group_type (must be 2 = SPLIT_K)
TEST_F(GroupedMatmulAddTiling, test_abnormal_wrong_grouptype_arch35)
{
    auto compileInfo = CreateArch35CompileInfo();
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd",
        {{{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},   // 3 is invalid
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "3510", 32, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// EZ0027: wrong transposeX / transposeWeight combo (transposeX must be TRUE)
TEST_F(GroupedMatmulAddTiling, test_abnormal_wrong_transpose_arch35)
{
    auto compileInfo = CreateArch35CompileInfo();
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd",
        {{{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},  // must be true
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "3510", 32, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// EZ0018: non-ND format (use FORMAT_FRACTAL_NZ for x)
TEST_F(GroupedMatmulAddTiling, test_abnormal_wrong_format_arch35)
{
    auto compileInfo = CreateArch35CompileInfo();
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd",
        {{{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},  // NZ format not supported
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "3510", 32, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// EZ0015: groupNum out of range (groupList shape {0})
TEST_F(GroupedMatmulAddTiling, test_abnormal_groupnum_zero_arch35)
{
    auto compileInfo = CreateArch35CompileInfo();
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd",
        {{{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},     // groupList shape {0} -> groupNum=0
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "3510", 32, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// EZ0015: groupNum out of range (groupList shape {1025})
TEST_F(GroupedMatmulAddTiling, test_abnormal_groupnum_large_arch35)
{
    auto compileInfo = CreateArch35CompileInfo();
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd",
        {{{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1025}, {1025}}, ge::DT_INT64, ge::FORMAT_ND},  // groupList shape {1025} -> groupNum=1025
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "3510", 32, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulAddTiling, test_tiling_bf16)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd", // op_name
                                              {                   // input info
                                               // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                               {{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              // output info
                                              {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData =
        "2 128 2 0 1481763717121 5497558139447 1481763718400 5497558139136 1099511627904 34359738432 4294967304 1 0 "
        "1970324836974592 131072 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupedMatmulAddTiling, test_tiling_fp16)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd", // op_name
                                              {                   // input info
                                               // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                               {{{1280, 345}, {1280, 345}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1280, 567}, {1280, 567}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              // output info
                                              {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData =
        "2 128 2 0 1481763717121 5497558139447 1481763718400 5497558139136 1099511627904 34359738432 4294967304 1 0 "
        "1970324836974592 131072 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupedMatmulAddTiling, test_tiling_bf16_v2)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd", // op_name
                                              {                   // input info
                                               // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                               {{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              // output info
                                              {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                  {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                  {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    int groupListTypeMode = 2;
    int groupListTypeIndex = 3;
    string expectTilingData = "";
    for (int i = 0; i < groupListTypeMode; i++) {
        expectTilingData = "2 128 2 " + to_string(i) + " 1481763717121 5497558139447 1481763718400 5497558139136 "
        "1099511627904 34359738432 4294967304 1 0 1970324836974592 131072 4294967297 4294967297 17179869188 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 ";
        tilingContextPara.attrs_.at(groupListTypeIndex).attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(i);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    }
}

TEST_F(GroupedMatmulAddTiling, test_tiling_fp16_v2)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd", // op_name
                                              {                   // input info
                                               // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                               {{{1280, 345}, {1280, 345}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1280, 567}, {1280, 567}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              // output info
                                              {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                  {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                  {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                              },
                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    int groupListTypeMode = 2;
    int groupListTypeIndex = 3;
    string expectTilingData = "";
    for (int i = 0; i < groupListTypeMode; i++) {
        expectTilingData = "2 128 2 " + to_string(i) + " 1481763717121 5497558139447 1481763718400 5497558139136 "
        "1099511627904 34359738432 4294967304 1 0 1970324836974592 131072 4294967297 4294967297 17179869188 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 ";
        tilingContextPara.attrs_.at(groupListTypeIndex).attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(i);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    }
}
