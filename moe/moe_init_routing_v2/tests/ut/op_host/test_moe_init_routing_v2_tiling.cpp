/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeInitRoutingV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingV2Tiling TearDown" << std::endl;
  }
};

// 单核+drop
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_01)
{
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{8, 6, 30}, {8, 6, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                                             },
                                             &compileInfo);
  uint64_t expectTilingKey = 10011;
  string expectTilingData = "64 8 30 6 6 8 1 0 1 1 48 1 48 48 48 1 48 48 8192 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
  std::vector<size_t> expectWorkspaces = {16779264};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+非drop
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_perf)
{
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{48, 30}, {48, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                             },
                                             &compileInfo);
  uint64_t expectTilingKey = 20000;
  string expectTilingData = "64 8 30 6 6 8 0 0 0 1 48 1 48 48 48 1 48 48 8192 0 2040 48 0 1 1 1 1 1 1 0 0 0 0 0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
  std::vector<size_t> expectWorkspaces = {16779264};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核++dropless  11000
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_one_core_dropless) {
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{80, 3000}, {80, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{80, 60}, {80, 60}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8, 3000}, {8, 3000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{4800, 3000}, {4800, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4800}, {4800}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                             },
                                             &compileInfo);

  uint64_t expectTilingKey = 20000;
  string expectTilingData = "64 80 3000 60 6 8 0 1 0 1 4800 1 4800 4800 4800 1 4800 4800 8192 0 2040 64 0 75 75 75 75 75 75 0 0 0 0 0 64 0 75 75 75 75 75 75 1 1 3000 3000 1 64 4800 75 75 75 75 75 75 1 1 3000 3000 1 ";
  std::vector<size_t> expectWorkspaces = {16931328};

  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// // 多核+静态quant+drop  10110
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_muticore_drop) {
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{32, 200, 3000}, {32, 200, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                                             },
                                             &compileInfo);

  uint64_t expectTilingKey = 10012;
  string expectTilingData = "64 320 3000 56 200 32 1 0 1 4 4480 1 4480 4480 4480 1 4480 4480 8192 0 2040 64 0 280 280 280 280 280 280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
  std::vector<size_t> expectWorkspaces = {17351168};

  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// // 多核+静态quant+dropless  10110
TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_muticore_dropless) {
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{17920, 3000}, {17920, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                                             },
                                             &compileInfo);

  uint64_t expectTilingKey = 10002;
  string expectTilingData = "64 320 3000 56 200 32 0 0 0 4 4480 1 4480 4480 4480 1 4480 4480 8192 0 2040 64 0 280 280 280 280 280 280 0 0 0 0 0 64 0 280 280 280 280 280 280 1 1 3000 3000 1 64 17920 280 280 280 280 280 280 1 1 3000 3000 1 ";
  std::vector<size_t> expectWorkspaces = {17351168};

  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_error_expert_id) {
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{320, 3000}, {320, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{17920, 3000}, {17920, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                             },
                                             &compileInfo);

  // 对于错误用例，不需要检查 tilingKey 和 tilingData
  ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeInitRoutingV2Tiling, moe_init_routing_v2_tiling_error_token) {
  optiling::MoeInitRoutingV2CompileInfo compileInfo = {64, 262144};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingV2",
                                             {
                                               {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{320, 56}, {320, 56}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {{{17920, 3000}, {17920, 3000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{17920}, {17920}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                             },
                                             {
                                               {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
                                               {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                               {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                             },
                                             &compileInfo);

  // 对于错误用例，不需要检查 tilingKey 和 tilingData
  ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
