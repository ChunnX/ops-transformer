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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/ring_attention_update_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace gert;
using namespace optiling;

class RingAttentionUpdateTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RingAttentionUpdateTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RingAttentionUpdateTiling TearDown" << std::endl;
  }
};

struct RingAttentionUpdateCompileInfo {};

TEST_F(RingAttentionUpdateTiling, test_ring_attention_update_success) {
    RingAttentionUpdateCompileInfo compileInfo = {};
    std::string input_layout = "SBH";
    
    gert::TilingContextPara tilingContextPara("RingAttentionUpdate",
                                              {{{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                               {gert::TilingContextPara::OpAttr("input_layout", Ops::Math::AnyValue::CreateFrom<string>(input_layout))},
                                                &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "2 12 1024 128 8 64 8 3 128 128 56 0 128 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}