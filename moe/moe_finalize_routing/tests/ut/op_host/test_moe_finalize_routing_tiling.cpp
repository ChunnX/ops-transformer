/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_finalize_routing_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeFinalizeRoutingTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingTiling TearDown" << std::endl;
    }
};

// ----------------------------------------------------------------------------------------------------------

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20015;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}