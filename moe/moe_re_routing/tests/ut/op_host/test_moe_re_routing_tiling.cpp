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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/moe_re_routing_tiling.h"

using namespace std;
using namespace ge;

class MoeReRoutingTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeReRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeReRoutingTiling TearDown" << std::endl;
    }
};

std::map<std::string, std::string> short_soc_version = {{"Short_SoC_version", "Ascend910_95"}};
std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};

TEST_F(MoeReRoutingTiling, moe_re_routing_tiling_000) {
    optiling::MoeReRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeReRouting",
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"expert_token_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 210100;
    string expectTilingData = "7168 1 16 16 16 1 4063 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}