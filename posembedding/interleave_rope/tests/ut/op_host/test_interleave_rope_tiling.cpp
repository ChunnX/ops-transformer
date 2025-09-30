/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_interleave_rope_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/interleave_rope_tiling.h"

using namespace std;

class InterleaveRopeTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "InterleaveRopeTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InterleaveRopeTiling TearDown" << std::endl;
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

TEST_F(InterleaveRopeTiling, interleave_rope_tiling_succ_01) {
    optiling::InterleaveRopeCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("InterleaveRope",
                                              {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{32, 1, 1, 64}, {32, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {
                                                {{{32, 32, 1, 64}, {32, 32, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                              &compileInfo);
    uint64_t expectTilingKey = 1000;
    string expectTilingData = "8 0 32 32 1 64 4 4 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
