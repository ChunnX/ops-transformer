/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_tutel_combine_x_tiling.cpp
 * \brief Unit tests of the MoeTutelCombineX host tiling (arch35).
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_host/moe_tutel_combine_x_tiling.h"
#include "../../../op_kernel/arch35/moe_tutel_combine_x_tiling_struct.h"

using namespace std;

class MoeTutelCombineXTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelCombineXTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTutelCombineXTiling TearDown" << std::endl;
    }
};

// Normal no-batch: y_grad [4,2], gates/indices/locations [4], x_grad [4,2].
// Only relationship assertions are made on the produced tiling data so the
// case stays independent of the exact fabricated core/UB numbers.
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_normal_no_batch)
{
    optiling::MoeTutelCombineXCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTutelCombineX",
        {
            {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_NORMAL_NO_BATCH, "", {});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    ASSERT_NE(info.tilingData, nullptr);
    ASSERT_GE(info.tilingDataSize, sizeof(MoetutelcombinexTilingData));
    const auto *td = reinterpret_cast<const MoetutelcombinexTilingData *>(info.tilingData.get());
    EXPECT_EQ(td->batchSize, 1);
    EXPECT_EQ(td->samples, 4);
    EXPECT_EQ(td->hidden, 2);
    EXPECT_EQ(td->eC, 4);
    EXPECT_GE(td->usedCoreNum, 1);
    EXPECT_EQ(td->tokenTotal, td->batchSize * td->samples);
    EXPECT_EQ(td->hChunkCnt, (td->hidden + td->hChunk - 1) / td->hChunk);
    EXPECT_EQ(td->hChunkTail, td->hidden - (td->hChunkCnt - 1) * td->hChunk);
}

// Empty tensor (S=0): the empty shortcut path must be taken (key=1).
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_empty)
{
    optiling::MoeTutelCombineXCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTutelCombineX",
        {
            {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{0, 2}, {0, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_EMPTY, "", {});
}

// Normal batched: y_grad [2,4,2], gates/indices/locations [2,4].
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_normal_batch)
{
    optiling::MoeTutelCombineXCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTutelCombineX",
        {
            {{{2, 4, 2}, {2, 4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 4, 2}, {2, 4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_NORMAL_BATCH, "", {});
}
