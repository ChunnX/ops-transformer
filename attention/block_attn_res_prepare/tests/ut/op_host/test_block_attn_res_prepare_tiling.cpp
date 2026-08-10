/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../op_host/block_attn_res_prepare_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t ASCEND_950_UB_SIZE = 253952UL;
constexpr uint64_t ASCEND_950_AIV_CORE_NUM = 64UL;

gert::TilingContextPara BuildCase(optiling::BlockAttnResPrepareCompileInfo &compileInfo,
                                  const gert::StorageShape &vShape, const gert::StorageShape &qShape,
                                  ge::DataType vType = ge::DT_FLOAT, ge::DataType qType = ge::DT_FLOAT,
                                  ge::DataType validType = ge::DT_INT32, float epsilon = 1e-6F)
{
    const int64_t totalT = vShape.GetOriginShape().GetDim(0);
    const int64_t totalD = vShape.GetOriginShape().GetDim(2);
    const int64_t totalS = qShape.GetOriginShape().GetDim(0);
    return gert::TilingContextPara(
        "BlockAttnResPrepare",
        {
            {vShape, vType, ge::FORMAT_ND},
            {qShape, qType, ge::FORMAT_ND},
            {{{1}, {1}}, validType, ge::FORMAT_ND},
        },
        {
            {{{totalS, totalT}, {totalS, totalT}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{totalS, totalT, totalD}, {totalS, totalT, totalD}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{totalS, totalT}, {totalS, totalT}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)}}, &compileInfo, "Ascend950",
        ASCEND_950_AIV_CORE_NUM, ASCEND_950_UB_SIZE, 4096);
}

optiling::BlockAttnResPrepareCompileInfo Ascend950CompileInfo()
{
    return {ASCEND_950_AIV_CORE_NUM, ASCEND_950_UB_SIZE, SYSTEM_WORKSPACE_SIZE};
}

} // namespace

class BlockAttnResPrepareTilingTest : public testing::Test {};

TEST_F(BlockAttnResPrepareTilingTest, NullTilingContextIsRejected)
{
    EXPECT_EQ(optiling::TilingBlockAttnResPrepare(nullptr), ge::GRAPH_FAILED);
}

TEST_F(BlockAttnResPrepareTilingTest, NullTilingParseContextIsRejected)
{
    EXPECT_EQ(optiling::TilingPrepareBlockAttnResPrepare(nullptr), ge::GRAPH_FAILED);
}

TEST_F(BlockAttnResPrepareTilingTest, NetworkDecodeShapeSucceeds)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 8, 7168}, {1, 8, 7168}}, {{18, 7168}, {18, 7168}});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::BlockAttnResPrepareTilingData));
    const auto *tilingData = reinterpret_cast<const optiling::BlockAttnResPrepareTilingData *>(
        tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->baseD, 7168U);
    EXPECT_EQ(tilingData->qBufferNum, 1U);
    EXPECT_EQ(tilingData->vBufferNum, 2U);
    EXPECT_EQ(tilingData->oBufferNum, 1U);
    EXPECT_EQ(tilingData->vCacheRows, 4U);
}

TEST_F(BlockAttnResPrepareTilingTest, SingleStaticVLoopUsesOneBufferForAllQueues)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 1, 7168}, {1, 1, 7168}}, {{18, 7168}, {18, 7168}});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::BlockAttnResPrepareTilingData));
    const auto *tilingData = reinterpret_cast<const optiling::BlockAttnResPrepareTilingData *>(
        tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->qBufferNum, 1U);
    EXPECT_EQ(tilingData->vBufferNum, 1U);
    EXPECT_EQ(tilingData->oBufferNum, 1U);
    EXPECT_EQ(tilingData->vCacheRows, 1U);
}

TEST_F(BlockAttnResPrepareTilingTest, QUsesDoubleBufferWhenEachCoreHasMultipleWorkRounds)
{
    auto compileInfo = Ascend950CompileInfo();
    // totalWorkUnits=8*16=128, usedCoreNum=64, so each core handles two rounds.
    auto context = BuildCase(compileInfo, {{8, 1, 7168}, {8, 1, 7168}}, {{16, 7168}, {16, 7168}});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::BlockAttnResPrepareTilingData));
    const auto *tilingData = reinterpret_cast<const optiling::BlockAttnResPrepareTilingData *>(
        tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->qBufferNum, 2U);
    EXPECT_EQ(tilingData->vBufferNum, 1U);
    EXPECT_EQ(tilingData->oBufferNum, 1U);
}

TEST_F(BlockAttnResPrepareTilingTest, NetworkPrefillShapeSucceeds)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{254, 8, 7168}, {254, 8, 7168}}, {{24, 7168}, {24, 7168}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, optiling::BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY);
}

TEST_F(BlockAttnResPrepareTilingTest, CubeTilingUsesAlignedWorkspacePitch)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{254, 8, 7168}, {254, 8, 7168}}, {{24, 7168}, {24, 7168}});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, optiling::BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::BlockAttnResPrepareCubeTilingData));
    const auto *tilingData = reinterpret_cast<const optiling::BlockAttnResPrepareCubeTilingData *>(
        tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->baseS, 24U);
    EXPECT_EQ(tilingData->nAlign, 16U);
    EXPECT_EQ(tilingData->workspacePerCoreElems, static_cast<uint64_t>(tilingData->sAlign) * tilingData->nAlign);
}

TEST_F(BlockAttnResPrepareTilingTest, MoreThanOneVRegOfBlocksIsRejected)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 65, 7168}, {1, 65, 7168}}, {{18, 7168}, {18, 7168}});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(BlockAttnResPrepareTilingTest, MismatchedDIsRejected)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 8, 7168}, {1, 8, 7168}}, {{18, 4096}, {18, 4096}});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(BlockAttnResPrepareTilingTest, NonPositiveEpsilonIsRejected)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 8, 7168}, {1, 8, 7168}}, {{18, 7168}, {18, 7168}},
                             ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, 0.0F);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(BlockAttnResPrepareTilingTest, InvalidDtypeIsRejected)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 8, 7168}, {1, 8, 7168}}, {{18, 7168}, {18, 7168}},
                             ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
