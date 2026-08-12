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
constexpr uint64_t ASCEND_950_L1_SIZE = 512UL * 1024UL;
constexpr uint64_t ASCEND_950_L0A_SIZE = 64UL * 1024UL;
constexpr uint64_t ASCEND_950_L0B_SIZE = 64UL * 1024UL;
constexpr uint64_t ASCEND_950_L0C_SIZE = 256UL * 1024UL;
constexpr uint64_t ASCEND_950_AIC_CORE_NUM = 32UL;
constexpr uint64_t ASCEND_950_AIV_CORE_NUM = 64UL;

// TilingInfo stores the serialized kernel ABI, not the host tiling class. The
// host class derives from TilingDataBase and therefore cannot be reinterpreted
// directly from this buffer.
struct VectorTilingDataView {
    uint32_t totalT;
    uint32_t totalN;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint64_t totalD;
    uint32_t usedCoreNum;
    uint32_t bigCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t baseD;
    uint8_t vCacheRows;
    uint8_t qBufferNum;
    uint8_t vBufferNum;
    uint8_t oBufferNum;
    uint32_t statUbElems;
    uint32_t workUbElems;
    float epsilon;
};

struct CubeTilingDataView {
    uint32_t totalT;
    uint32_t totalN;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint64_t totalD;
    uint32_t usedCoreNum;
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t bigCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t baseT;
    uint32_t baseS;
    uint32_t baseD;
    uint32_t baseDAlign;
    uint32_t sTileNum;
    uint32_t dTileNum;
    uint32_t sAlign;
    uint32_t nAlign;
    uint32_t dAlign;
    uint8_t qL1BufferNum;
    uint8_t vL1BufferNum;
    uint8_t vUbBufferNum;
    uint32_t l1CacheMode;
    uint8_t vL1CacheRows;
    uint64_t qL1Elems;
    uint64_t vL1Elems;
    uint64_t eL1Elems;
    uint64_t vUbElems;
    uint64_t dotUbElems;
    uint64_t reduceUbElems;
    uint64_t softmaxUbElems;
    uint64_t workspacePerCoreElems;
    float epsilon;
};

static_assert(sizeof(VectorTilingDataView) == 64U);
static_assert(sizeof(CubeTilingDataView) == 168U);

gert::TilingContextPara BuildCase(optiling::BlockAttnResPrepareCompileInfo &compileInfo,
                                  const gert::StorageShape &vShape, const gert::StorageShape &qShape,
                                  ge::DataType vType = ge::DT_FLOAT, ge::DataType qType = ge::DT_FLOAT,
                                  ge::DataType validType = ge::DT_INT32, float epsilon = 1e-6F,
                                  uint64_t coreNum = ASCEND_950_AIV_CORE_NUM)
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
        coreNum, ASCEND_950_UB_SIZE, 4096);
}

optiling::BlockAttnResPrepareCompileInfo Ascend950CompileInfo()
{
    return {ASCEND_950_AIC_CORE_NUM, ASCEND_950_AIV_CORE_NUM, ASCEND_950_UB_SIZE, ASCEND_950_L1_SIZE,
            ASCEND_950_L0A_SIZE, ASCEND_950_L0B_SIZE, ASCEND_950_L0C_SIZE, SYSTEM_WORKSPACE_SIZE};
}

} // namespace

class BlockAttnResPrepareTilingTest : public testing::Test {};

TEST_F(BlockAttnResPrepareTilingTest, NetworkDecodeShapeSucceeds)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{1, 8, 7168}, {1, 8, 7168}}, {{18, 7168}, {18, 7168}});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(VectorTilingDataView));
    const auto *tilingData = reinterpret_cast<const VectorTilingDataView *>(tilingInfo.tilingData.get());
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
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(VectorTilingDataView));
    const auto *tilingData = reinterpret_cast<const VectorTilingDataView *>(tilingInfo.tilingData.get());
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
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(VectorTilingDataView));
    const auto *tilingData = reinterpret_cast<const VectorTilingDataView *>(tilingInfo.tilingData.get());
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

TEST_F(BlockAttnResPrepareTilingTest, CubeTilingAllocatesMm1AndMm2Workspace)
{
    auto compileInfo = Ascend950CompileInfo();
    auto context = BuildCase(compileInfo, {{254, 8, 7168}, {254, 8, 7168}}, {{24, 7168}, {24, 7168}});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, optiling::BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CubeTilingDataView));
    const auto *tilingData = reinterpret_cast<const CubeTilingDataView *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->baseS, 24U);
    EXPECT_EQ(tilingData->nAlign, 16U);
    const uint64_t expectedWorkspaceElems = static_cast<uint64_t>(tilingData->baseS) * tilingData->totalN +
                                            static_cast<uint64_t>(tilingData->sAlign) *
                                                (tilingData->nAlign + tilingData->baseDAlign);
    EXPECT_EQ(tilingData->workspacePerCoreElems, expectedWorkspaceElems);
}

TEST_F(BlockAttnResPrepareTilingTest, CubeTilingFallsBackWhenTwoAivPeersAreUnavailable)
{
    auto compileInfo = Ascend950CompileInfo();
    compileInfo.aivCoreNum = 1U;
    auto context = BuildCase(compileInfo, {{32, 8, 7168}, {32, 8, 7168}}, {{24, 7168}, {24, 7168}},
                             ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, 1e-6F, 1U);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY);
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
