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
 * \file test_lightning_indexer_v2_checker.cpp
 * \brief Tests Lightning Indexer V2 checker behavior on A2 and A3 platforms.
 */

#include <gtest/gtest.h>

#include "../../../../op_host/checkers/checker_adapter.h"
#include "../../../../op_host/checkers/liv2_checker.h"

namespace {
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::LIV2Checker;
using optiling::indexer_v2_checker::PopulateDerivedInfo;

constexpr char OP_NAME[] = "LightningIndexerV2";
constexpr char BSND_LAYOUT[] = "BSND";
constexpr int64_t BATCH_SIZE = 2;
constexpr int64_t QUERY_SEQUENCE_LENGTH = 4;
constexpr int64_t KEY_SEQUENCE_LENGTH = 16;
constexpr int64_t QUERY_HEAD_NUM = 64;
constexpr int64_t LARGE_QUERY_HEAD_NUM = 128;
constexpr int64_t KEY_HEAD_NUM = 1;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BASE_TOPK = 2048;
constexpr int64_t SUPPORTED_TOPK_ABOVE_2K = 3072;
constexpr int64_t TOPK_JUST_ABOVE_2K = 2049;
constexpr int64_t UNSUPPORTED_NON_ALIGNED_TOPK = 3000;
constexpr int64_t MAX_TOPK = 8192;
constexpr int64_t TOPK_ABOVE_MAX = 8193;
constexpr int64_t HISTORICAL_INVALID_MAX_SEQLEN_Q = -2;
constexpr int64_t HISTORICAL_RETURN_VALUE = 2;

class LightningIndexerV2Arch22CheckerTest : public testing::Test {
protected:
    void SetUp() override
    {
        queryDesc.SetDataType(ge::DT_BF16);
        keyDesc.SetDataType(ge::DT_BF16);
        weightsDesc.SetDataType(ge::DT_FLOAT);
        indicesDesc.SetDataType(ge::DT_INT32);
        valuesDesc.SetDataType(ge::DT_FLOAT);

        info.opName = OP_NAME;
        info.arch35 = false;
        info.layoutQText = BSND_LAYOUT;
        info.layoutKText = BSND_LAYOUT;
        info.topk = BASE_TOPK;
        info.maxSeqlenQ = -1;
        info.maskMode = 0;
        info.cmpRatio = 1;
        info.returnValue = 0;
        info.query = {&queryDesc, &queryShape, nullptr, nullptr};
        info.key = {&keyDesc, &keyShape, nullptr, nullptr};
        info.weights = {&weightsDesc, &weightsShape, nullptr, nullptr};
        info.sparseIndices = {&indicesDesc, &indicesShape, nullptr, nullptr};
        info.sparseValues = {&valuesDesc, &valuesShape, nullptr, nullptr};
        PopulateDerivedInfo(info);
    }

    void SetTopk(int64_t topk)
    {
        info.topk = topk;
        indicesShape = gert::StorageShape{{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, topk},
                                          {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, topk}};
        info.sparseIndices.storageShape = &indicesShape;
    }

    gert::CompileTimeTensorDesc queryDesc;
    gert::CompileTimeTensorDesc keyDesc;
    gert::CompileTimeTensorDesc weightsDesc;
    gert::CompileTimeTensorDesc indicesDesc;
    gert::CompileTimeTensorDesc valuesDesc;
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape weightsShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape indicesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, BASE_TOPK},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, BASE_TOPK}};
    gert::StorageShape valuesShape = {{0}, {0}};
    IndexerV2CheckerInfo info;
};
} // namespace

TEST_F(LightningIndexerV2Arch22CheckerTest, AcceptsBaselineBsndInput)
{
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, AcceptsSupportedLargeTopkBoundaries)
{
    SetTopk(SUPPORTED_TOPK_ABOVE_2K);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
    SetTopk(MAX_TOPK);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, RejectsUnsupportedLargeTopkValues)
{
    SetTopk(TOPK_JUST_ABOVE_2K);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
    SetTopk(UNSUPPORTED_NON_ALIGNED_TOPK);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
    SetTopk(TOPK_ABOVE_MAX);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, KeepsHistoricalA2A3AttributeBehavior)
{
    info.maxSeqlenQ = HISTORICAL_INVALID_MAX_SEQLEN_Q;
    info.returnValue = HISTORICAL_RETURN_VALUE;
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, KeepsHistoricalA2A3HeadGroupBehavior)
{
    queryShape = gert::StorageShape{{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, LARGE_QUERY_HEAD_NUM, HEAD_DIM},
                                    {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, LARGE_QUERY_HEAD_NUM, HEAD_DIM}};
    weightsShape = gert::StorageShape{{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, LARGE_QUERY_HEAD_NUM},
                                      {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, LARGE_QUERY_HEAD_NUM}};
    info.query.storageShape = &queryShape;
    info.weights.storageShape = &weightsShape;
    PopulateDerivedInfo(info);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, IgnoresUnsupportedOutputOffsetDetails)
{
    gert::CompileTimeTensorDesc offsetDesc;
    gert::StorageShape offsetShape = {{1}, {1}};
    offsetDesc.SetDataType(ge::DT_FLOAT);
    info.outputIdxOffset = {&offsetDesc, &offsetShape, nullptr, nullptr};
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}
