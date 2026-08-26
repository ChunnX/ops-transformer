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

#include "../../../../op_host/checkers/liv2_checker.h"

namespace {
using optiling::indexer_v2_checker::FinalizeCommonInfo;
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::LIV2Checker;

class LightningIndexerV2Arch22CheckerTest : public testing::Test {
protected:
    void SetUp() override
    {
        queryDesc.SetDataType(ge::DT_BF16);
        keyDesc.SetDataType(ge::DT_BF16);
        weightsDesc.SetDataType(ge::DT_FLOAT);
        indicesDesc.SetDataType(ge::DT_INT32);
        valuesDesc.SetDataType(ge::DT_FLOAT);

        info.opName = "LightningIndexerV2";
        info.arch35 = false;
        info.layoutQText = "BSND";
        info.layoutKText = "BSND";
        info.topk = 2048;
        info.maxSeqlenQ = -1;
        info.maskMode = 0;
        info.cmpRatio = 1;
        info.returnValue = 0;
        info.query = {&queryDesc, &queryShape, nullptr, nullptr};
        info.key = {&keyDesc, &keyShape, nullptr, nullptr};
        info.weights = {&weightsDesc, &weightsShape, nullptr, nullptr};
        info.sparseIndices = {&indicesDesc, &indicesShape, nullptr, nullptr};
        info.sparseValues = {&valuesDesc, &valuesShape, nullptr, nullptr};
        FinalizeCommonInfo(info);
    }

    void SetTopk(int64_t topk)
    {
        info.topk = topk;
        indicesShape = gert::StorageShape{{2, 4, 1, topk}, {2, 4, 1, topk}};
        info.sparseIndices.storageShape = &indicesShape;
    }

    gert::CompileTimeTensorDesc queryDesc;
    gert::CompileTimeTensorDesc keyDesc;
    gert::CompileTimeTensorDesc weightsDesc;
    gert::CompileTimeTensorDesc indicesDesc;
    gert::CompileTimeTensorDesc valuesDesc;
    gert::StorageShape queryShape = {{2, 4, 64, 128}, {2, 4, 64, 128}};
    gert::StorageShape keyShape = {{2, 16, 1, 128}, {2, 16, 1, 128}};
    gert::StorageShape weightsShape = {{2, 4, 64}, {2, 4, 64}};
    gert::StorageShape indicesShape = {{2, 4, 1, 2048}, {2, 4, 1, 2048}};
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
    SetTopk(3072);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
    SetTopk(8192);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, RejectsUnsupportedLargeTopkValues)
{
    SetTopk(2049);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
    SetTopk(3000);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
    SetTopk(8193);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, KeepsHistoricalA2A3AttributeBehavior)
{
    info.maxSeqlenQ = -2;
    info.returnValue = 2;
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(LightningIndexerV2Arch22CheckerTest, KeepsHistoricalA2A3HeadGroupBehavior)
{
    queryShape = gert::StorageShape{{2, 4, 128, 128}, {2, 4, 128, 128}};
    weightsShape = gert::StorageShape{{2, 4, 128}, {2, 4, 128}};
    info.query.storageShape = &queryShape;
    info.weights.storageShape = &weightsShape;
    FinalizeCommonInfo(info);
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
