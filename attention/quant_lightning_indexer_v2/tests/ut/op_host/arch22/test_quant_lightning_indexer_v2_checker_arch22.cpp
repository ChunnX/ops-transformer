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

#include "../../../../op_host/checkers/qliv2_checker.h"

namespace {
using optiling::indexer_v2_checker::FinalizeCommonInfo;
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::QLIV2Checker;

class QuantLightningIndexerV2Arch22CheckerTest : public testing::Test {
protected:
    void SetUp() override
    {
        queryDesc.SetDataType(ge::DT_INT8);
        keyDesc.SetDataType(ge::DT_INT8);
        weightsDesc.SetDataType(ge::DT_FLOAT16);
        queryDescaleDesc.SetDataType(ge::DT_FLOAT16);
        keyDescaleDesc.SetDataType(ge::DT_FLOAT16);
        indicesDesc.SetDataType(ge::DT_INT32);
        valuesDesc.SetDataType(ge::DT_BF16);
        blockTableDesc.SetDataType(ge::DT_INT32);
        sequsedKDesc.SetDataType(ge::DT_INT32);
        metadataDesc.SetDataType(ge::DT_UINT8);

        info.opName = "QuantLightningIndexerV2";
        info.quantized = true;
        info.arch35 = false;
        info.layoutQText = "BSND";
        info.layoutKText = "PA_BBND";
        info.topk = 2048;
        info.quantMode = 2;
        info.maxSeqlenQ = -1;
        info.maskMode = 0;
        info.cmpRatio = 1;
        info.returnValue = 0;
        info.query = {&queryDesc, &queryShape, nullptr, nullptr};
        info.key = {&keyDesc, &keyShape, nullptr, &keyStride};
        info.weights = {&weightsDesc, &weightsShape, nullptr, nullptr};
        info.queryDescale = {&queryDescaleDesc, &queryDescaleShape, nullptr, nullptr};
        info.keyDescale = {&keyDescaleDesc, &keyDescaleShape, nullptr, &keyDescaleStride};
        info.blockTable = {&blockTableDesc, &blockTableShape, nullptr, nullptr};
        info.sequsedK = {&sequsedKDesc, &sequsedKShape, nullptr, nullptr};
        info.sparseIndices = {&indicesDesc, &indicesShape, nullptr, nullptr};
        info.sparseValues = {&valuesDesc, &valuesShape, nullptr, nullptr};
        info.metadata = {&metadataDesc, &metadataShape, nullptr, nullptr};
        FinalizeCommonInfo(info);
    }

    gert::CompileTimeTensorDesc queryDesc;
    gert::CompileTimeTensorDesc keyDesc;
    gert::CompileTimeTensorDesc weightsDesc;
    gert::CompileTimeTensorDesc queryDescaleDesc;
    gert::CompileTimeTensorDesc keyDescaleDesc;
    gert::CompileTimeTensorDesc indicesDesc;
    gert::CompileTimeTensorDesc valuesDesc;
    gert::CompileTimeTensorDesc blockTableDesc;
    gert::CompileTimeTensorDesc sequsedKDesc;
    gert::CompileTimeTensorDesc metadataDesc;
    gert::StorageShape queryShape = {{2, 4, 64, 128}, {2, 4, 64, 128}};
    gert::StorageShape keyShape = {{4, 16, 1, 128}, {4, 16, 1, 128}};
    gert::StorageShape weightsShape = {{2, 4, 64}, {2, 4, 64}};
    gert::StorageShape queryDescaleShape = {{2, 4, 64}, {2, 4, 64}};
    gert::StorageShape keyDescaleShape = {{4, 16, 1}, {4, 16, 1}};
    gert::StorageShape blockTableShape = {{2, 4}, {2, 4}};
    gert::StorageShape sequsedKShape = {{2}, {2}};
    gert::StorageShape indicesShape = {{2, 4, 1, 2048}, {2, 4, 1, 2048}};
    gert::StorageShape valuesShape = {{0}, {0}};
    gert::StorageShape metadataShape = {{1024}, {1024}};
    gert::Stride keyStride = {2048, 128, 128, 1};
    gert::Stride keyDescaleStride = {16, 1, 1};
    IndexerV2CheckerInfo info;
};
} // namespace

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, AcceptsBaselinePagedAttentionInput)
{
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, AcceptsTndPagedAttentionInput)
{
    gert::CompileTimeTensorDesc cuSeqlensQDesc;
    gert::StorageShape tndQueryShape = {{8, 64, 128}, {8, 64, 128}};
    gert::StorageShape tndWeightsShape = {{8, 64}, {8, 64}};
    gert::StorageShape tndQueryDescaleShape = {{8, 64}, {8, 64}};
    gert::StorageShape tndIndicesShape = {{8, 1, 2048}, {8, 1, 2048}};
    gert::StorageShape cuSeqlensQShape = {{3}, {3}};
    cuSeqlensQDesc.SetDataType(ge::DT_INT32);
    info.layoutQText = "TND";
    info.query.storageShape = &tndQueryShape;
    info.weights.storageShape = &tndWeightsShape;
    info.queryDescale.storageShape = &tndQueryDescaleShape;
    info.sparseIndices.storageShape = &tndIndicesShape;
    info.cuSeqlensQ = {&cuSeqlensQDesc, &cuSeqlensQShape, nullptr, nullptr};
    FinalizeCommonInfo(info);
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsNonPagedAttentionLayout)
{
    info.layoutKText = "BSND";
    info.layoutK = optiling::indexer_v2_checker::CheckerLayout::BSND;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsUnsupportedQuantModes)
{
    for (const int64_t quantMode : {1, 3, 4, 5}) {
        info.quantMode = quantMode;
        EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
    }
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsTopkAbove2048)
{
    info.topk = 2049;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsNonPowerOfTwoCompressionRatio)
{
    info.cmpRatio = 3;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsReturningSparseValues)
{
    info.returnValue = 1;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsHeadGroupOtherThan64)
{
    info.qHeads = 32;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsMissingMetadata)
{
    info.metadata = {};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsCuSeqlensKForPagedAttention)
{
    gert::CompileTimeTensorDesc cuSeqlensKDesc;
    gert::StorageShape cuSeqlensKShape = {{2}, {2}};
    cuSeqlensKDesc.SetDataType(ge::DT_INT32);
    info.cuSeqlensK = {&cuSeqlensKDesc, &cuSeqlensKShape, nullptr, nullptr};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, ChecksCompressionResidualPresence)
{
    info.maskMode = 3;
    info.cmpRatio = 2;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);

    gert::CompileTimeTensorDesc residualDesc;
    gert::StorageShape residualShape = {{2}, {2}};
    residualDesc.SetDataType(ge::DT_INT32);
    info.cmpResidualK = {&residualDesc, &residualShape, nullptr, nullptr};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsWrongOptionalSequenceSize)
{
    gert::CompileTimeTensorDesc sequsedQDesc;
    gert::StorageShape sequsedQShape = {{1}, {1}};
    sequsedQDesc.SetDataType(ge::DT_INT32);
    info.sequsedQ = {&sequsedQDesc, &sequsedQShape, nullptr, nullptr};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, PreservesOutputOffsetDtypeBehavior)
{
    gert::CompileTimeTensorDesc offsetDesc;
    gert::StorageShape offsetShape = {{2, 4, 64}, {2, 4, 64}};
    offsetDesc.SetDataType(ge::DT_FLOAT);
    info.outputIdxOffset = {&offsetDesc, &offsetShape, nullptr, nullptr};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}
