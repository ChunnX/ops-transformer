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
 * \file test_quant_lightning_indexer_v2_checker.cpp
 * \brief Tests common and A5 parameter checks for Quant Lightning Indexer V2.
 */

#include <gtest/gtest.h>

#include "../../../op_host/checkers/qliv2_checker.h"
#include "../../../op_host/checkers/quant_checker_qliv2.h"

namespace {
using optiling::indexer_v2_checker::CheckerLayout;
using optiling::indexer_v2_checker::FinalizeCommonInfo;
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::QLIV2Checker;
using optiling::indexer_v2_checker::QuantChecker;

class ValidQliCheckerInfoTest : public testing::Test {
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
        metadataDesc.SetDataType(ge::DT_UINT8);

        info.opName = "QuantLightningIndexerV2";
        info.quantized = true;
        info.arch35 = true;
        info.layoutQText = "BSND";
        info.layoutKText = "BSND";
        info.topk = 64;
        info.quantMode = 2;
        info.maxSeqlenQ = -1;
        info.maskMode = 0;
        info.cmpRatio = 1;
        info.returnValue = 1;
        info.query = {&queryDesc, &queryShape, nullptr, nullptr};
        info.key = {&keyDesc, &keyShape, nullptr, nullptr};
        info.weights = {&weightsDesc, &weightsShape, nullptr, nullptr};
        info.queryDescale = {&queryDescaleDesc, &queryDescaleShape, nullptr, nullptr};
        info.keyDescale = {&keyDescaleDesc, &keyDescaleShape, nullptr, nullptr};
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
    gert::CompileTimeTensorDesc metadataDesc;
    gert::StorageShape queryShape = {{2, 4, 8, 128}, {2, 4, 8, 128}};
    gert::StorageShape keyShape = {{2, 16, 1, 128}, {2, 16, 1, 128}};
    gert::StorageShape weightsShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape queryDescaleShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape keyDescaleShape = {{2, 16, 1}, {2, 16, 1}};
    gert::StorageShape indicesShape = {{2, 4, 1, 64}, {2, 4, 1, 64}};
    gert::StorageShape valuesShape = {{2, 4, 1, 64}, {2, 4, 1, 64}};
    gert::StorageShape metadataShape = {{1024}, {1024}};
    IndexerV2CheckerInfo info;
};
} // namespace

TEST(QuantLightningIndexerV2Checker, RejectsMissingDescaleInputs)
{
    IndexerV2CheckerInfo info;
    info.opName = "QuantLightningIndexerV2";

    QuantChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST(QuantLightningIndexerV2Checker, AcceptsInt8ScaleShapes)
{
    gert::StorageShape queryShape = {{2, 4, 8, 128}, {2, 4, 8, 128}};
    gert::StorageShape keyShape = {{2, 16, 1, 128}, {2, 16, 1, 128}};
    gert::StorageShape queryDescaleShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape keyDescaleShape = {{2, 16, 1}, {2, 16, 1}};

    IndexerV2CheckerInfo info;
    info.opName = "QuantLightningIndexerV2";
    info.quantMode = 2;
    info.layoutK = CheckerLayout::BSND;
    info.query.storageShape = &queryShape;
    info.key.storageShape = &keyShape;
    info.queryDescale.storageShape = &queryDescaleShape;
    info.keyDescale.storageShape = &keyDescaleShape;

    QuantChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
}

TEST(QuantLightningIndexerV2Checker, AcceptsMxfp4PackedScaleShapes)
{
    gert::StorageShape queryShape = {{2, 4, 8, 128}, {2, 4, 8, 128}};
    gert::StorageShape keyShape = {{2, 16, 1, 128}, {2, 16, 1, 128}};
    gert::StorageShape queryDescaleShape = {{2, 4, 8, 2, 2}, {2, 4, 8, 2, 2}};
    gert::StorageShape keyDescaleShape = {{2, 16, 1, 2, 2}, {2, 16, 1, 2, 2}};

    IndexerV2CheckerInfo info;
    info.opName = "QuantLightningIndexerV2";
    info.quantMode = 5;
    info.layoutK = CheckerLayout::BSND;
    info.query.storageShape = &queryShape;
    info.key.storageShape = &keyShape;
    info.queryDescale.storageShape = &queryDescaleShape;
    info.keyDescale.storageShape = &keyDescaleShape;

    QuantChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
}

TEST(QuantLightningIndexerV2Checker, RejectsInvalidMxfp4ScalePack)
{
    gert::StorageShape queryShape = {{2, 4, 8, 128}, {2, 4, 8, 128}};
    gert::StorageShape keyShape = {{2, 16, 1, 128}, {2, 16, 1, 128}};
    gert::StorageShape queryDescaleShape = {{2, 4, 8, 2, 1}, {2, 4, 8, 2, 1}};
    gert::StorageShape keyDescaleShape = {{2, 16, 1, 2, 2}, {2, 16, 1, 2, 2}};

    IndexerV2CheckerInfo info;
    info.opName = "QuantLightningIndexerV2";
    info.quantMode = 5;
    info.layoutK = CheckerLayout::BSND;
    info.query.storageShape = &queryShape;
    info.key.storageShape = &keyShape;
    info.queryDescale.storageShape = &queryDescaleShape;
    info.keyDescale.storageShape = &keyDescaleShape;

    QuantChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, AcceptsCompleteInt8InputThroughWrapper)
{
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, RejectsDescaleWithoutDescriptor)
{
    info.queryDescale.desc = nullptr;
    EXPECT_EQ(QuantChecker().CheckParaExistence(info), ge::GRAPH_FAILED);
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, AcceptsFp8Mode)
{
    queryDesc.SetDataType(ge::DT_FLOAT8_E4M3FN);
    keyDesc.SetDataType(ge::DT_FLOAT8_E4M3FN);
    weightsDesc.SetDataType(ge::DT_FLOAT);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT);
    keyDescaleDesc.SetDataType(ge::DT_FLOAT);
    info.quantMode = 1;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, AcceptsMxfp8Mode)
{
    gert::StorageShape queryMxScaleShape = {{2, 4, 8, 2, 2}, {2, 4, 8, 2, 2}};
    gert::StorageShape keyMxScaleShape = {{2, 16, 1, 2, 2}, {2, 16, 1, 2, 2}};
    queryDesc.SetDataType(ge::DT_FLOAT8_E4M3FN);
    keyDesc.SetDataType(ge::DT_FLOAT8_E4M3FN);
    weightsDesc.SetDataType(ge::DT_FLOAT);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    keyDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    info.queryDescale.storageShape = &queryMxScaleShape;
    info.keyDescale.storageShape = &keyMxScaleShape;
    info.quantMode = 3;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, AcceptsHifloat8ModeWithScalarScales)
{
    gert::StorageShape scalarScaleShape = {{1}, {1}};
    queryDesc.SetDataType(ge::DT_HIFLOAT8);
    keyDesc.SetDataType(ge::DT_HIFLOAT8);
    weightsDesc.SetDataType(ge::DT_FLOAT);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT);
    keyDescaleDesc.SetDataType(ge::DT_FLOAT);
    info.queryDescale.storageShape = &scalarScaleShape;
    info.keyDescale.storageShape = &scalarScaleShape;
    info.quantMode = 4;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, AcceptsMxfp4ModeThroughWrapper)
{
    gert::StorageShape queryMxScaleShape = {{2, 4, 8, 2, 2}, {2, 4, 8, 2, 2}};
    gert::StorageShape keyMxScaleShape = {{2, 16, 1, 2, 2}, {2, 16, 1, 2, 2}};
    queryDesc.SetDataType(ge::DT_FLOAT4_E2M1);
    keyDesc.SetDataType(ge::DT_FLOAT4_E2M1);
    weightsDesc.SetDataType(ge::DT_FLOAT);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    keyDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    info.queryDescale.storageShape = &queryMxScaleShape;
    info.keyDescale.storageShape = &keyMxScaleShape;
    info.quantMode = 5;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, RejectsUnsupportedQuantMode)
{
    info.quantMode = 0;
    EXPECT_EQ(QuantChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.quantMode = 6;
    EXPECT_EQ(QuantChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsWrongInputAndScaleDtypes)
{
    queryDesc.SetDataType(ge::DT_FLOAT16);
    EXPECT_EQ(QuantChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    queryDesc.SetDataType(ge::DT_INT8);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT);
    EXPECT_EQ(QuantChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsScalePrefixMismatch)
{
    gert::StorageShape wrongQueryScaleShape = {{2, 5, 8}, {2, 5, 8}};
    info.queryDescale.storageShape = &wrongQueryScaleShape;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsNonScalarHifloat8Scale)
{
    gert::StorageShape wrongScaleShape = {{2}, {2}};
    info.quantMode = 4;
    info.queryDescale.storageShape = &wrongScaleShape;
    info.keyDescale.storageShape = &wrongScaleShape;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsOddMxfp4PagedAttentionStrides)
{
    gert::StorageShape queryMxScaleShape = {{2, 4, 8, 2, 2}, {2, 4, 8, 2, 2}};
    gert::StorageShape keyMxScaleShape = {{2, 16, 1, 2, 2}, {2, 16, 1, 2, 2}};
    gert::Stride oddKeyStride = {4095, 128, 128, 1};
    gert::Stride oddScaleStride = {127, 4, 4, 2, 1};
    info.layoutK = CheckerLayout::PA_BBND;
    info.quantMode = 5;
    info.queryDescale.storageShape = &queryMxScaleShape;
    info.keyDescale.storageShape = &keyMxScaleShape;
    info.key.stride = &oddKeyStride;
    info.keyDescale.stride = &oddScaleStride;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    gert::Stride evenKeyStride = {4096, 128, 128, 1};
    info.key.stride = &evenKeyStride;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsMissingSparseValuesWhenReturningValues)
{
    info.sparseValues = {};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsNonPagedLayoutOnA2A3)
{
    info.arch35 = false;
    info.returnValue = 0;
    EXPECT_EQ(QuantChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsWrongScaleRank)
{
    gert::StorageShape wrongRankScaleShape = {{2, 4}, {2, 4}};
    info.queryDescale.storageShape = &wrongRankScaleShape;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}
