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

#include "../../../../../lightning_indexer_v2/op_host/checkers/checker_adapter.h"
#include "../../../../op_host/checkers/qliv2_checker.h"
#include "../../../../op_host/checkers/quant_checker_qliv2.h"

namespace {
using optiling::indexer_v2_checker::CheckerLayout;
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::PopulateDerivedInfo;
using optiling::indexer_v2_checker::QLIV2Checker;
using optiling::indexer_v2_checker::QuantChecker;

constexpr char OP_NAME[] = "QuantLightningIndexerV2";
constexpr char BSND_LAYOUT[] = "BSND";
constexpr char PAGED_ATTENTION_LAYOUT[] = "PA_BBND";
constexpr int64_t BATCH_SIZE = 2;
constexpr int64_t QUERY_SEQUENCE_LENGTH = 4;
constexpr int64_t INVALID_QUERY_SEQUENCE_LENGTH = 5;
constexpr int64_t KEY_SEQUENCE_LENGTH = 16;
constexpr int64_t QUERY_HEAD_NUM = 8;
constexpr int64_t KEY_HEAD_NUM = 1;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t TOPK = 64;
constexpr int64_t METADATA_SIZE = 1024;
constexpr int64_t INT8_QUANT_MODE = 2;
constexpr int64_t FP8_QUANT_MODE = 3;
constexpr int64_t HIFLOAT8_QUANT_MODE = 4;
constexpr int64_t MXFP4_QUANT_MODE = 5;
constexpr int64_t UNSUPPORTED_QUANT_MODE = 6;
constexpr int64_t MX_SCALE_GROUP_NUM = 2;
constexpr int64_t MX_SCALE_PACK_SIZE = 2;
constexpr int64_t NON_SCALAR_SCALE_SIZE = 2;
constexpr int64_t ODD_KEY_LEADING_STRIDE = 4095;
constexpr int64_t EVEN_KEY_LEADING_STRIDE = 4096;
constexpr int64_t ODD_SCALE_LEADING_STRIDE = 127;
constexpr int64_t SCALE_SEQUENCE_STRIDE = 4;
constexpr int64_t SCALE_PACK_STRIDE = 2;

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

        info.opName = OP_NAME;
        info.quantized = true;
        info.arch35 = true;
        info.layoutQText = BSND_LAYOUT;
        info.layoutKText = BSND_LAYOUT;
        info.topk = TOPK;
        info.quantMode = INT8_QUANT_MODE;
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
        PopulateDerivedInfo(info);
    }

    gert::CompileTimeTensorDesc queryDesc;
    gert::CompileTimeTensorDesc keyDesc;
    gert::CompileTimeTensorDesc weightsDesc;
    gert::CompileTimeTensorDesc queryDescaleDesc;
    gert::CompileTimeTensorDesc keyDescaleDesc;
    gert::CompileTimeTensorDesc indicesDesc;
    gert::CompileTimeTensorDesc valuesDesc;
    gert::CompileTimeTensorDesc metadataDesc;
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape weightsShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape queryDescaleShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                            {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape keyDescaleShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM},
                                          {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM}};
    gert::StorageShape indicesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK}};
    gert::StorageShape valuesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK},
                                      {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK}};
    gert::StorageShape metadataShape = {{METADATA_SIZE}, {METADATA_SIZE}};
    IndexerV2CheckerInfo info;
};
} // namespace

TEST(QuantLightningIndexerV2Checker, RejectsMissingDescaleInputs)
{
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;

    QuantChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST(QuantLightningIndexerV2Checker, AcceptsInt8ScaleShapes)
{
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape queryDescaleShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                            {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape keyDescaleShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM},
                                          {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM}};

    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.quantMode = INT8_QUANT_MODE;
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
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape queryDescaleShape = {
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    gert::StorageShape keyDescaleShape = {
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};

    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.quantMode = MXFP4_QUANT_MODE;
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
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape queryDescaleShape = {
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, KEY_HEAD_NUM},
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, KEY_HEAD_NUM}};
    gert::StorageShape keyDescaleShape = {
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};

    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.quantMode = MXFP4_QUANT_MODE;
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
    gert::StorageShape queryMxScaleShape = {
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    gert::StorageShape keyMxScaleShape = {
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    queryDesc.SetDataType(ge::DT_FLOAT8_E4M3FN);
    keyDesc.SetDataType(ge::DT_FLOAT8_E4M3FN);
    weightsDesc.SetDataType(ge::DT_FLOAT);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    keyDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    info.queryDescale.storageShape = &queryMxScaleShape;
    info.keyDescale.storageShape = &keyMxScaleShape;
    info.quantMode = FP8_QUANT_MODE;
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
    info.quantMode = HIFLOAT8_QUANT_MODE;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, AcceptsMxfp4ModeThroughWrapper)
{
    gert::StorageShape queryMxScaleShape = {
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    gert::StorageShape keyMxScaleShape = {
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    queryDesc.SetDataType(ge::DT_FLOAT4_E2M1);
    keyDesc.SetDataType(ge::DT_FLOAT4_E2M1);
    weightsDesc.SetDataType(ge::DT_FLOAT);
    queryDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    keyDescaleDesc.SetDataType(ge::DT_FLOAT8_E8M0);
    info.queryDescale.storageShape = &queryMxScaleShape;
    info.keyDescale.storageShape = &keyMxScaleShape;
    info.quantMode = MXFP4_QUANT_MODE;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidQliCheckerInfoTest, RejectsUnsupportedQuantMode)
{
    info.quantMode = 0;
    EXPECT_EQ(QuantChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.quantMode = UNSUPPORTED_QUANT_MODE;
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
    gert::StorageShape wrongQueryScaleShape = {{BATCH_SIZE, INVALID_QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                               {BATCH_SIZE, INVALID_QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    info.queryDescale.storageShape = &wrongQueryScaleShape;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsNonScalarHifloat8Scale)
{
    gert::StorageShape wrongScaleShape = {{NON_SCALAR_SCALE_SIZE}, {NON_SCALAR_SCALE_SIZE}};
    info.quantMode = HIFLOAT8_QUANT_MODE;
    info.queryDescale.storageShape = &wrongScaleShape;
    info.keyDescale.storageShape = &wrongScaleShape;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidQliCheckerInfoTest, RejectsOddMxfp4PagedAttentionStrides)
{
    gert::StorageShape queryMxScaleShape = {
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    gert::StorageShape keyMxScaleShape = {
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE},
        {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, MX_SCALE_GROUP_NUM, MX_SCALE_PACK_SIZE}};
    gert::Stride oddKeyStride = {ODD_KEY_LEADING_STRIDE, HEAD_DIM, HEAD_DIM, 1};
    gert::Stride oddScaleStride = {ODD_SCALE_LEADING_STRIDE, SCALE_SEQUENCE_STRIDE, SCALE_SEQUENCE_STRIDE,
                                   SCALE_PACK_STRIDE, 1};
    info.layoutK = CheckerLayout::PA_BBND;
    info.quantMode = MXFP4_QUANT_MODE;
    info.queryDescale.storageShape = &queryMxScaleShape;
    info.keyDescale.storageShape = &keyMxScaleShape;
    info.key.stride = &oddKeyStride;
    info.keyDescale.stride = &oddScaleStride;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    gert::Stride evenKeyStride = {EVEN_KEY_LEADING_STRIDE, HEAD_DIM, HEAD_DIM, 1};
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
    gert::StorageShape wrongRankScaleShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH}, {BATCH_SIZE, QUERY_SEQUENCE_LENGTH}};
    info.queryDescale.storageShape = &wrongRankScaleShape;
    EXPECT_EQ(QuantChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}
