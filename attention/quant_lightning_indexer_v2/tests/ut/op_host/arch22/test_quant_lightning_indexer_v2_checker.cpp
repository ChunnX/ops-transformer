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
 * \brief Tests Quant Lightning Indexer V2 checker behavior on A2 and A3 platforms.
 */

#include <array>
#include <gtest/gtest.h>

#include "../../../../../lightning_indexer_v2/op_host/checkers/checker_adapter.h"
#include "../../../../op_host/checkers/qliv2_checker.h"

namespace {
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::PopulateDerivedInfo;
using optiling::indexer_v2_checker::QLIV2Checker;

constexpr char OP_NAME[] = "QuantLightningIndexerV2";
constexpr char BSND_LAYOUT[] = "BSND";
constexpr char TND_LAYOUT[] = "TND";
constexpr char PAGED_ATTENTION_LAYOUT[] = "PA_BBND";
constexpr int64_t BATCH_SIZE = 2;
constexpr int64_t QUERY_SEQUENCE_LENGTH = 4;
constexpr int64_t KEY_BLOCK_NUM = 4;
constexpr int64_t KEY_BLOCK_SIZE = 16;
constexpr int64_t QUERY_HEAD_NUM = 64;
constexpr int64_t INVALID_QUERY_HEAD_NUM = 32;
constexpr int64_t KEY_HEAD_NUM = 1;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BASE_TOPK = 2048;
constexpr int64_t TOPK_ABOVE_A2_A3_LIMIT = 2049;
constexpr int64_t DEFAULT_QUANT_MODE = 1;
constexpr int64_t INT8_QUANT_MODE = 2;
constexpr int64_t MXFP8_QUANT_MODE = 3;
constexpr int64_t HIFLOAT8_QUANT_MODE = 4;
constexpr int64_t MXFP4_QUANT_MODE = 5;
constexpr int64_t NON_POWER_OF_TWO_COMPRESSION_RATIO = 3;
constexpr int64_t COMPRESSED_MASK_MODE = 3;
constexpr int64_t COMPRESSED_RATIO = 2;
constexpr int64_t TND_TOKEN_NUM = 8;
constexpr int64_t TND_CU_SEQLENS_SIZE = 3;
constexpr int64_t METADATA_SIZE = 1024;
constexpr std::array<int64_t, 4> UNSUPPORTED_QUANT_MODES = {DEFAULT_QUANT_MODE, MXFP8_QUANT_MODE, HIFLOAT8_QUANT_MODE,
                                                            MXFP4_QUANT_MODE};

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

        info.opName = OP_NAME;
        info.quantized = true;
        info.arch35 = false;
        info.layoutQText = BSND_LAYOUT;
        info.layoutKText = PAGED_ATTENTION_LAYOUT;
        info.topk = BASE_TOPK;
        info.quantMode = INT8_QUANT_MODE;
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
        PopulateDerivedInfo(info);
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
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape weightsShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape queryDescaleShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                            {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape keyDescaleShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM},
                                          {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM}};
    gert::StorageShape blockTableShape = {{BATCH_SIZE, KEY_BLOCK_NUM}, {BATCH_SIZE, KEY_BLOCK_NUM}};
    gert::StorageShape sequsedKShape = {{BATCH_SIZE}, {BATCH_SIZE}};
    gert::StorageShape indicesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, BASE_TOPK},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, BASE_TOPK}};
    gert::StorageShape valuesShape = {{0}, {0}};
    gert::StorageShape metadataShape = {{METADATA_SIZE}, {METADATA_SIZE}};
    gert::Stride keyStride = {KEY_BLOCK_SIZE * KEY_HEAD_NUM * HEAD_DIM, HEAD_DIM, HEAD_DIM, 1};
    gert::Stride keyDescaleStride = {KEY_BLOCK_SIZE, 1, 1};
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
    gert::StorageShape tndQueryShape = {{TND_TOKEN_NUM, QUERY_HEAD_NUM, HEAD_DIM},
                                        {TND_TOKEN_NUM, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape tndWeightsShape = {{TND_TOKEN_NUM, QUERY_HEAD_NUM}, {TND_TOKEN_NUM, QUERY_HEAD_NUM}};
    gert::StorageShape tndQueryDescaleShape = {{TND_TOKEN_NUM, QUERY_HEAD_NUM}, {TND_TOKEN_NUM, QUERY_HEAD_NUM}};
    gert::StorageShape tndIndicesShape = {{TND_TOKEN_NUM, KEY_HEAD_NUM, BASE_TOPK},
                                          {TND_TOKEN_NUM, KEY_HEAD_NUM, BASE_TOPK}};
    gert::StorageShape cuSeqlensQShape = {{TND_CU_SEQLENS_SIZE}, {TND_CU_SEQLENS_SIZE}};
    cuSeqlensQDesc.SetDataType(ge::DT_INT32);
    info.layoutQText = TND_LAYOUT;
    info.query.storageShape = &tndQueryShape;
    info.weights.storageShape = &tndWeightsShape;
    info.queryDescale.storageShape = &tndQueryDescaleShape;
    info.sparseIndices.storageShape = &tndIndicesShape;
    info.cuSeqlensQ = {&cuSeqlensQDesc, &cuSeqlensQShape, nullptr, nullptr};
    PopulateDerivedInfo(info);
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsNonPagedAttentionLayout)
{
    info.layoutKText = BSND_LAYOUT;
    info.layoutK = optiling::indexer_v2_checker::CheckerLayout::BSND;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsUnsupportedQuantModes)
{
    for (const int64_t quantMode : UNSUPPORTED_QUANT_MODES) {
        info.quantMode = quantMode;
        EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
    }
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsTopkAbove2048)
{
    info.topk = TOPK_ABOVE_A2_A3_LIMIT;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsNonPowerOfTwoCompressionRatio)
{
    info.cmpRatio = NON_POWER_OF_TWO_COMPRESSION_RATIO;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsReturningSparseValues)
{
    info.returnValue = 1;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, RejectsHeadGroupOtherThan64)
{
    info.qHeads = INVALID_QUERY_HEAD_NUM;
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
    gert::StorageShape cuSeqlensKShape = {{BATCH_SIZE}, {BATCH_SIZE}};
    cuSeqlensKDesc.SetDataType(ge::DT_INT32);
    info.cuSeqlensK = {&cuSeqlensKDesc, &cuSeqlensKShape, nullptr, nullptr};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2Arch22CheckerTest, ChecksCompressionResidualPresence)
{
    info.maskMode = COMPRESSED_MASK_MODE;
    info.cmpRatio = COMPRESSED_RATIO;
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_FAILED);

    gert::CompileTimeTensorDesc residualDesc;
    gert::StorageShape residualShape = {{BATCH_SIZE}, {BATCH_SIZE}};
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
    gert::StorageShape offsetShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                      {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    offsetDesc.SetDataType(ge::DT_FLOAT);
    info.outputIdxOffset = {&offsetDesc, &offsetShape, nullptr, nullptr};
    EXPECT_EQ(QLIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}
