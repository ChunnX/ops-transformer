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
 * \brief Tests common and A5 parameter checks for Lightning Indexer V2.
 */

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "../../../../op_host/checkers/checker_adapter.h"
#include "../../../../op_host/checkers/checker_runner.h"
#include "../../../../op_host/checkers/common_checker_indexer_v2.h"
#include "../../../../op_host/checkers/compression_checker_indexer_v2.h"
#include "../../../../op_host/checkers/liv2_checker.h"
#include "../../../../op_host/checkers/paged_attention_checker_indexer_v2.h"
#include "../../../../op_host/checkers/seq_len_checker_indexer_v2.h"

namespace {
using optiling::indexer_v2_checker::CheckerLayout;
using optiling::indexer_v2_checker::CheckerRunner;
using optiling::indexer_v2_checker::CheckerTensor;
using optiling::indexer_v2_checker::CommonChecker;
using optiling::indexer_v2_checker::CompressionChecker;
using optiling::indexer_v2_checker::IndexerV2BaseChecker;
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::LIV2Checker;
using optiling::indexer_v2_checker::PagedAttentionChecker;
using optiling::indexer_v2_checker::PopulateDerivedInfo;
using optiling::indexer_v2_checker::SeqLenChecker;

constexpr char OP_NAME[] = "LightningIndexerV2";
constexpr char QUANT_OP_NAME[] = "QuantLightningIndexerV2";
constexpr char BSND_LAYOUT[] = "BSND";
constexpr char TND_LAYOUT[] = "TND";
constexpr char PAGED_ATTENTION_LAYOUT[] = "PA_BBND";
constexpr char INVALID_LAYOUT[] = "INVALID";
constexpr int32_t EVENT_PHASE_FACTOR = 10;
constexpr int32_t SHAPE_PHASE_OFFSET = 2;
constexpr int32_t MULTI_PARA_PHASE_OFFSET = 3;
constexpr int32_t SECOND_CHECKER_ID = 2;
constexpr int64_t BATCH_SIZE = 2;
constexpr int64_t QUERY_SEQUENCE_LENGTH = 4;
constexpr int64_t INVALID_QUERY_SEQUENCE_LENGTH = 5;
constexpr int64_t KEY_SEQUENCE_LENGTH = 16;
constexpr int64_t QUERY_HEAD_NUM = 8;
constexpr int64_t A2_A3_QUERY_HEAD_NUM = 32;
constexpr int64_t A5_MAX_QUERY_HEAD_NUM = 64;
constexpr int64_t QUERY_HEAD_NUM_ABOVE_A5_LIMIT = 65;
constexpr int64_t KEY_HEAD_NUM = 1;
constexpr int64_t INVALID_KEY_HEAD_NUM = 2;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t INVALID_HEAD_DIM = 64;
constexpr int64_t TOPK = 64;
constexpr int64_t TOPK_BELOW_EXPECTED = 63;
constexpr int64_t TOPK_ABOVE_COMMON_LIMIT = 8193;
constexpr int64_t TOPK_ABOVE_A2_A3_LIMIT = 2049;
constexpr int64_t METADATA_SIZE = 1024;
constexpr int64_t INVALID_METADATA_SIZE = 1023;
constexpr int64_t INVALID_MASK_MODE = 2;
constexpr int64_t INVALID_COMPRESSION_RATIO = 129;
constexpr int64_t HISTORICAL_INVALID_MAX_SEQLEN_Q = -2;
constexpr int64_t HISTORICAL_RETURN_VALUE = 2;
constexpr int64_t COMPRESSED_RATIO = 2;
constexpr int64_t COMPRESSED_MASK_MODE = 3;
constexpr int64_t NON_POWER_OF_TWO_COMPRESSION_RATIO = 3;
constexpr int64_t TND_TOKEN_NUM = 8;
constexpr int64_t INVALID_TND_TOKEN_NUM = 7;
constexpr int64_t TND_QUERY_HEAD_NUM = 4;
constexpr int64_t TND_CU_SEQLENS_SIZE = 3;
constexpr int64_t SEQUENCE_SIZE_ABOVE_EXPECTED = 4;
constexpr int64_t BLOCK_TABLE_COLUMN_NUM = 8;
constexpr int64_t KEY_BLOCK_NUM = 4;
constexpr int64_t KEY_BLOCK_SIZE = 16;
constexpr int64_t INVALID_KEY_BLOCK_SIZE = 15;
constexpr int64_t BLOCK_SIZE_ABOVE_LIMIT = 1025;
constexpr int64_t INVALID_BATCH_SIZE = 3;
constexpr int64_t OPTIONAL_VECTOR_SIZE = 2;
constexpr int64_t GENERIC_STRIDE_DIM = 2;
constexpr int64_t CONTIGUOUS_KEY_LEADING_STRIDE = 4096;
constexpr int64_t CONTIGUOUS_KEY_BLOCK_STRIDE = 2048;
constexpr int64_t TEST_STRIDE_LEADING = 8;
constexpr int64_t TEST_STRIDE_SECOND = 4;
constexpr int64_t TEST_STRIDE_THIRD = 2;
constexpr int64_t INVALID_DESCALE_LEADING_STRIDE = 64;
constexpr int64_t INVALID_DESCALE_SECOND_STRIDE = 32;
constexpr int64_t HIFLOAT8_QUANT_MODE = 4;

class TestableBaseChecker : public IndexerV2BaseChecker {
public:
    using IndexerV2BaseChecker::CheckDtype;
    using IndexerV2BaseChecker::CheckRank;
};

class RecordingChecker : public IndexerV2BaseChecker {
public:
    RecordingChecker(std::vector<int32_t> &events, int32_t id, bool failExistence = false)
        : events_(events),
          id_(id),
          failExistence_(failExistence)
    {}

    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * EVENT_PHASE_FACTOR);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * EVENT_PHASE_FACTOR + 1);
        return failExistence_ ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckFeature(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * EVENT_PHASE_FACTOR + SHAPE_PHASE_OFFSET);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * EVENT_PHASE_FACTOR + MULTI_PARA_PHASE_OFFSET);
        return ge::GRAPH_SUCCESS;
    }

private:
    std::vector<int32_t> &events_;
    int32_t id_;
    bool failExistence_;
};

class ValidLiCheckerInfoTest : public testing::Test {
protected:
    void SetUp() override
    {
        queryDesc.SetDataType(ge::DT_BF16);
        keyDesc.SetDataType(ge::DT_BF16);
        weightsDesc.SetDataType(ge::DT_FLOAT);
        indicesDesc.SetDataType(ge::DT_INT32);
        valuesDesc.SetDataType(ge::DT_FLOAT);
        metadataDesc.SetDataType(ge::DT_UINT8);

        info.opName = OP_NAME;
        info.arch35 = true;
        info.layoutQText = BSND_LAYOUT;
        info.layoutKText = BSND_LAYOUT;
        info.topk = TOPK;
        info.maxSeqlenQ = -1;
        info.maskMode = 0;
        info.cmpRatio = 1;
        info.returnValue = 1;
        info.query = {&queryDesc, &queryShape, nullptr, nullptr};
        info.key = {&keyDesc, &keyShape, nullptr, nullptr};
        info.weights = {&weightsDesc, &weightsShape, nullptr, nullptr};
        info.sparseIndices = {&indicesDesc, &indicesShape, nullptr, nullptr};
        info.sparseValues = {&valuesDesc, &valuesShape, nullptr, nullptr};
        info.metadata = {&metadataDesc, &metadataShape, nullptr, nullptr};
        PopulateDerivedInfo(info);
    }

    gert::CompileTimeTensorDesc queryDesc;
    gert::CompileTimeTensorDesc keyDesc;
    gert::CompileTimeTensorDesc weightsDesc;
    gert::CompileTimeTensorDesc indicesDesc;
    gert::CompileTimeTensorDesc valuesDesc;
    gert::CompileTimeTensorDesc metadataDesc;
    gert::StorageShape queryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM},
                                     {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape weightsShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    gert::StorageShape indicesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK},
                                       {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK}};
    gert::StorageShape valuesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK},
                                      {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK}};
    gert::StorageShape metadataShape = {{METADATA_SIZE}, {METADATA_SIZE}};
    IndexerV2CheckerInfo info;
};
} // namespace

TEST(LightningIndexerV2Checker, TreatsShapeOnlyOptionalTensorAsPresent)
{
    gert::StorageShape shape = {{OPTIONAL_VECTOR_SIZE}, {OPTIONAL_VECTOR_SIZE}};
    const CheckerTensor tensor = {nullptr, &shape, nullptr, nullptr};

    EXPECT_TRUE(tensor.IsPresent());
    ASSERT_NE(tensor.GetShape(), nullptr);
    EXPECT_EQ(tensor.GetShape()->GetDim(0), OPTIONAL_VECTOR_SIZE);
}

TEST(LightningIndexerV2Checker, RunsEachPhaseAcrossAllCheckers)
{
    std::vector<int32_t> events;
    CheckerRunner runner;
    runner.Add(std::make_unique<RecordingChecker>(events, 1));
    runner.Add(std::make_unique<RecordingChecker>(events, SECOND_CHECKER_ID));

    const IndexerV2CheckerInfo info;
    EXPECT_EQ(runner.Process(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(events,
              (std::vector<int32_t>{EVENT_PHASE_FACTOR, SECOND_CHECKER_ID * EVENT_PHASE_FACTOR, EVENT_PHASE_FACTOR + 1,
                                    SECOND_CHECKER_ID * EVENT_PHASE_FACTOR + 1, EVENT_PHASE_FACTOR + SHAPE_PHASE_OFFSET,
                                    SECOND_CHECKER_ID * EVENT_PHASE_FACTOR + SHAPE_PHASE_OFFSET,
                                    EVENT_PHASE_FACTOR + MULTI_PARA_PHASE_OFFSET,
                                    SECOND_CHECKER_ID * EVENT_PHASE_FACTOR + MULTI_PARA_PHASE_OFFSET}));
}

TEST(LightningIndexerV2Checker, StopsAtFirstFailureWithinAPhase)
{
    std::vector<int32_t> events;
    CheckerRunner runner;
    runner.Add(std::make_unique<RecordingChecker>(events, 1, true));
    runner.Add(std::make_unique<RecordingChecker>(events, SECOND_CHECKER_ID));

    const IndexerV2CheckerInfo info;
    EXPECT_EQ(runner.Process(info), ge::GRAPH_FAILED);
    EXPECT_EQ(events, (std::vector<int32_t>{EVENT_PHASE_FACTOR, SECOND_CHECKER_ID * EVENT_PHASE_FACTOR,
                                            EVENT_PHASE_FACTOR + 1}));
}

TEST(LightningIndexerV2Checker, RejectsNullChecker)
{
    CheckerRunner runner;
    runner.Add(nullptr);

    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    EXPECT_EQ(runner.Process(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, AcceptsTndPagedAttentionBatchShapes)
{
    gert::StorageShape cuSeqlensQShape = {{TND_CU_SEQLENS_SIZE}, {TND_CU_SEQLENS_SIZE}};
    gert::StorageShape sequsedKShape = {{BATCH_SIZE}, {BATCH_SIZE}};
    gert::StorageShape blockTableShape = {{BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}, {BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}};

    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.layoutQ = CheckerLayout::TND;
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = BATCH_SIZE;
    info.cuSeqlensQ.storageShape = &cuSeqlensQShape;
    info.sequsedK.storageShape = &sequsedKShape;
    info.blockTable.storageShape = &blockTableShape;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(seqLenChecker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(info), ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, RejectsTndWithoutCuSeqlensQ)
{
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.layoutQ = CheckerLayout::TND;
    info.layoutK = CheckerLayout::BSND;

    SeqLenChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, AcceptsCompleteBsndInputThroughWrapper)
{
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_SUCCESS);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMissingRequiredDescriptor)
{
    info.key.desc = nullptr;
    EXPECT_EQ(CommonChecker().CheckParaExistence(info), ge::GRAPH_FAILED);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMissingSparseValuesWhenReturnValueIsEnabled)
{
    info.sparseValues = {};
    EXPECT_EQ(CommonChecker().CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsWrongOutputIdxOffsetDtype)
{
    gert::CompileTimeTensorDesc offsetDesc;
    offsetDesc.SetDataType(ge::DT_FLOAT);
    gert::StorageShape offsetShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                      {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    info.outputIdxOffset = {&offsetDesc, &offsetShape, nullptr, nullptr};

    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMissingLayoutText)
{
    info.layoutKText.clear();
    EXPECT_EQ(CommonChecker().CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidQueryLayout)
{
    info.layoutQ = CheckerLayout::INVALID;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMismatchedNonPagedLayouts)
{
    info.layoutK = CheckerLayout::TND;
    info.layoutKText = TND_LAYOUT;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsTopkOutsideRange)
{
    info.topk = 0;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.topk = TOPK_ABOVE_COMMON_LIMIT;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidMaskMode)
{
    info.maskMode = INVALID_MASK_MODE;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidCompressionRatio)
{
    info.cmpRatio = 0;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.cmpRatio = INVALID_COMPRESSION_RATIO;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidMaxSeqlenAndReturnValue)
{
    info.maxSeqlenQ = HISTORICAL_INVALID_MAX_SEQLEN_Q;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.maxSeqlenQ = -1;
    info.returnValue = HISTORICAL_RETURN_VALUE;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsWrongInputAndOutputRanks)
{
    gert::StorageShape wrongQueryShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                          {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    info.query.storageShape = &wrongQueryShape;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);

    info.query.storageShape = &queryShape;
    gert::StorageShape wrongOutputShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, TOPK},
                                           {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, TOPK}};
    info.sparseIndices.storageShape = &wrongOutputShape;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsWrongSparseIndicesDtype)
{
    indicesDesc.SetDataType(ge::DT_INT64);
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMismatchedQueryAndKeyDtypes)
{
    keyDesc.SetDataType(ge::DT_FLOAT16);
    EXPECT_EQ(LIV2Checker(info).Process(), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsHeadGroupAboveA5Limit)
{
    info.qHeads = QUERY_HEAD_NUM_ABOVE_A5_LIMIT;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMainShapeMismatch)
{
    gert::StorageShape wrongWeightsShape = {{BATCH_SIZE, INVALID_QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM},
                                            {BATCH_SIZE, INVALID_QUERY_SEQUENCE_LENGTH, QUERY_HEAD_NUM}};
    info.weights.storageShape = &wrongWeightsShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsOutputShapeMismatch)
{
    gert::StorageShape wrongIndicesShape = {{BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK_BELOW_EXPECTED},
                                            {BATCH_SIZE, QUERY_SEQUENCE_LENGTH, KEY_HEAD_NUM, TOPK_BELOW_EXPECTED}};
    info.sparseIndices.storageShape = &wrongIndicesShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMissingOrWrongSizedMetadataOnA5)
{
    info.metadata = {};
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    gert::StorageShape wrongMetadataShape = {{INVALID_METADATA_SIZE}, {INVALID_METADATA_SIZE}};
    info.metadata = {&metadataDesc, &wrongMetadataShape, nullptr, nullptr};
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsForbiddenSequenceInputsAndResidualMismatch)
{
    gert::StorageShape oneDimShape = {{OPTIONAL_VECTOR_SIZE}, {OPTIONAL_VECTOR_SIZE}};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.arch35 = true;
    info.layoutQ = CheckerLayout::BSND;
    info.layoutK = CheckerLayout::BSND;
    info.cuSeqlensQ.storageShape = &oneDimShape;
    EXPECT_EQ(SeqLenChecker().CheckParaExistence(info), ge::GRAPH_FAILED);

    info.cuSeqlensQ = {};
    info.cmpRatio = COMPRESSED_RATIO;
    info.maskMode = COMPRESSED_MASK_MODE;
    EXPECT_EQ(CompressionChecker().CheckParaExistence(info), ge::GRAPH_FAILED);

    info.cmpResidualK.storageShape = &oneDimShape;
    EXPECT_EQ(CompressionChecker().CheckParaExistence(info), ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, RejectsInvalidTndSequenceLengths)
{
    gert::StorageShape cuQSingleShape = {{1}, {1}};
    gert::StorageShape cuQShape = {{TND_CU_SEQLENS_SIZE}, {TND_CU_SEQLENS_SIZE}};
    gert::StorageShape cuKShape = {{BATCH_SIZE}, {BATCH_SIZE}};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.layoutQ = CheckerLayout::TND;
    info.layoutK = CheckerLayout::TND;
    info.cuSeqlensQ.storageShape = &cuQSingleShape;
    info.cuSeqlensK.storageShape = &cuKShape;
    EXPECT_EQ(SeqLenChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    info.cuSeqlensQ.storageShape = &cuQShape;
    EXPECT_EQ(SeqLenChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsPagedAttentionExistenceRankBlockSizeAndBatchErrors)
{
    gert::CompileTimeTensorDesc blockDesc;
    blockDesc.SetDataType(ge::DT_INT32);
    gert::StorageShape keyShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape blockShape = {{BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}, {BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}};
    gert::StorageShape badRankBlockShape = {{KEY_BLOCK_SIZE}, {KEY_BLOCK_SIZE}};
    gert::StorageShape sequsedShape = {{BATCH_SIZE}, {BATCH_SIZE}};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = BATCH_SIZE;
    info.blockSize = KEY_BLOCK_SIZE;
    info.key.storageShape = &keyShape;
    info.sequsedK.storageShape = &sequsedShape;

    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.blockTable = {&blockDesc, &badRankBlockShape, nullptr, nullptr};
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockTable.storageShape = &blockShape;
    info.blockSize = INVALID_KEY_BLOCK_SIZE;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockSize = KEY_BLOCK_SIZE;
    info.batch = INVALID_BATCH_SIZE;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsNonContiguousPagedAttentionKey)
{
    gert::StorageShape keyShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape blockShape = {{BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}, {BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}};
    gert::Stride badStride = {CONTIGUOUS_KEY_LEADING_STRIDE, CONTIGUOUS_KEY_BLOCK_STRIDE, HEAD_DIM, 1};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.arch35 = true;
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = BATCH_SIZE;
    info.key = {nullptr, &keyShape, nullptr, &badStride};
    info.blockTable.storageShape = &blockShape;
    EXPECT_EQ(PagedAttentionChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsInvalidDimensionsDuringContiguousCheck)
{
    gert::StorageShape negativeShape = {{-1, GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM},
                                        {-1, GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM}};
    gert::StorageShape overflowingShape = {
        {std::numeric_limits<int64_t>::max(), GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM},
        {std::numeric_limits<int64_t>::max(), GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM, GENERIC_STRIDE_DIM}};
    gert::Stride stride = {TEST_STRIDE_LEADING, TEST_STRIDE_SECOND, TEST_STRIDE_THIRD, 1};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.layoutK = CheckerLayout::BSND;
    info.key = {nullptr, &negativeShape, nullptr, &stride};

    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    info.key.storageShape = &overflowingShape;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, CoversBaseCheckerDefaultsAndLayoutParsing)
{
    IndexerV2CheckerInfo info;
    IndexerV2BaseChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckFeature(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
    info.layoutQText = BSND_LAYOUT;
    info.layoutKText = TND_LAYOUT;
    PopulateDerivedInfo(info);
    EXPECT_EQ(info.layoutQ, CheckerLayout::BSND);
    EXPECT_EQ(info.layoutK, CheckerLayout::TND);
    info.layoutQText = PAGED_ATTENTION_LAYOUT;
    info.layoutKText = INVALID_LAYOUT;
    PopulateDerivedInfo(info);
    EXPECT_EQ(info.layoutQ, CheckerLayout::PA_BBND);
    EXPECT_EQ(info.layoutK, CheckerLayout::INVALID);
}

TEST(LightningIndexerV2Checker, CoversBaseCheckerOptionalAndDescriptorBranches)
{
    gert::StorageShape shape = {{OPTIONAL_VECTOR_SIZE}, {OPTIONAL_VECTOR_SIZE}};
    gert::CompileTimeTensorDesc desc;
    desc.SetDataType(ge::DT_FLOAT);
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    TestableBaseChecker checker;

    EXPECT_EQ(checker.CheckDtype(info, {}, "optional", {ge::DT_INT32}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckRank(info, {}, "optional", 1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckDtype(info, {nullptr, &shape, nullptr, nullptr}, "missing_desc", {ge::DT_INT32}),
              ge::GRAPH_FAILED);
    EXPECT_EQ(checker.CheckDtype(info, {&desc, &shape, nullptr, nullptr}, "dtype", {ge::DT_FLOAT, ge::DT_INT32}),
              ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, PopulatesTndAndPagedAttentionDimensions)
{
    gert::StorageShape queryShape = {{TND_TOKEN_NUM, TND_QUERY_HEAD_NUM, HEAD_DIM},
                                     {TND_TOKEN_NUM, TND_QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape blockShape = {{BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}, {BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}};
    gert::StorageShape cuShape = {{TND_CU_SEQLENS_SIZE}, {TND_CU_SEQLENS_SIZE}};
    IndexerV2CheckerInfo info;
    info.layoutQText = TND_LAYOUT;
    info.layoutKText = PAGED_ATTENTION_LAYOUT;
    info.query.storageShape = &queryShape;
    info.key.storageShape = &keyShape;
    info.blockTable.storageShape = &blockShape;
    info.cuSeqlensQ.storageShape = &cuShape;

    PopulateDerivedInfo(info);
    EXPECT_EQ(info.batch, BATCH_SIZE);
    EXPECT_EQ(info.qSeq, TND_TOKEN_NUM);
    EXPECT_EQ(info.qHeads, TND_QUERY_HEAD_NUM);
    EXPECT_EQ(info.headDim, HEAD_DIM);
    EXPECT_EQ(info.kHeads, 1);
    EXPECT_EQ(info.blockSize, KEY_BLOCK_SIZE);
    EXPECT_EQ(info.kSeq, KEY_BLOCK_SIZE * BLOCK_TABLE_COLUMN_NUM);
}

TEST(LightningIndexerV2Checker, RejectsEveryWrongSequenceTensorDtype)
{
    gert::CompileTimeTensorDesc badDesc;
    badDesc.SetDataType(ge::DT_FLOAT);
    gert::StorageShape shape = {{OPTIONAL_VECTOR_SIZE}, {OPTIONAL_VECTOR_SIZE}};
    const CheckerTensor badTensor = {&badDesc, &shape, nullptr, nullptr};
    SeqLenChecker checker;
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.arch35 = true;

    info.layoutK = CheckerLayout::TND;
    info.cuSeqlensK = badTensor;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.layoutK = CheckerLayout::BSND;
    info.cuSeqlensK = {};
    info.sequsedK = badTensor;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.sequsedK = {};
    info.layoutQ = CheckerLayout::TND;
    info.cuSeqlensQ = badTensor;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.layoutQ = CheckerLayout::BSND;
    info.cuSeqlensQ = {};
    info.sequsedQ = badTensor;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsWrongCompressionTensorDtype)
{
    gert::CompileTimeTensorDesc badDesc;
    badDesc.SetDataType(ge::DT_FLOAT);
    gert::StorageShape shape = {{OPTIONAL_VECTOR_SIZE}, {OPTIONAL_VECTOR_SIZE}};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.arch35 = true;
    info.cmpResidualK = {&badDesc, &shape, nullptr, nullptr};

    EXPECT_EQ(CompressionChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsInvalidSequenceTensorPresenceCombinations)
{
    gert::StorageShape shape = {{OPTIONAL_VECTOR_SIZE}, {OPTIONAL_VECTOR_SIZE}};
    SeqLenChecker checker;
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;

    info.layoutK = CheckerLayout::TND;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.layoutK = CheckerLayout::BSND;
    info.cuSeqlensK.storageShape = &shape;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.layoutK = CheckerLayout::PA_BBND;
    info.arch35 = true;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.cuSeqlensK = {};
    info.layoutQ = CheckerLayout::BSND;
    info.cuSeqlensQ.storageShape = &shape;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.cuSeqlensQ = {};
    info.cmpResidualK.storageShape = &shape;
    EXPECT_EQ(CompressionChecker().CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsMismatchedOptionalSequenceTensorSizes)
{
    gert::StorageShape sizeTwo = {{BATCH_SIZE}, {BATCH_SIZE}};
    gert::StorageShape sizeThree = {{TND_CU_SEQLENS_SIZE}, {TND_CU_SEQLENS_SIZE}};
    gert::StorageShape sizeFour = {{SEQUENCE_SIZE_ABOVE_EXPECTED}, {SEQUENCE_SIZE_ABOVE_EXPECTED}};
    SeqLenChecker checker;
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.arch35 = true;
    info.batch = BATCH_SIZE;
    info.layoutQ = CheckerLayout::TND;
    info.layoutK = CheckerLayout::PA_BBND;
    info.cuSeqlensQ.storageShape = &sizeThree;
    info.sequsedK.storageShape = &sizeFour;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);

    info.layoutQ = CheckerLayout::BSND;
    info.layoutK = CheckerLayout::BSND;
    info.quantized = true;
    info.sequsedK = {};
    info.sequsedQ.storageShape = &sizeThree;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    info.sequsedQ = {};
    info.sequsedK.storageShape = &sizeThree;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    info.sequsedK = {};
    info.cmpResidualK.storageShape = &sizeThree;
    EXPECT_EQ(CompressionChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    info.cmpResidualK.storageShape = &sizeTwo;
    EXPECT_EQ(CompressionChecker().CheckMultiPara(info), ge::GRAPH_SUCCESS);
}

TEST_F(ValidLiCheckerInfoTest, RejectsA2A3SpecificAttributeRestrictions)
{
    CommonChecker checker;
    info.arch35 = false;
    info.quantized = true;
    info.returnValue = 0;
    info.topk = TOPK_ABOVE_A2_A3_LIMIT;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.quantized = false;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.quantized = true;
    info.topk = TOPK;
    info.cmpRatio = NON_POWER_OF_TWO_COMPRESSION_RATIO;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.cmpRatio = 1;
    info.returnValue = 1;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidKeyLayoutAndFeatureDimensions)
{
    info.layoutK = CheckerLayout::INVALID;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.layoutK = CheckerLayout::BSND;
    info.kHeads = INVALID_KEY_HEAD_NUM;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
    info.kHeads = 1;
    info.headDim = INVALID_HEAD_DIM;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
    info.headDim = HEAD_DIM;
    info.arch35 = false;
    info.quantized = true;
    info.qHeads = A2_A3_QUERY_HEAD_NUM;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsBatchAndKeyHeadShapeMismatches)
{
    gert::StorageShape wrongBatchKeyShape = {{INVALID_BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                             {INVALID_BATCH_SIZE, KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    info.key.storageShape = &wrongBatchKeyShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    gert::StorageShape wrongHeadKeyShape = {{BATCH_SIZE, KEY_SEQUENCE_LENGTH, INVALID_KEY_HEAD_NUM, HEAD_DIM},
                                            {BATCH_SIZE, KEY_SEQUENCE_LENGTH, INVALID_KEY_HEAD_NUM, HEAD_DIM}};
    info.key.storageShape = &wrongHeadKeyShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsTndValueAndOffsetShapeMismatches)
{
    gert::StorageShape queryShape = {{TND_TOKEN_NUM, TND_QUERY_HEAD_NUM, HEAD_DIM},
                                     {TND_TOKEN_NUM, TND_QUERY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape keyShape = {{KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_SEQUENCE_LENGTH, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape weightsShape = {{TND_TOKEN_NUM, TND_QUERY_HEAD_NUM}, {TND_TOKEN_NUM, TND_QUERY_HEAD_NUM}};
    gert::StorageShape outputShape = {{TND_TOKEN_NUM, KEY_HEAD_NUM, TOPK}, {TND_TOKEN_NUM, KEY_HEAD_NUM, TOPK}};
    gert::StorageShape wrongValuesShape = {{INVALID_TND_TOKEN_NUM, KEY_HEAD_NUM, TOPK},
                                           {INVALID_TND_TOKEN_NUM, KEY_HEAD_NUM, TOPK}};
    gert::StorageShape wrongOffsetShape = {{INVALID_TND_TOKEN_NUM, TND_QUERY_HEAD_NUM},
                                           {INVALID_TND_TOKEN_NUM, TND_QUERY_HEAD_NUM}};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.arch35 = true;
    info.returnValue = 1;
    info.topk = TOPK;
    info.layoutQ = CheckerLayout::TND;
    info.layoutK = CheckerLayout::TND;
    info.query.storageShape = &queryShape;
    info.key.storageShape = &keyShape;
    info.weights.storageShape = &weightsShape;
    info.sparseIndices.storageShape = &outputShape;
    info.sparseValues.storageShape = &wrongValuesShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    info.sparseValues.storageShape = &outputShape;
    info.outputIdxOffset.storageShape = &wrongOffsetShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, CoversPagedAttentionExistenceAndBlockBoundaries)
{
    gert::CompileTimeTensorDesc blockDesc;
    blockDesc.SetDataType(ge::DT_INT32);
    gert::CompileTimeTensorDesc badBlockDesc;
    badBlockDesc.SetDataType(ge::DT_FLOAT);
    gert::StorageShape blockShape = {{BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}, {BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}};
    gert::StorageShape sequenceShape = {{BATCH_SIZE}, {BATCH_SIZE}};
    gert::StorageShape keyShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    gert::StorageShape emptyKeyShape = {{0, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                        {0, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    IndexerV2CheckerInfo info;
    info.opName = OP_NAME;
    info.layoutK = CheckerLayout::BSND;
    info.blockTable = {&blockDesc, &blockShape, nullptr, nullptr};
    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);

    info.layoutK = CheckerLayout::PA_BBND;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.sequsedK.storageShape = &sequenceShape;
    info.blockTable.desc = &badBlockDesc;
    info.blockSize = KEY_BLOCK_SIZE;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockTable.desc = &blockDesc;
    info.blockSize = BLOCK_SIZE_ABOVE_LIMIT;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockSize = KEY_BLOCK_SIZE;
    info.key.storageShape = &emptyKeyShape;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.key = {};
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_SUCCESS);
    info.key.storageShape = &keyShape;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, CoversPagedAttentionContiguousAndDescaleBranches)
{
    gert::StorageShape blockShape = {{BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}, {BATCH_SIZE, BLOCK_TABLE_COLUMN_NUM}};
    gert::StorageShape keyShape = {{KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM},
                                   {KEY_BLOCK_NUM, KEY_BLOCK_SIZE, KEY_HEAD_NUM, HEAD_DIM}};
    gert::Stride keyStride = {CONTIGUOUS_KEY_BLOCK_STRIDE, HEAD_DIM, HEAD_DIM, 1};
    gert::Stride badDescaleStride = {INVALID_DESCALE_LEADING_STRIDE, INVALID_DESCALE_SECOND_STRIDE, 1, 1};
    IndexerV2CheckerInfo info;
    info.opName = QUANT_OP_NAME;
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = BATCH_SIZE;
    info.quantized = true;
    info.quantMode = 0;
    info.blockTable.storageShape = &blockShape;
    info.key = {nullptr, &keyShape, nullptr, &keyStride};
    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);

    info.keyDescale = {nullptr, &keyShape, nullptr, &badDescaleStride};
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    info.quantMode = HIFLOAT8_QUANT_MODE;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
}
