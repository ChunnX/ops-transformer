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

#include <limits>
#include <memory>
#include <vector>

#include "../../../op_host/checkers/checker_runner.h"
#include "../../../op_host/checkers/common_checker_indexer_v2.h"
#include "../../../op_host/checkers/liv2_checker.h"
#include "../../../op_host/checkers/paged_attention_checker_indexer_v2.h"
#include "../../../op_host/checkers/seq_len_checker_indexer_v2.h"

namespace {
using optiling::indexer_v2_checker::CheckerLayout;
using optiling::indexer_v2_checker::CheckerRunner;
using optiling::indexer_v2_checker::CheckerTensor;
using optiling::indexer_v2_checker::CommonChecker;
using optiling::indexer_v2_checker::FinalizeCommonInfo;
using optiling::indexer_v2_checker::IndexerV2BaseChecker;
using optiling::indexer_v2_checker::IndexerV2CheckerInfo;
using optiling::indexer_v2_checker::LIV2Checker;
using optiling::indexer_v2_checker::PagedAttentionChecker;
using optiling::indexer_v2_checker::ParseLayout;
using optiling::indexer_v2_checker::SeqLenChecker;

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
        events_.push_back(id_ * 10);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * 10 + 1);
        return failExistence_ ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckFeature(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * 10 + 2);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &) const override
    {
        events_.push_back(id_ * 10 + 3);
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

        info.opName = "LightningIndexerV2";
        info.arch35 = true;
        info.layoutQText = "BSND";
        info.layoutKText = "BSND";
        info.topk = 64;
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
        FinalizeCommonInfo(info);
    }

    gert::CompileTimeTensorDesc queryDesc;
    gert::CompileTimeTensorDesc keyDesc;
    gert::CompileTimeTensorDesc weightsDesc;
    gert::CompileTimeTensorDesc indicesDesc;
    gert::CompileTimeTensorDesc valuesDesc;
    gert::CompileTimeTensorDesc metadataDesc;
    gert::StorageShape queryShape = {{2, 4, 8, 128}, {2, 4, 8, 128}};
    gert::StorageShape keyShape = {{2, 16, 1, 128}, {2, 16, 1, 128}};
    gert::StorageShape weightsShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape indicesShape = {{2, 4, 1, 64}, {2, 4, 1, 64}};
    gert::StorageShape valuesShape = {{2, 4, 1, 64}, {2, 4, 1, 64}};
    gert::StorageShape metadataShape = {{1024}, {1024}};
    IndexerV2CheckerInfo info;
};
} // namespace

TEST(LightningIndexerV2Checker, TreatsShapeOnlyOptionalTensorAsPresent)
{
    gert::StorageShape shape = {{2}, {2}};
    const CheckerTensor tensor = {nullptr, &shape, nullptr, nullptr};

    EXPECT_TRUE(tensor.IsPresent());
    ASSERT_NE(tensor.GetShape(), nullptr);
    EXPECT_EQ(tensor.GetShape()->GetDim(0), 2);
}

TEST(LightningIndexerV2Checker, RunsEachPhaseAcrossAllCheckers)
{
    std::vector<int32_t> events;
    CheckerRunner runner;
    runner.Add(std::make_unique<RecordingChecker>(events, 1));
    runner.Add(std::make_unique<RecordingChecker>(events, 2));

    const IndexerV2CheckerInfo info;
    EXPECT_EQ(runner.Process(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(events, (std::vector<int32_t>{10, 20, 11, 21, 12, 22, 13, 23}));
}

TEST(LightningIndexerV2Checker, StopsAtFirstFailureWithinAPhase)
{
    std::vector<int32_t> events;
    CheckerRunner runner;
    runner.Add(std::make_unique<RecordingChecker>(events, 1, true));
    runner.Add(std::make_unique<RecordingChecker>(events, 2));

    const IndexerV2CheckerInfo info;
    EXPECT_EQ(runner.Process(info), ge::GRAPH_FAILED);
    EXPECT_EQ(events, (std::vector<int32_t>{10, 20, 11}));
}

TEST(LightningIndexerV2Checker, RejectsNullChecker)
{
    CheckerRunner runner;
    runner.Add(nullptr);

    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    EXPECT_EQ(runner.Process(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, AcceptsTndPagedAttentionBatchShapes)
{
    gert::StorageShape cuSeqlensQShape = {{3}, {3}};
    gert::StorageShape sequsedKShape = {{2}, {2}};
    gert::StorageShape blockTableShape = {{2, 8}, {2, 8}};

    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.layoutQ = CheckerLayout::TND;
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = 2;
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
    info.opName = "LightningIndexerV2";
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
    info.layoutKText = "TND";
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsTopkOutsideRange)
{
    info.topk = 0;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.topk = 8193;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidMaskMode)
{
    info.maskMode = 2;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidCompressionRatio)
{
    info.cmpRatio = 0;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.cmpRatio = 129;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsInvalidMaxSeqlenAndReturnValue)
{
    info.maxSeqlenQ = -2;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
    info.maxSeqlenQ = -1;
    info.returnValue = 2;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsWrongInputAndOutputRanks)
{
    gert::StorageShape wrongQueryShape = {{2, 4, 8}, {2, 4, 8}};
    info.query.storageShape = &wrongQueryShape;
    EXPECT_EQ(CommonChecker().CheckSinglePara(info), ge::GRAPH_FAILED);

    info.query.storageShape = &queryShape;
    gert::StorageShape wrongOutputShape = {{2, 4, 64}, {2, 4, 64}};
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
    info.qHeads = 65;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMainShapeMismatch)
{
    gert::StorageShape wrongWeightsShape = {{2, 5, 8}, {2, 5, 8}};
    info.weights.storageShape = &wrongWeightsShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsOutputShapeMismatch)
{
    gert::StorageShape wrongIndicesShape = {{2, 4, 1, 63}, {2, 4, 1, 63}};
    info.sparseIndices.storageShape = &wrongIndicesShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsMissingOrWrongSizedMetadataOnA5)
{
    info.metadata = {};
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    gert::StorageShape wrongMetadataShape = {{1023}, {1023}};
    info.metadata = {&metadataDesc, &wrongMetadataShape, nullptr, nullptr};
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsForbiddenSequenceInputsAndResidualMismatch)
{
    gert::StorageShape oneDimShape = {{2}, {2}};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.arch35 = true;
    info.layoutQ = CheckerLayout::BSND;
    info.layoutK = CheckerLayout::BSND;
    info.cuSeqlensQ.storageShape = &oneDimShape;
    EXPECT_EQ(SeqLenChecker().CheckParaExistence(info), ge::GRAPH_FAILED);

    info.cuSeqlensQ = {};
    info.cmpRatio = 2;
    info.maskMode = 3;
    EXPECT_EQ(SeqLenChecker().CheckParaExistence(info), ge::GRAPH_FAILED);

    info.cmpResidualK.storageShape = &oneDimShape;
    EXPECT_EQ(SeqLenChecker().CheckParaExistence(info), ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, RejectsInvalidTndSequenceLengths)
{
    gert::StorageShape cuQSingleShape = {{1}, {1}};
    gert::StorageShape cuQShape = {{3}, {3}};
    gert::StorageShape cuKShape = {{2}, {2}};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
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
    gert::StorageShape keyShape = {{4, 16, 1, 128}, {4, 16, 1, 128}};
    gert::StorageShape blockShape = {{2, 8}, {2, 8}};
    gert::StorageShape badRankBlockShape = {{16}, {16}};
    gert::StorageShape sequsedShape = {{2}, {2}};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = 2;
    info.blockSize = 16;
    info.key.storageShape = &keyShape;
    info.sequsedK.storageShape = &sequsedShape;

    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.blockTable = {&blockDesc, &badRankBlockShape, nullptr, nullptr};
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockTable.storageShape = &blockShape;
    info.blockSize = 15;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockSize = 16;
    info.batch = 3;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsNonContiguousPagedAttentionKey)
{
    gert::StorageShape keyShape = {{4, 16, 1, 128}, {4, 16, 1, 128}};
    gert::StorageShape blockShape = {{2, 8}, {2, 8}};
    gert::Stride badStride = {4096, 2048, 128, 1};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.arch35 = true;
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = 2;
    info.key = {nullptr, &keyShape, nullptr, &badStride};
    info.blockTable.storageShape = &blockShape;
    EXPECT_EQ(PagedAttentionChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsInvalidDimensionsDuringContiguousCheck)
{
    gert::StorageShape negativeShape = {{-1, 2, 2, 2}, {-1, 2, 2, 2}};
    gert::StorageShape overflowingShape = {{std::numeric_limits<int64_t>::max(), 2, 2, 2},
                                           {std::numeric_limits<int64_t>::max(), 2, 2, 2}};
    gert::Stride stride = {8, 4, 2, 1};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.layoutK = CheckerLayout::BSND;
    info.key = {nullptr, &negativeShape, nullptr, &stride};

    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    info.key.storageShape = &overflowingShape;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, CoversBaseCheckerDefaultsAndLayoutParsing)
{
    const IndexerV2CheckerInfo info;
    IndexerV2BaseChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckFeature(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ParseLayout("BSND"), CheckerLayout::BSND);
    EXPECT_EQ(ParseLayout("TND"), CheckerLayout::TND);
    EXPECT_EQ(ParseLayout("PA_BBND"), CheckerLayout::PA_BBND);
    EXPECT_EQ(ParseLayout("INVALID"), CheckerLayout::INVALID);
}

TEST(LightningIndexerV2Checker, CoversBaseCheckerOptionalAndDescriptorBranches)
{
    gert::StorageShape shape = {{2}, {2}};
    gert::CompileTimeTensorDesc desc;
    desc.SetDataType(ge::DT_FLOAT);
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    TestableBaseChecker checker;

    EXPECT_EQ(checker.CheckDtype(info, {}, "optional", {ge::DT_INT32}), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckRank(info, {}, "optional", 1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckDtype(info, {nullptr, &shape, nullptr, nullptr}, "missing_desc", {ge::DT_INT32}),
              ge::GRAPH_FAILED);
    EXPECT_EQ(checker.CheckDtype(info, {&desc, &shape, nullptr, nullptr}, "dtype", {ge::DT_FLOAT, ge::DT_INT32}),
              ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, FinalizesTndAndPagedAttentionDimensions)
{
    gert::StorageShape queryShape = {{8, 4, 128}, {8, 4, 128}};
    gert::StorageShape keyShape = {{4, 16, 1, 128}, {4, 16, 1, 128}};
    gert::StorageShape blockShape = {{2, 8}, {2, 8}};
    gert::StorageShape cuShape = {{3}, {3}};
    IndexerV2CheckerInfo info;
    info.layoutQText = "TND";
    info.layoutKText = "PA_BBND";
    info.query.storageShape = &queryShape;
    info.key.storageShape = &keyShape;
    info.blockTable.storageShape = &blockShape;
    info.cuSeqlensQ.storageShape = &cuShape;

    FinalizeCommonInfo(info);
    EXPECT_EQ(info.batch, 2);
    EXPECT_EQ(info.qSeq, 8);
    EXPECT_EQ(info.qHeads, 4);
    EXPECT_EQ(info.headDim, 128);
    EXPECT_EQ(info.kHeads, 1);
    EXPECT_EQ(info.blockSize, 16);
    EXPECT_EQ(info.kSeq, 128);
}

TEST(LightningIndexerV2Checker, RejectsEveryWrongSequenceTensorDtype)
{
    gert::CompileTimeTensorDesc badDesc;
    badDesc.SetDataType(ge::DT_FLOAT);
    gert::StorageShape shape = {{2}, {2}};
    const CheckerTensor badTensor = {&badDesc, &shape, nullptr, nullptr};
    SeqLenChecker checker;
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
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
    info.sequsedQ = {};
    info.cmpResidualK = badTensor;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.cmpResidualK = {};
    info.outputIdxOffset = badTensor;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsInvalidSequenceTensorPresenceCombinations)
{
    gert::StorageShape shape = {{2}, {2}};
    SeqLenChecker checker;
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";

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
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsMismatchedOptionalSequenceTensorSizes)
{
    gert::StorageShape sizeTwo = {{2}, {2}};
    gert::StorageShape sizeThree = {{3}, {3}};
    gert::StorageShape sizeFour = {{4}, {4}};
    SeqLenChecker checker;
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.arch35 = true;
    info.batch = 2;
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
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);

    info.cmpResidualK.storageShape = &sizeTwo;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
}

TEST_F(ValidLiCheckerInfoTest, RejectsA2A3SpecificAttributeRestrictions)
{
    CommonChecker checker;
    info.arch35 = false;
    info.quantized = true;
    info.returnValue = 0;
    info.topk = 2049;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.quantized = false;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.quantized = true;
    info.topk = 64;
    info.cmpRatio = 3;
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
    info.kHeads = 2;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
    info.kHeads = 1;
    info.headDim = 64;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
    info.headDim = 128;
    info.arch35 = false;
    info.quantized = true;
    info.qHeads = 32;
    EXPECT_EQ(CommonChecker().CheckFeature(info), ge::GRAPH_FAILED);
}

TEST_F(ValidLiCheckerInfoTest, RejectsBatchAndKeyHeadShapeMismatches)
{
    gert::StorageShape wrongBatchKeyShape = {{3, 16, 1, 128}, {3, 16, 1, 128}};
    info.key.storageShape = &wrongBatchKeyShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);

    gert::StorageShape wrongHeadKeyShape = {{2, 16, 2, 128}, {2, 16, 2, 128}};
    info.key.storageShape = &wrongHeadKeyShape;
    EXPECT_EQ(CommonChecker().CheckMultiPara(info), ge::GRAPH_FAILED);
}

TEST(LightningIndexerV2Checker, RejectsTndValueAndOffsetShapeMismatches)
{
    gert::StorageShape queryShape = {{8, 4, 128}, {8, 4, 128}};
    gert::StorageShape keyShape = {{16, 1, 128}, {16, 1, 128}};
    gert::StorageShape weightsShape = {{8, 4}, {8, 4}};
    gert::StorageShape outputShape = {{8, 1, 64}, {8, 1, 64}};
    gert::StorageShape wrongValuesShape = {{7, 1, 64}, {7, 1, 64}};
    gert::StorageShape wrongOffsetShape = {{7, 4}, {7, 4}};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.arch35 = true;
    info.returnValue = 1;
    info.topk = 64;
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
    gert::StorageShape blockShape = {{2, 8}, {2, 8}};
    gert::StorageShape sequenceShape = {{2}, {2}};
    gert::StorageShape keyShape = {{4, 16, 1, 128}, {4, 16, 1, 128}};
    gert::StorageShape emptyKeyShape = {{0, 16, 1, 128}, {0, 16, 1, 128}};
    IndexerV2CheckerInfo info;
    info.opName = "LightningIndexerV2";
    info.layoutK = CheckerLayout::BSND;
    info.blockTable = {&blockDesc, &blockShape, nullptr, nullptr};
    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);

    info.layoutK = CheckerLayout::PA_BBND;
    EXPECT_EQ(checker.CheckParaExistence(info), ge::GRAPH_FAILED);
    info.sequsedK.storageShape = &sequenceShape;
    info.blockTable.desc = &badBlockDesc;
    info.blockSize = 16;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockTable.desc = &blockDesc;
    info.blockSize = 1025;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.blockSize = 16;
    info.key.storageShape = &emptyKeyShape;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_FAILED);
    info.key = {};
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_SUCCESS);
    info.key.storageShape = &keyShape;
    EXPECT_EQ(checker.CheckSinglePara(info), ge::GRAPH_SUCCESS);
}

TEST(LightningIndexerV2Checker, CoversPagedAttentionContiguousAndDescaleBranches)
{
    gert::StorageShape blockShape = {{2, 8}, {2, 8}};
    gert::StorageShape keyShape = {{4, 16, 1, 128}, {4, 16, 1, 128}};
    gert::Stride keyStride = {2048, 128, 128, 1};
    gert::Stride badDescaleStride = {64, 32, 1, 1};
    IndexerV2CheckerInfo info;
    info.opName = "QuantLightningIndexerV2";
    info.layoutK = CheckerLayout::PA_BBND;
    info.batch = 2;
    info.quantized = true;
    info.quantMode = 0;
    info.blockTable.storageShape = &blockShape;
    info.key = {nullptr, &keyShape, nullptr, &keyStride};
    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);

    info.keyDescale = {nullptr, &keyShape, nullptr, &badDescaleStride};
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    info.quantMode = 4;
    EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
}
