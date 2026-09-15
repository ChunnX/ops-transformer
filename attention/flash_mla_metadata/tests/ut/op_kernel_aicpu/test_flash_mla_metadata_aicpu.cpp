/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "../../../op_kernel_aicpu/flash_mla_metadata_aicpu_sequence.h"

namespace {
using namespace FlashMla;
using namespace aicpu;

class FlashMlaMetadataAicpuTest : public testing::Test {
protected:
    void SetUp() override
    {
        config_.numHeads = 8;
        config_.batchSize = 4;
        config_.qSeqSize = 4;
        config_.kvSeqSize = 4096;
        platform_.aicNum = 24;
        platform_.aivNum = 48;
        std::memset(&metadata_, 0xa5, sizeof(metadata_));
        metadata_.header.status = MlaStatus::SUCCESS;
    }

    void UseQuantizedDecode()
    {
        config_.queryDataType = kMlaInt8;
        config_.qSeqSize = 1;
        config_.layoutQ = MlaLayout::BSND;
        config_.layoutKv = MlaLayout::NZ;
        config_.blockSize = 128;
        seqKv_ = {1024};
    }

    void UseTnd()
    {
        config_.layoutQ = MlaLayout::TND;
        config_.layoutKv = MlaLayout::TND;
        config_.qSeqSize = 7;
        config_.kvSeqSize = 224;
        seqQ_ = {0, 1, 3, 7};
        seqKv_ = {0, 32, 96, 224};
    }

    std::shared_ptr<NodeDef> BuildNode(const std::string &missingAttr = "", bool addInputs = true,
                                       bool addOutput = true)
    {
        static const std::array<std::string, 5> layouts = {"BSH", "BSND", "BNSD", "TND", "NZ"};
        auto node = CpuKernelUtils::CreateNodeDef();
        NodeDefBuilder builder(node.get(), "FlashMlaMetadata", "FlashMlaMetadata");
        if (addInputs) {
            builder
                .Input({"actual_seq_lengths_q",
                        DT_INT64,
                        {static_cast<int64_t>(seqQ_.size())},
                        seqQ_.empty() ? nullptr : seqQ_.data(),
                        FORMAT_ND})
                .Input({"actual_seq_lengths_kv",
                        DT_INT64,
                        {static_cast<int64_t>(seqKv_.size())},
                        seqKv_.empty() ? nullptr : seqKv_.data(),
                        FORMAT_ND});
        }
        if (addOutput) {
            builder.Output({"metadata", DT_INT32, {kMlaMetadataElements}, &metadata_, FORMAT_ND});
        }
        const std::pair<const char *, int64_t> integers[] = {{"num_heads", config_.numHeads},
                                                             {"batch_size", config_.batchSize},
                                                             {"q_seq_size", static_cast<int64_t>(config_.qSeqSize)},
                                                             {"kv_seq_size", static_cast<int64_t>(config_.kvSeqSize)},
                                                             {"block_size", config_.blockSize},
                                                             {"sparse_mode", config_.sparseMode},
                                                             {"pre_tokens", config_.preTokens},
                                                             {"next_tokens", config_.nextTokens},
                                                             {"quant_mode", MlaQuantMode(config_)},
                                                             {"aic_core_num", platform_.aicNum},
                                                             {"aiv_core_num", platform_.aivNum}};
        for (const auto &attr : integers) {
            if (missingAttr != attr.first) {
                builder.Attr(attr.first, attr.second);
            }
        }
        if (missingAttr != "input_layer") {
            builder.Attr("input_layer", layouts[static_cast<uint32_t>(config_.layoutQ)]);
        }
        if (missingAttr != "has_atten_mask") {
            builder.Attr("has_atten_mask", config_.hasAttenMask != 0);
        }
        return node;
    }

    uint32_t Run(const std::shared_ptr<NodeDef> &node)
    {
        CpuKernelContext context(HOST);
        const auto status = context.Init(node.get());
        EXPECT_EQ(status, KERNEL_STATUS_OK);
        if (status != KERNEL_STATUS_OK) {
            return status;
        }
        return CpuKernelRegister::Instance().RunCpuKernel(context);
    }

    void ExpectPublished(MlaFamily family)
    {
        const auto &header = metadata_.header;
        EXPECT_EQ(header.status, MlaStatus::SUCCESS);
        EXPECT_EQ(header.magic, kMlaMetadataMagic);
        EXPECT_EQ(header.headerBytes, 128U);
        EXPECT_EQ(header.totalBytes, 1920U);
        EXPECT_EQ(header.family, family);
        EXPECT_EQ(header.payloadBytes, MlaPayloadBytes(family));
        EXPECT_EQ(header.payloadAbiId, MlaPayloadAbiId(family));
        EXPECT_EQ(header.aicNum, platform_.aicNum);
        EXPECT_EQ(header.aivNum, platform_.aivNum);
        EXPECT_EQ(header.cvRatio, platform_.aivNum / platform_.aicNum);
        EXPECT_LE(header.originalBlockDim, platform_.aicNum);
        EXPECT_EQ(header.dynamicFlags & ~kMlaDynamicFlagsMask, 0U);
        EXPECT_EQ(header.errorDetail, 0U);
        for (auto reserved : header.reserved) {
            EXPECT_EQ(reserved, 0U);
        }

        auto normalized = config_;
        normalized.seqQPresent = !seqQ_.empty();
        normalized.seqKvPresent = !seqKv_.empty();
        normalized.seqQCount = seqQ_.size();
        normalized.seqKvCount = seqKv_.size();
        ASSERT_EQ(NormalizeMlaScheduleConfig(normalized), MlaStatus::SUCCESS);
        const auto signature = MlaScheduleSignature(normalized, platform_);
        EXPECT_EQ(header.scheduleSignatureLo, signature.lo);
        EXPECT_EQ(header.scheduleSignatureHi, signature.hi);
        EXPECT_EQ(header.platformSignature, MlaPlatformSignature(platform_));

        uint64_t capacity = 0;
        ASSERT_EQ(ComputeMlaWorkspaceCapacity(normalized, platform_, capacity), MlaStatus::SUCCESS);
        EXPECT_LE(header.workspaceNeed, capacity);
        // Static fields belong to the consumer, and must never travel in metadata.
        const auto *fields = family == MlaFamily::ARCH22_MLA ? kMlaNonQuantFields : kMlaLegacyFields;
        const size_t fieldCount = family == MlaFamily::ARCH22_MLA ?
                                      std::size(kMlaNonQuantFields) :
                                      (family == MlaFamily::LEGACY_MLA ? std::size(kMlaLegacyFields) : 0);
        for (size_t i = 0; i < fieldCount; ++i) {
            if (fields[i].role == MlaFieldRole::DYNAMIC) {
                continue;
            }
            for (uint32_t byte = fields[i].offset; byte < fields[i].offset + fields[i].count * fields[i].elementBytes;
                 ++byte) {
                EXPECT_EQ(metadata_.payload[byte], 0U) << fields[i].name << ", byte " << byte;
            }
        }
        for (size_t i = header.payloadBytes; i < sizeof(metadata_.payload); ++i) {
            EXPECT_EQ(metadata_.payload[i], 0U) << "metadata tail byte " << i;
        }
    }

    template <typename T>
    T ReadPayload() const
    {
        T payload{};
        std::memcpy(&payload, metadata_.payload, sizeof(payload));
        return payload;
    }

    MlaScheduleConfig config_;
    MlaPlatformInfo platform_;
    std::vector<int64_t> seqQ_;
    std::vector<int64_t> seqKv_;
    alignas(64) MlaMetadata metadata_;
};

TEST_F(FlashMlaMetadataAicpuTest, DenseAbsentSequencesUseStorageLengths)
{
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 4U);
    EXPECT_EQ(metadata_.header.maxKv, 4096U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 64U);
    const auto payload = ReadPayload<MlaNonQuantTiling>();
    EXPECT_EQ(payload.baseParams.s1Size, 4U);
    EXPECT_EQ(payload.baseParams.s2Size, 4096U);
    EXPECT_EQ(payload.baseParams.needInit, 0U);
    EXPECT_EQ(payload.baseParams.usedCoreNum, metadata_.header.originalBlockDim);
    EXPECT_GT(metadata_.header.originalBlockDim, 0U);
    EXPECT_GT(metadata_.header.workspaceNeed, 0U);
}

TEST_F(FlashMlaMetadataAicpuTest, DenseActualLengthsPublishStatisticsAndInitialization)
{
    seqQ_ = {1, 2, 0, 4};
    seqKv_ = {64, 128, 0, 4096};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 4U);
    EXPECT_EQ(metadata_.header.maxKv, 4096U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 21U);
    EXPECT_EQ(ReadPayload<MlaNonQuantTiling>().baseParams.needInit, 1U);
}

TEST_F(FlashMlaMetadataAicpuTest, OneElementSequencesBroadcastAcrossBatch)
{
    seqQ_ = {2};
    seqKv_ = {1024};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxKv, 1024U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 16U);
    const auto broadcast = metadata_;

    seqQ_.assign(config_.batchSize, 2);
    seqKv_.assign(config_.batchSize, 1024);
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    EXPECT_EQ(std::memcmp(broadcast.payload, metadata_.payload, sizeof(metadata_.payload)), 0);
    EXPECT_EQ(broadcast.header.workspaceNeed, metadata_.header.workspaceNeed);
}

TEST_F(FlashMlaMetadataAicpuTest, DensePaddedDescriptorsIgnoreValuesPastLogicalBatch)
{
    seqQ_ = {1, 2, 0, 4, std::numeric_limits<int64_t>::min()};
    seqKv_ = {64, 128, 0, 4096, std::numeric_limits<int64_t>::max()};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 21U);
    EXPECT_EQ(metadata_.header.maxKv, 4096U);
}

TEST_F(FlashMlaMetadataAicpuTest, ZeroActualKvPublishesInitializationForNonemptyStorage)
{
    seqKv_ = {0};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxKv, 0U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 64U);
    EXPECT_EQ(ReadPayload<MlaNonQuantTiling>().baseParams.needInit, 1U);
}

TEST_F(FlashMlaMetadataAicpuTest, EmptyAndMissingOptionalSequencesHaveTheSameRepresentation)
{
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    const auto empty = metadata_;
    auto scalarInputs = BuildNode();
    scalarInputs->MutableInputs(0)->GetTensorShape()->SetDimSizes({});
    scalarInputs->MutableInputs(1)->GetTensorShape()->SetDimSizes({});
    ASSERT_EQ(Run(scalarInputs), KERNEL_STATUS_OK);
    EXPECT_EQ(std::memcmp(&empty, &metadata_, sizeof(metadata_)), 0);
    ASSERT_EQ(Run(BuildNode("", false)), KERNEL_STATUS_OK);
    EXPECT_EQ(std::memcmp(&empty, &metadata_, sizeof(metadata_)), 0);
}

TEST_F(FlashMlaMetadataAicpuTest, TndCumulativeLengthsIncludeEmptyBatches)
{
    UseTnd();
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 4U);
    EXPECT_EQ(metadata_.header.maxKv, 128U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 21U);
    EXPECT_EQ(ReadPayload<MlaNonQuantTiling>().baseParams.s2Size, 128U);
}

TEST_F(FlashMlaMetadataAicpuTest, TndQueryWithPagedKvUsesAbsoluteKvLengths)
{
    config_.layoutQ = MlaLayout::TND;
    config_.layoutKv = MlaLayout::NZ;
    config_.blockSize = 128;
    seqQ_ = {1, 3, 3, 4};
    seqKv_ = {1024, 128, 0, 2048};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 2U);
    EXPECT_EQ(metadata_.header.maxKv, 2048U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 6U);
}

TEST_F(FlashMlaMetadataAicpuTest, PagedKvRoundsSchedulingExtentToWholeBlocks)
{
    config_.blockSize = 112;
    config_.kvSeqSize = 7168;
    seqKv_ = {1, 113, 224, 337};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxKv, 337U);
    const auto payload = ReadPayload<MlaNonQuantTiling>();
    EXPECT_EQ(payload.baseParams.s2Size, 448U);
    EXPECT_EQ(payload.pageAttenParams.blockSize, 0U);
    EXPECT_EQ(payload.pageAttenParams.maxBlockNumPerBatch, 0U);
    EXPECT_EQ(payload.innerSplitParams.s2BaseSize % 112U, 0U);
}

TEST_F(FlashMlaMetadataAicpuTest, LongDecodePublishesFlashDecodeSplit)
{
    config_.batchSize = 1;
    config_.qSeqSize = 1;
    config_.kvSeqSize = 16384;
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    const auto payload = ReadPayload<MlaNonQuantTiling>();
    EXPECT_EQ(metadata_.header.dynamicFlags, kMlaFlagFd);
    EXPECT_GT(payload.fdParams.numOfFdHead, 0U);
    EXPECT_LE(payload.fdParams.numOfFdHead, platform_.aicNum);
    EXPECT_GT(*std::max_element(std::begin(payload.fdParams.s2SplitNumOfFdHead),
                                std::end(payload.fdParams.s2SplitNumOfFdHead)),
              1U);
}

class FlashMlaMetadataCoreRatioTest : public FlashMlaMetadataAicpuTest, public testing::WithParamInterface<uint32_t> {};

TEST_P(FlashMlaMetadataCoreRatioTest, FlashDecodeDoesNotReadOrPublishInactiveVectorCores)
{
    platform_.aivNum = platform_.aicNum * GetParam();
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    const auto payload = ReadPayload<MlaNonQuantTiling>();
    ASSERT_EQ(metadata_.header.dynamicFlags, kMlaFlagFd);
    const auto activeVectors = metadata_.header.originalBlockDim * GetParam();
    ASSERT_LE(activeVectors, std::size(payload.fdParams.gS1IdxEndOfFdHead));
    for (size_t i = activeVectors; i < std::size(payload.fdParams.gS1IdxEndOfFdHead); ++i) {
        EXPECT_EQ(payload.fdParams.gS1IdxEndOfFdHead[i], 0U);
        EXPECT_EQ(payload.fdParams.gS1IdxEndOfFdHeadSplit[i], 0U);
    }
}

INSTANTIATE_TEST_SUITE_P(CvRatios, FlashMlaMetadataCoreRatioTest, testing::Values(1U, 2U));

TEST_F(FlashMlaMetadataAicpuTest, Bf16PublishesNonQuantizedProtocol)
{
    config_.queryDataType = kMlaBf16;
    config_.layoutQ = MlaLayout::BNSD;
    config_.layoutKv = MlaLayout::BNSD;
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 64U);
}

TEST_F(FlashMlaMetadataAicpuTest, OutputSuffixesPreserveSchedulingBytes)
{
    for (auto layout : {MlaLayout::BSH, MlaLayout::BSND, MlaLayout::BNSD, MlaLayout::TND}) {
        config_.layoutQ = config_.layoutKv = layout;
        seqQ_.clear();
        seqKv_.clear();
        config_.qSeqSize = 4;
        config_.kvSeqSize = 4096;
        if (layout == MlaLayout::TND) {
            UseTnd();
        }
        auto node = BuildNode();
        ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
        const auto withoutSuffix = metadata_;
        const auto inputLayer = node->Attrs().at("input_layer")->GetString();
        node->Attrs().at("input_layer")->SetString(inputLayer + (layout == MlaLayout::TND ? "_NTD" : "_NBSD"));
        ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
        EXPECT_EQ(std::memcmp(&metadata_, &withoutSuffix, sizeof(metadata_)), 0);
    }
    UseQuantizedDecode();
    seqQ_.clear();
    config_.kvSeqSize = 4096;
    auto node = BuildNode();
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    const auto quantized = metadata_;
    node->Attrs().at("input_layer")->SetString("BSND_NBSD");
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    EXPECT_EQ(std::memcmp(&metadata_, &quantized, sizeof(metadata_)), 0);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantModeSelectsNonQuantOrInt8WithoutKvLayoutAttribute)
{
    UseQuantizedDecode();
    auto node = BuildNode();
    node->Attrs().at("quant_mode")->SetInt(0);
    config_.queryDataType = kMlaFp16;
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    const auto nonquant = metadata_;
    config_.queryDataType = kMlaBf16;
    ExpectPublished(MlaFamily::ARCH22_MLA); // Same metadata also matches BF16.
    node->Attrs().at("quant_mode")->SetInt(1);
    config_.queryDataType = kMlaInt8;
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_NE(metadata_.header.scheduleSignatureLo, nonquant.header.scheduleSignatureLo);
    EXPECT_NE(metadata_.header.scheduleSignatureHi, nonquant.header.scheduleSignatureHi);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantizedDecodePublishesLegacyProtocol)
{
    UseQuantizedDecode();
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 1U);
    EXPECT_EQ(metadata_.header.maxKv, 1024U);
    const auto payload = ReadPayload<MlaLegacyTiling>();
    EXPECT_EQ(payload.baseParams.qSeqSize, 1U);
    EXPECT_EQ(payload.increFlashAttentionSingleCoreParams.usedCoreNum, metadata_.header.originalBlockDim);
    EXPECT_GT(metadata_.header.originalBlockDim, 0U);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantizedDenseQueryValuesAndItemCountAreIgnored)
{
    UseQuantizedDecode();
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    const auto absentQuery = metadata_;
    seqQ_ = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 1U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, absentQuery.header.qSeqSquareSum);
    EXPECT_EQ(std::memcmp(absentQuery.payload, metadata_.payload, sizeof(metadata_.payload)), 0);
    EXPECT_NE(metadata_.header.scheduleSignatureLo, absentQuery.header.scheduleSignatureLo);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantizedTndAcceptsPerBatchQueryLimitAndUsesBalance)
{
    UseQuantizedDecode();
    config_.batchSize = 2;
    config_.qSeqSize = 64;
    config_.layoutQ = MlaLayout::TND;
    config_.sparseMode = 3;
    config_.hasAttenMask = 1;
    seqQ_ = {32, 64};
    seqKv_ = {4096, 2048};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 32U);
    EXPECT_EQ(metadata_.header.maxKv, 4096U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 2048U);
    EXPECT_NE(metadata_.header.dynamicFlags & kMlaFlagBalance, 0U);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantizedUnequalKvLengthsEnableBalance)
{
    UseQuantizedDecode();
    seqKv_ = {1, 256, 512, 2048};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxKv, 2048U);
    EXPECT_NE(metadata_.header.dynamicFlags & kMlaFlagBalance, 0U);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantizedPagedKvBroadcastMatchesExpandedLengths)
{
    UseQuantizedDecode();
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    const auto broadcast = metadata_;
    seqKv_.assign(config_.batchSize, 1024);
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    EXPECT_EQ(std::memcmp(broadcast.payload, metadata_.payload, sizeof(metadata_.payload)), 0);
    EXPECT_EQ(broadcast.header.workspaceNeed, metadata_.header.workspaceNeed);
}

struct QuantizedBalancedDecodeCase {
    uint32_t batch;
    uint32_t heads;
    uint32_t query;
    uint32_t kv;
    uint32_t cvRatio;
};

class FlashMlaMetadataBalancedDecodeTest : public FlashMlaMetadataAicpuTest,
                                           public testing::WithParamInterface<QuantizedBalancedDecodeCase> {};

TEST_P(FlashMlaMetadataBalancedDecodeTest, QuantizedMultiTokenDecodePublishesBalancedReductionTasks)
{
    const auto &testCase = GetParam();
    UseQuantizedDecode();
    config_.batchSize = testCase.batch;
    config_.numHeads = testCase.heads;
    config_.qSeqSize = testCase.query;
    config_.kvSeqSize = testCase.kv;
    config_.sparseMode = 3;
    config_.hasAttenMask = 1;
    platform_.aivNum = platform_.aicNum * testCase.cvRatio;
    seqKv_.assign(testCase.batch, testCase.kv);
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxQ, testCase.query);
    EXPECT_EQ(metadata_.header.maxKv, testCase.kv);
    EXPECT_EQ(metadata_.header.dynamicFlags, kMlaFlagFd | kMlaFlagBalance);
    const auto payload = ReadPayload<MlaLegacyTiling>();
    EXPECT_EQ(payload.tndSplitCoreParams.FdBalanceFlag, 1U);
    EXPECT_GT(payload.tndSplitCoreParams.tndFDCoreArrLen, 0U);
    EXPECT_LE(payload.tndSplitCoreParams.tndFDCoreArrLen, platform_.aicNum);
    EXPECT_GT(payload.tndSplitCoreParams.usedVecNumOfFd, 0U);
    EXPECT_LE(payload.tndSplitCoreParams.usedVecNumOfFd, metadata_.header.originalBlockDim * testCase.cvRatio);
    for (uint32_t i = 0; i < payload.tndSplitCoreParams.tndFDCoreArrLen; ++i) {
        EXPECT_LT(payload.tndSplitCoreParams.balanceFDCoreBArr[i], testCase.batch);
        EXPECT_GT(payload.tndSplitCoreParams.balanceFDCoreKVSplitArr[i], 1U);
        EXPECT_LE(payload.tndSplitCoreParams.balanceFDCoreKVSplitArr[i], platform_.aicNum);
        EXPECT_GT(payload.tndSplitCoreParams.gS1SplitNumOfFdHeadMla[i], 0U);
    }
}

INSTANTIATE_TEST_SUITE_P(MultiTokenDecode, FlashMlaMetadataBalancedDecodeTest,
                         testing::Values(QuantizedBalancedDecodeCase{1, 8, 4, 4096, 1},
                                         QuantizedBalancedDecodeCase{2, 8, 4, 5120, 2},
                                         QuantizedBalancedDecodeCase{4, 64, 3, 4096, 2},
                                         QuantizedBalancedDecodeCase{1, 128, 2, 65536, 2}));

class FlashMlaMetadataShortDecodeTest : public FlashMlaMetadataAicpuTest,
                                        public testing::WithParamInterface<uint32_t> {};

TEST_P(FlashMlaMetadataShortDecodeTest, ShortKvDecodeUsesTwoInnerIterations)
{
    UseQuantizedDecode();
    config_.batchSize = 1;
    config_.numHeads = 1;
    config_.kvSeqSize = GetParam();
    seqKv_ = {GetParam()};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxKv, GetParam());
    EXPECT_EQ(metadata_.header.dynamicFlags, kMlaFlagFd);
    const auto payload = ReadPayload<MlaLegacyTiling>();
    EXPECT_EQ(payload.splitKVParams.s2, 1U);
    EXPECT_EQ(payload.increFlashAttentionSingleCoreParams.singleProcessSInnerSize, GetParam() == 256U ? 128U : 256U);
}

INSTANTIATE_TEST_SUITE_P(InnerKvBoundary, FlashMlaMetadataShortDecodeTest, testing::Values(256U, 384U));

TEST_F(FlashMlaMetadataAicpuTest, QuantizedMaximumBatchPublishesMonotonicCoreBoundaries)
{
    UseQuantizedDecode();
    config_.batchSize = 65536;
    config_.numHeads = 128;
    config_.kvSeqSize = 512;
    seqKv_ = {512};
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.dynamicFlags, 0U);
    EXPECT_EQ(metadata_.header.maxQ, 1U);
    EXPECT_EQ(metadata_.header.maxKv, 512U);
    const auto payload = ReadPayload<MlaLegacyTiling>();
    const auto *boundaries = payload.increFlashAttentionCoreParams.coreSidxEnd;
    EXPECT_EQ(boundaries[0], 0U);
    for (uint32_t i = 1; i < metadata_.header.originalBlockDim; ++i) {
        EXPECT_GT(boundaries[i], boundaries[i - 1]);
        EXPECT_LT(boundaries[i], config_.batchSize);
    }
    EXPECT_EQ(boundaries[metadata_.header.originalBlockDim], config_.batchSize);
}

TEST_F(FlashMlaMetadataAicpuTest, QuantizedTndTreeMaskAllowsEmptyKvButRejectsShortNonemptyKv)
{
    UseQuantizedDecode();
    config_.layoutQ = MlaLayout::TND;
    config_.batchSize = 3;
    config_.qSeqSize = 4;
    config_.sparseMode = 9;
    config_.hasAttenMask = 1;
    seqQ_ = {1, 3, 4};
    seqKv_ = {0, 2, 4};
    auto node = BuildNode();
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::LEGACY_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 2U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 6U);
    seqKv_[1] = 1;
    ASSERT_EQ(Run(node), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::INVALID_SEQ);
    EXPECT_EQ(metadata_.header.workspaceNeed, 0U);
}

class FlashMlaMetadataWindowTest : public FlashMlaMetadataAicpuTest, public testing::WithParamInterface<MlaLayout> {};

TEST_P(FlashMlaMetadataWindowTest, SlidingWindowChangesAssignmentWithoutChangingSequenceStatistics)
{
    config_.batchSize = 3;
    config_.qSeqSize = 257;
    config_.kvSeqSize = 4096;
    config_.layoutQ = config_.layoutKv = GetParam();
    if (GetParam() == MlaLayout::TND) {
        config_.qSeqSize *= config_.batchSize;
        config_.kvSeqSize *= config_.batchSize;
        seqQ_ = {257, 514, 771};
        seqKv_ = {4096, 8192, 12288};
    }
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    const auto fullAttention = metadata_;
    config_.sparseMode = 4;
    config_.hasAttenMask = 1;
    config_.preTokens = 16;
    config_.nextTokens = 16;
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::ARCH22_MLA);
    EXPECT_EQ(metadata_.header.maxQ, 257U);
    EXPECT_EQ(metadata_.header.maxKv, 4096U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 3U * 257U * 257U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, fullAttention.header.qSeqSquareSum);
    EXPECT_NE(std::memcmp(fullAttention.payload, metadata_.payload, sizeof(metadata_.payload)), 0);
}

INSTANTIATE_TEST_SUITE_P(MaskedLayouts, FlashMlaMetadataWindowTest,
                         testing::Values(MlaLayout::BSH, MlaLayout::BNSD, MlaLayout::TND));

TEST_F(FlashMlaMetadataAicpuTest, GroupedQuerySizeOverflowPublishesFailureBeforeSplitting)
{
    config_.batchSize = 1;
    config_.numHeads = 128;
    config_.qSeqSize = UINT32_MAX;
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::OVERFLOW);
    EXPECT_EQ(metadata_.header.workspaceNeed, 0U);
}

class FlashMlaMetadataEmptyTest : public FlashMlaMetadataAicpuTest, public testing::WithParamInterface<std::string> {};

TEST_P(FlashMlaMetadataEmptyTest, EmptyStoragePublishesValidEmptyMetadata)
{
    if (GetParam() == "batch") {
        config_.batchSize = 0;
    } else if (GetParam() == "query") {
        config_.qSeqSize = 0;
    } else {
        config_.kvSeqSize = 0;
    }
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_OK);
    ExpectPublished(MlaFamily::NO_SCHEDULE);
    EXPECT_EQ(metadata_.header.workspaceNeed, 0U);
    EXPECT_EQ(metadata_.header.originalBlockDim, 0U);
    EXPECT_EQ(metadata_.header.dynamicFlags, 0U);
    EXPECT_EQ(metadata_.header.maxQ, 0U);
    EXPECT_EQ(metadata_.header.maxKv, 0U);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 0U);
}

INSTANTIATE_TEST_SUITE_P(EmptyDimensions, FlashMlaMetadataEmptyTest, testing::Values("batch", "query", "kv"));

TEST_F(FlashMlaMetadataAicpuTest, RegenerationReadsNewValuesWithoutChangingDescriptorSignature)
{
    seqQ_ = {1, 2, 0, 4};
    seqKv_ = {64, 128, 0, 4096};
    auto node = BuildNode();
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    const auto previous = metadata_;
    seqQ_[3] = 3;
    seqKv_[3] = 512;
    ASSERT_EQ(Run(node), KERNEL_STATUS_OK);
    EXPECT_EQ(metadata_.header.qSeqSquareSum, 14U);
    EXPECT_EQ(metadata_.header.maxKv, 512U);
    EXPECT_EQ(metadata_.header.scheduleSignatureLo, previous.header.scheduleSignatureLo);
    EXPECT_EQ(metadata_.header.scheduleSignatureHi, previous.header.scheduleSignatureHi);
    EXPECT_NE(std::memcmp(previous.payload, metadata_.payload, sizeof(metadata_.payload)), 0);
}

class FlashMlaMetadataMissingAttrTest : public FlashMlaMetadataAicpuTest,
                                        public testing::WithParamInterface<std::string> {};

TEST_P(FlashMlaMetadataMissingAttrTest, MissingAttributeInvalidatesPreviousMetadata)
{
    ASSERT_EQ(Run(BuildNode(GetParam())), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::INVALID);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes, FlashMlaMetadataMissingAttrTest,
                         testing::Values("num_heads", "batch_size", "q_seq_size", "kv_seq_size", "input_layer",
                                         "block_size", "sparse_mode", "pre_tokens", "next_tokens", "has_atten_mask",
                                         "quant_mode", "aic_core_num", "aiv_core_num"));

struct InvalidIntegerAttr {
    const char *name;
    int64_t value;
    MlaStatus metadataStatus;
};

class FlashMlaMetadataIntegerAttrTest : public FlashMlaMetadataAicpuTest,
                                        public testing::WithParamInterface<InvalidIntegerAttr> {};

TEST_P(FlashMlaMetadataIntegerAttrTest, RejectsOutOfRangeSchedulingAttributes)
{
    const auto &attr = GetParam();
    auto node = BuildNode();
    node->Attrs().at(attr.name)->SetInt(attr.value);
    ASSERT_EQ(Run(node), KERNEL_STATUS_PARAM_INVALID) << attr.name;
    EXPECT_EQ(metadata_.header.status, attr.metadataStatus) << attr.name;
}

INSTANTIATE_TEST_SUITE_P(
    InvalidIntegers, FlashMlaMetadataIntegerAttrTest,
    testing::Values(InvalidIntegerAttr{"num_heads", -1, MlaStatus::INVALID},
                    InvalidIntegerAttr{"num_heads", INT64_C(1) << 32, MlaStatus::INVALID},
                    InvalidIntegerAttr{"q_seq_size", -1, MlaStatus::INVALID},
                    InvalidIntegerAttr{"num_heads", 0, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"num_heads", 129, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"batch_size", 65537, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"q_seq_size", INT64_C(1) << 32, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"kv_seq_size", INT64_C(1) << 31, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"aic_core_num", 0, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"aic_core_num", 26, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"aiv_core_num", 47, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"sparse_mode", 3, MlaStatus::INVALID_ARGUMENT},
                    InvalidIntegerAttr{"quant_mode", -1, MlaStatus::INVALID},
                    InvalidIntegerAttr{"quant_mode", 2, MlaStatus::INVALID},
                    InvalidIntegerAttr{"quant_mode", 27, MlaStatus::INVALID}));

TEST_F(FlashMlaMetadataAicpuTest, InvalidLayoutInvalidatesPreviousMetadata)
{
    auto node = BuildNode();
    node->Attrs().at("input_layer")->SetString("BSND_NTD");
    ASSERT_EQ(Run(node), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::INVALID);
}

TEST_F(FlashMlaMetadataAicpuTest, InvalidWindowFailsBeforePublishingPatch)
{
    config_.hasAttenMask = 1;
    config_.preTokens = -5;
    config_.nextTokens = 4;
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::INVALID);
}

enum class SequenceDescriptorError {
    DTYPE,
    RANK,
    NEGATIVE_SIZE,
    NULL_DATA,
    SCALAR_DATA
};
class FlashMlaMetadataSeqDescriptorTest
    : public FlashMlaMetadataAicpuTest,
      public testing::WithParamInterface<std::pair<uint32_t, SequenceDescriptorError>> {};

TEST_P(FlashMlaMetadataSeqDescriptorTest, InvalidDescriptorClearsSuccessStatus)
{
    seqQ_ = {4};
    seqKv_ = {4096};
    auto node = BuildNode();
    const auto tensor = node->MutableInputs(GetParam().first);
    switch (GetParam().second) {
        case SequenceDescriptorError::DTYPE:
            tensor->SetDataType(DT_INT32);
            break;
        case SequenceDescriptorError::RANK:
            tensor->GetTensorShape()->SetDimSizes({1, 1});
            break;
        case SequenceDescriptorError::NEGATIVE_SIZE:
            tensor->GetTensorShape()->SetDimSizes({-1});
            break;
        case SequenceDescriptorError::NULL_DATA:
            tensor->SetData(nullptr);
            break;
        case SequenceDescriptorError::SCALAR_DATA:
            tensor->GetTensorShape()->SetDimSizes({});
            break;
    }
    ASSERT_EQ(Run(node), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::INVALID);
}

INSTANTIATE_TEST_SUITE_P(SequenceDescriptors, FlashMlaMetadataSeqDescriptorTest,
                         testing::Values(std::make_pair(0U, SequenceDescriptorError::DTYPE),
                                         std::make_pair(1U, SequenceDescriptorError::DTYPE),
                                         std::make_pair(0U, SequenceDescriptorError::RANK),
                                         std::make_pair(1U, SequenceDescriptorError::RANK),
                                         std::make_pair(0U, SequenceDescriptorError::NEGATIVE_SIZE),
                                         std::make_pair(1U, SequenceDescriptorError::NEGATIVE_SIZE),
                                         std::make_pair(0U, SequenceDescriptorError::NULL_DATA),
                                         std::make_pair(1U, SequenceDescriptorError::NULL_DATA),
                                         std::make_pair(0U, SequenceDescriptorError::SCALAR_DATA),
                                         std::make_pair(1U, SequenceDescriptorError::SCALAR_DATA)));

enum class OutputDescriptorError {
    MISSING,
    DTYPE,
    TOO_SMALL,
    TOO_LARGE,
    RANK,
    NULL_DATA,
    UNALIGNED
};
class FlashMlaMetadataOutputTest : public FlashMlaMetadataAicpuTest,
                                   public testing::WithParamInterface<OutputDescriptorError> {};

TEST_P(FlashMlaMetadataOutputTest, InvalidOutputIsRejectedBeforeWriting)
{
    auto node = BuildNode("", true, GetParam() != OutputDescriptorError::MISSING);
    if (GetParam() != OutputDescriptorError::MISSING) {
        const auto tensor = node->MutableOutputs(0);
        switch (GetParam()) {
            case OutputDescriptorError::DTYPE:
                tensor->SetDataType(DT_INT64);
                break;
            case OutputDescriptorError::TOO_SMALL:
                tensor->GetTensorShape()->SetDimSizes({479});
                break;
            case OutputDescriptorError::TOO_LARGE:
                tensor->GetTensorShape()->SetDimSizes({481});
                break;
            case OutputDescriptorError::RANK:
                tensor->GetTensorShape()->SetDimSizes({1, 480});
                break;
            case OutputDescriptorError::NULL_DATA:
                tensor->SetData(nullptr);
                break;
            case OutputDescriptorError::UNALIGNED:
                tensor->SetData(reinterpret_cast<uint8_t *>(&metadata_) + sizeof(int32_t));
                break;
            default:
                break;
        }
    }
    const auto previous = metadata_;
    ASSERT_EQ(Run(node), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(std::memcmp(&previous, &metadata_, sizeof(metadata_)), 0);
}

INSTANTIATE_TEST_SUITE_P(OutputDescriptors, FlashMlaMetadataOutputTest,
                         testing::Values(OutputDescriptorError::MISSING, OutputDescriptorError::DTYPE,
                                         OutputDescriptorError::TOO_SMALL, OutputDescriptorError::TOO_LARGE,
                                         OutputDescriptorError::RANK, OutputDescriptorError::NULL_DATA,
                                         OutputDescriptorError::UNALIGNED));

enum class SequenceValueError {
    NEGATIVE_Q,
    NEGATIVE_KV,
    Q_OVER_STORAGE,
    KV_OVER_STORAGE,
    Q_INT64_MIN,
    KV_INT64_MAX,
    TOO_FEW_Q,
    TOO_FEW_KV,
    TND_MISSING_Q,
    TND_MISSING_KV,
    TND_Q_DECREASING,
    TND_KV_DECREASING,
    TND_Q_EXTRA,
    TND_KV_EXTRA,
    TND_Q_TOTAL,
    TND_KV_TOTAL,
    PA_MISSING_KV,
    INT8_TND_Q_OVER_32
};

class FlashMlaMetadataSequenceValueTest : public FlashMlaMetadataAicpuTest,
                                          public testing::WithParamInterface<SequenceValueError> {};

TEST_P(FlashMlaMetadataSequenceValueTest, InvalidValuesPublishFailureAndEraseStalePayload)
{
    if (GetParam() >= SequenceValueError::TND_MISSING_Q && GetParam() <= SequenceValueError::TND_KV_TOTAL) {
        UseTnd();
    }
    switch (GetParam()) {
        case SequenceValueError::NEGATIVE_Q:
            seqQ_ = {-1};
            break;
        case SequenceValueError::NEGATIVE_KV:
            seqKv_ = {-1};
            break;
        case SequenceValueError::Q_OVER_STORAGE:
            seqQ_ = {5};
            break;
        case SequenceValueError::KV_OVER_STORAGE:
            seqKv_ = {4097};
            break;
        case SequenceValueError::Q_INT64_MIN:
            seqQ_ = {std::numeric_limits<int64_t>::min()};
            break;
        case SequenceValueError::KV_INT64_MAX:
            seqKv_ = {std::numeric_limits<int64_t>::max()};
            break;
        case SequenceValueError::TOO_FEW_Q:
            seqQ_ = {1, 1};
            break;
        case SequenceValueError::TOO_FEW_KV:
            seqKv_ = {1, 1};
            break;
        case SequenceValueError::TND_MISSING_Q:
            seqQ_.clear();
            break;
        case SequenceValueError::TND_MISSING_KV:
            seqKv_.clear();
            break;
        case SequenceValueError::TND_Q_DECREASING:
            seqQ_ = {0, 3, 2, 7};
            break;
        case SequenceValueError::TND_KV_DECREASING:
            seqKv_ = {0, 96, 32, 224};
            break;
        case SequenceValueError::TND_Q_EXTRA:
            seqQ_.push_back(7);
            break;
        case SequenceValueError::TND_KV_EXTRA:
            seqKv_.push_back(224);
            break;
        case SequenceValueError::TND_Q_TOTAL:
            seqQ_.back() = 6;
            break;
        case SequenceValueError::TND_KV_TOTAL:
            seqKv_.back() = 223;
            break;
        case SequenceValueError::PA_MISSING_KV:
            config_.blockSize = 128;
            break;
        case SequenceValueError::INT8_TND_Q_OVER_32:
            UseQuantizedDecode();
            config_.layoutQ = MlaLayout::TND;
            config_.batchSize = 1;
            config_.qSeqSize = 33;
            config_.sparseMode = 3;
            config_.hasAttenMask = 1;
            seqQ_ = {33};
            break;
    }
    ASSERT_EQ(Run(BuildNode()), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(metadata_.header.status, MlaStatus::INVALID_SEQ);
    EXPECT_EQ(metadata_.header.magic, kMlaMetadataMagic);
    EXPECT_EQ(metadata_.header.workspaceNeed, 0U);
    for (auto byte : metadata_.payload) {
        EXPECT_EQ(byte, 0U);
    }
}

INSTANTIATE_TEST_SUITE_P(SequenceValues, FlashMlaMetadataSequenceValueTest,
                         testing::Values(SequenceValueError::NEGATIVE_Q, SequenceValueError::NEGATIVE_KV,
                                         SequenceValueError::Q_OVER_STORAGE, SequenceValueError::KV_OVER_STORAGE,
                                         SequenceValueError::Q_INT64_MIN, SequenceValueError::KV_INT64_MAX,
                                         SequenceValueError::TOO_FEW_Q, SequenceValueError::TOO_FEW_KV,
                                         SequenceValueError::TND_MISSING_Q, SequenceValueError::TND_MISSING_KV,
                                         SequenceValueError::TND_Q_DECREASING, SequenceValueError::TND_KV_DECREASING,
                                         SequenceValueError::TND_Q_EXTRA, SequenceValueError::TND_KV_EXTRA,
                                         SequenceValueError::TND_Q_TOTAL, SequenceValueError::TND_KV_TOTAL,
                                         SequenceValueError::PA_MISSING_KV, SequenceValueError::INT8_TND_Q_OVER_32));
} // namespace
