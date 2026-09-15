/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <functional>
#include <limits>
#include <gtest/gtest.h>

#include "../../../op_kernel/arch22/flash_mla_tiling_adapter.h"

namespace {
using namespace FlashMla;

class FlashMlaMetadataProtocolTest : public testing::Test {
protected:
    void SetUp() override
    {
        Reset(MlaFamily::ARCH22_MLA);
    }

    void Reset(MlaFamily family)
    {
        MlaScheduleConfig config;
        config.numHeads = 8;
        config.batchSize = 2;
        config.qSeqSize = 2;
        config.kvSeqSize = 128;
        MlaPlatformInfo platform;
        platform.aicNum = 2;
        platform.aivNum = 4;
        InitializeMlaPatch(config, platform, family, patch_);
        patch_.header.status = MlaStatus::SUCCESS;
        patch_.header.workspaceNeed = 2560;
        launch_ = {};
        launch_.family = family;
        launch_.aicNum = 2;
        launch_.aivNum = 4;
        launch_.cvRatio = 2;
        launch_.physicalBlockDim = 4;
        launch_.queryDataType = family == MlaFamily::LEGACY_MLA ? kMlaInt8 : kMlaFp16;
        launch_.payloadAbiId = patch_.header.payloadAbiId;
        launch_.platformSignature = patch_.header.platformSignature;
        launch_.scheduleSignatureLo = patch_.header.scheduleSignatureLo;
        launch_.scheduleSignatureHi = patch_.header.scheduleSignatureHi;
        launch_.sysWorkspaceSize = 4096;
        launch_.workspaceCap = 6656;
        launch_.qStorageSize = 2;
        launch_.kvStorageSize = 128;
        if (family == MlaFamily::LEGACY_MLA) {
            launch_.pageAttention = 1;
            launch_.layoutKv = MlaLayout::NZ;
        }
        UpdateLaunchKey();
        ASSERT_EQ(WriteMlaMetadata(&metadata_, sizeof(metadata_), patch_), MlaStatus::SUCCESS);
    }

    void UpdateLaunchKey()
    {
        launch_.staticLaunchKey =
            MakeMlaStaticLaunchKey(launch_.family, launch_.queryDataType, launch_.layoutQ, launch_.layoutKv,
                                   launch_.pageAttention != 0, launch_.treeMask != 0, launch_.cvRatio);
    }

    MlaValuePatch patch_{};
    alignas(64) MlaMetadata metadata_{};
    MlaStaticLaunchInfo launch_{};
};

TEST_F(FlashMlaMetadataProtocolTest, PublishedMetadataSupportsEveryFamily)
{
    for (auto family : {MlaFamily::NO_SCHEDULE, MlaFamily::ARCH22_MLA, MlaFamily::LEGACY_MLA}) {
        SCOPED_TRACE(static_cast<uint32_t>(family));
        Reset(family);
        EXPECT_EQ(metadata_.header.magic, 0x34414c4dU);
        EXPECT_EQ(metadata_.header.headerBytes, 128U);
        EXPECT_EQ(metadata_.header.totalBytes, 1920U);
        EXPECT_EQ(ValidateMlaMetadata(metadata_, launch_), MlaStatus::SUCCESS);
    }
}

struct HeaderMutation {
    const char *name;
    std::function<void(MlaMetadataHeader &)> mutate;
    MlaStatus expected;
};

class FlashMlaMetadataHeaderTest : public FlashMlaMetadataProtocolTest,
                                   public testing::WithParamInterface<HeaderMutation> {};

TEST_P(FlashMlaMetadataHeaderTest, RejectsInvalidHeaderBeforeApplyingPayload)
{
    MlaNonQuantTiling tiling{};
    tiling.baseParams.s1Size = 17;
    const auto original = tiling;
    GetParam().mutate(metadata_.header);
    EXPECT_EQ(ValidateMlaMetadata(metadata_, launch_), GetParam().expected);
    EXPECT_EQ(ApplyMlaValuePatch(metadata_, launch_, &tiling, sizeof(tiling)), GetParam().expected);
    EXPECT_EQ(std::memcmp(&tiling, &original, sizeof(tiling)), 0);
}

INSTANTIATE_TEST_SUITE_P(
    HeaderContract, FlashMlaMetadataHeaderTest,
    testing::Values(
        HeaderMutation{"Magic", [](auto &h) { h.magic ^= 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"HeaderSize", [](auto &h) { --h.headerBytes; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"TotalSize", [](auto &h) { --h.totalBytes; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"Family", [](auto &h) { h.family = MlaFamily::LEGACY_MLA; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"PayloadSize", [](auto &h) { ++h.payloadBytes; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"PayloadAbi", [](auto &h) { h.payloadAbiId ^= 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"Reserved0", [](auto &h) { h.reserved[0] = 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"Reserved1", [](auto &h) { h.reserved[1] = 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"Reserved2", [](auto &h) { h.reserved[2] = 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"Reserved3", [](auto &h) { h.reserved[3] = 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"ErrorDetail", [](auto &h) { h.errorDetail = 1; }, MlaStatus::ABI_MISMATCH},
        HeaderMutation{"Platform", [](auto &h) { h.platformSignature ^= 1; }, MlaStatus::SIGNATURE_MISMATCH},
        HeaderMutation{"ScheduleLo", [](auto &h) { h.scheduleSignatureLo ^= 1; }, MlaStatus::SIGNATURE_MISMATCH},
        HeaderMutation{"ScheduleHi", [](auto &h) { h.scheduleSignatureHi ^= 1; }, MlaStatus::SIGNATURE_MISMATCH},
        HeaderMutation{"CubeCores", [](auto &h) { ++h.aicNum; }, MlaStatus::SIGNATURE_MISMATCH},
        HeaderMutation{"VectorCores", [](auto &h) { ++h.aivNum; }, MlaStatus::SIGNATURE_MISMATCH},
        HeaderMutation{"CvRatio", [](auto &h) { ++h.cvRatio; }, MlaStatus::SIGNATURE_MISMATCH},
        HeaderMutation{"UnknownFlags", [](auto &h) { h.dynamicFlags = 4; }, MlaStatus::INVALID_PATCH},
        HeaderMutation{"Unpublished", [](auto &h) { h.status = MlaStatus::INVALID; }, MlaStatus::INVALID_PATCH},
        HeaderMutation{"SequenceError", [](auto &h) { h.status = MlaStatus::INVALID_SEQ; }, MlaStatus::INVALID_SEQ},
        HeaderMutation{"WorkspaceTooSmall", [](auto &h) { ++h.workspaceNeed; }, MlaStatus::CAPACITY_EXCEEDED}),
    [](const testing::TestParamInfo<HeaderMutation> &info) { return info.param.name; });

TEST_F(FlashMlaMetadataProtocolTest, NonQuantPatchPreservesStaticShapeScaleAndPageStride)
{
    MlaNonQuantTiling dynamic{};
    dynamic.baseParams.bSize = 999;
    dynamic.baseParams.scaleValue = 9.0F;
    dynamic.baseParams.s1Size = 1;
    dynamic.baseParams.s2Size = 16;
    dynamic.baseParams.usedCoreNum = 2;
    dynamic.baseParams.isEmptyBatchOverHalf = 1;
    dynamic.pageAttenParams.maxBlockNumPerBatch = 1;
    dynamic.workspaceParams.mm1ResSize = 64;
    std::memcpy(patch_.payload, &dynamic, sizeof(dynamic));
    ASSERT_EQ(WriteMlaMetadata(&metadata_, sizeof(metadata_), patch_), MlaStatus::SUCCESS);
    MlaNonQuantTiling tiling{};
    tiling.baseParams.bSize = 2;
    tiling.baseParams.scaleValue = 0.25F;
    tiling.pageAttenParams.blockSize = 16;
    tiling.pageAttenParams.maxBlockNumPerBatch = 8;
    auto expected = tiling;
    expected.baseParams.s1Size = 1;
    expected.baseParams.s2Size = 16;
    expected.baseParams.usedCoreNum = 2;
    expected.baseParams.isEmptyBatchOverHalf = 1;
    expected.workspaceParams.mm1ResSize = 64;
    ASSERT_EQ(ApplyMlaValuePatch(metadata_, launch_, &tiling, sizeof(tiling)), MlaStatus::SUCCESS);
    EXPECT_EQ(std::memcmp(&tiling, &expected, sizeof(tiling)), 0);
}

TEST_F(FlashMlaMetadataProtocolTest, LegacyPatchPreservesStaticPageStrideAndQuantization)
{
    Reset(MlaFamily::LEGACY_MLA);
    MlaLegacyTiling dynamic{};
    dynamic.baseParams.batchSize = 999;
    dynamic.baseParams.maxBlockNumPerBatch = 1;
    dynamic.baseParams.scaleValue = 9.0F;
    dynamic.baseParams.seqSize = 128;
    dynamic.baseParams.qSeqSize = 1;
    dynamic.splitKVParams.s2 = 64;
    dynamic.increFlashAttentionSingleCoreParams.usedCoreNum = 2;
    std::memcpy(patch_.payload, &dynamic, sizeof(dynamic));
    ASSERT_EQ(WriteMlaMetadata(&metadata_, sizeof(metadata_), patch_), MlaStatus::SUCCESS);
    MlaLegacyTiling tiling{};
    tiling.baseParams.batchSize = 2;
    tiling.baseParams.maxBlockNumPerBatch = 8;
    tiling.baseParams.blockSize = 128;
    tiling.baseParams.scaleValue = 0.25F;
    auto expected = tiling;
    expected.baseParams.seqSize = 128;
    expected.baseParams.qSeqSize = 1;
    expected.splitKVParams.s2 = 64;
    expected.increFlashAttentionSingleCoreParams.usedCoreNum = 2;
    ASSERT_EQ(ApplyMlaValuePatch(metadata_, launch_, &tiling, sizeof(tiling)), MlaStatus::SUCCESS);
    EXPECT_EQ(std::memcmp(&tiling, &expected, sizeof(tiling)), 0);
}

TEST_F(FlashMlaMetadataProtocolTest, RejectsNonCanonicalStaticPaddingAndTailBytes)
{
    // bSize, padding after isEmptyBatchOverHalf, and the unused metadata tail must stay zero.
    for (uint32_t offset : {0U, 81U, 83U, 1600U, 1791U}) {
        SCOPED_TRACE(offset);
        metadata_.payload[offset] = 1;
        EXPECT_EQ(ValidateMlaMetadata(metadata_, launch_), MlaStatus::INVALID_PATCH);
        metadata_.payload[offset] = 0;
    }
    EXPECT_EQ(ValidateMlaMetadata(metadata_, launch_), MlaStatus::SUCCESS);
}

TEST_F(FlashMlaMetadataProtocolTest, RejectsShortOrNullTilingAndWorkspaceUnderflow)
{
    MlaNonQuantTiling tiling{};
    EXPECT_EQ(ApplyMlaValuePatch(metadata_, launch_, nullptr, sizeof(tiling)), MlaStatus::CAPACITY_EXCEEDED);
    EXPECT_EQ(ApplyMlaValuePatch(metadata_, launch_, &tiling, sizeof(tiling) - 1), MlaStatus::CAPACITY_EXCEEDED);
    launch_.sysWorkspaceSize = launch_.workspaceCap + 1;
    EXPECT_EQ(ValidateMlaMetadata(metadata_, launch_), MlaStatus::CAPACITY_EXCEEDED);
    EXPECT_EQ(MlaCheckWorkspace(launch_, metadata_.header, 0), MlaStatus::CAPACITY_EXCEEDED);
}

TEST_F(FlashMlaMetadataProtocolTest, RejectsInvalidPublicationBuffersWithoutWriting)
{
    alignas(64) uint8_t storage[kMlaMetadataBytes + 64];
    std::memset(storage, 0x5a, sizeof(storage));
    EXPECT_EQ(WriteMlaMetadata(nullptr, kMlaMetadataBytes, patch_), MlaStatus::CAPACITY_EXCEEDED);
    EXPECT_EQ(WriteMlaMetadata(storage, kMlaMetadataBytes - 1, patch_), MlaStatus::CAPACITY_EXCEEDED);
    EXPECT_EQ(WriteMlaMetadata(storage + 1, kMlaMetadataBytes, patch_), MlaStatus::CAPACITY_EXCEEDED);
    for (auto value : storage) {
        ASSERT_EQ(value, 0x5a);
    }
}

TEST_F(FlashMlaMetadataProtocolTest, LaunchKeyAndLogicalCubeCountMustMatch)
{
    EXPECT_EQ(ValidateMlaLaunch(launch_, 2), MlaStatus::SUCCESS);
    EXPECT_EQ(ValidateMlaLaunch(launch_, 4), MlaStatus::INVALID_PATCH);
    launch_.staticLaunchKey ^= 1;
    EXPECT_EQ(ValidateMlaLaunch(launch_, 2), MlaStatus::INVALID_PATCH);
    UpdateLaunchKey();
    launch_.aivNum = 3;
    EXPECT_EQ(ValidateMlaLaunch(launch_, 2), MlaStatus::INVALID_PATCH);
}

TEST_F(FlashMlaMetadataProtocolTest, NonQuantWorkspaceUsesAllLaunchedCores)
{
    MlaNonQuantTiling tiling{};
    tiling.baseParams.bSize = 2;
    tiling.baseParams.n2Size = 1;
    tiling.baseParams.gSize = 8;
    tiling.baseParams.s1Size = 2;
    tiling.baseParams.s2Size = 128;
    tiling.baseParams.headDim = 512;
    tiling.baseParams.headDimRope = 64;
    tiling.baseParams.usedCoreNum = 1;
    tiling.innerSplitParams.mBaseSize = 128;
    tiling.innerSplitParams.s2BaseSize = 128;
    tiling.outerSplitParams.bN2End[0] = 2;
    tiling.workspaceParams.mm1ResSize = 64;
    tiling.workspaceParams.mm2ResSize = 32;
    uint64_t workspaceLayout = 0;
    ASSERT_EQ(FinalizeMlaTiling(launch_, metadata_.header, tiling, 2, workspaceLayout), MlaStatus::SUCCESS);
    // Two cube cores each have two vector workers, even though only one core has tasks.
    EXPECT_EQ(workspaceLayout, 6656U);
    --launch_.workspaceCap;
    EXPECT_EQ(FinalizeMlaTiling(launch_, metadata_.header, tiling, 2, workspaceLayout), MlaStatus::CAPACITY_EXCEEDED);
    ++launch_.workspaceCap;
    launch_.pageAttention = 1;
    UpdateLaunchKey();
    tiling.pageAttenParams.blockSize = 16;
    tiling.pageAttenParams.maxBlockNumPerBatch = 8;
    metadata_.header.maxKv = 1;
    tiling.baseParams.s2Size = 16;
    EXPECT_EQ(FinalizeMlaTiling(launch_, metadata_.header, tiling, 2, workspaceLayout), MlaStatus::SUCCESS);
    tiling.pageAttenParams.maxBlockNumPerBatch = 1;
    EXPECT_EQ(FinalizeMlaTiling(launch_, metadata_.header, tiling, 2, workspaceLayout), MlaStatus::INVALID_PATCH);
}

TEST_F(FlashMlaMetadataProtocolTest, LegacyWorkspaceUsesInt8AndHalfIntermediateWidths)
{
    Reset(MlaFamily::LEGACY_MLA);
    launch_.qStorageSize = 1;
    launch_.workspaceCap = 5888;
    metadata_.header.workspaceNeed = 1792;
    MlaLegacyTiling tiling{};
    tiling.baseParams.batchSize = 2;
    tiling.baseParams.seqSize = 128;
    tiling.baseParams.qSeqSize = 1;
    tiling.baseParams.blockSize = 128;
    tiling.baseParams.maxBlockNumPerBatch = 1;
    tiling.baseParams.nNumOfQInOneGroup = 8;
    tiling.increFlashAttentionSingleCoreParams.usedCoreNum = 1;
    tiling.increFlashAttentionSingleCoreParams.groupSplitSize = 8;
    tiling.increFlashAttentionSingleCoreParams.s1SplitSize = 1;
    tiling.increFlashAttentionSingleCoreParams.singleProcessSInnerSize = 128;
    tiling.increFlashAttentionSingleCoreTensorSize.mmResUbSize = 64;
    tiling.increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize = 32;
    uint64_t workspaceLayout = 0;
    ASSERT_EQ(FinalizeMlaTiling(launch_, metadata_.header, tiling, 2, workspaceLayout), MlaStatus::SUCCESS);
    EXPECT_EQ(workspaceLayout, 5888U);
    --launch_.workspaceCap;
    EXPECT_EQ(FinalizeMlaTiling(launch_, metadata_.header, tiling, 2, workspaceLayout), MlaStatus::CAPACITY_EXCEEDED);
}

TEST_F(FlashMlaMetadataProtocolTest, WorkspaceArithmeticRejectsOverflow)
{
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    uint64_t value = 0;
    EXPECT_TRUE(MlaCheckedMul(max, 1, value));
    EXPECT_EQ(value, max);
    EXPECT_TRUE(MlaCheckedMul(1, max, value));
    EXPECT_EQ(value, max);
    EXPECT_FALSE(MlaCheckedMul(UINT64_C(1) << 32, UINT64_C(1) << 32, value));
    EXPECT_FALSE(MlaCheckedMul(max, 2, value));
    EXPECT_FALSE(MlaCheckedMul(2, max, value));
    EXPECT_FALSE(MlaCheckedAdd(max, 1, value));
    value = max - 15;
    EXPECT_TRUE(MlaAddWorkspaceRegion(2, 3, 2, value));
    EXPECT_EQ(value, max - 3);
    EXPECT_FALSE(MlaAddWorkspaceRegion(1, 4, 1, value));
    EXPECT_FALSE(MlaAddWorkspaceRegion(max, 2, 1, value));
    EXPECT_FALSE(MlaAddWorkspaceRegion(1, max, 2, value));
}

TEST_F(FlashMlaMetadataProtocolTest, MaskCapacityFollowsActualKvAndPagedRounding)
{
    launch_.maskRank = 3;
    launch_.maskDim0 = 2;
    launch_.maskDim1 = 4;
    launch_.maskDim2 = 17;
    metadata_.header.maxKv = 17;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 128, 1, 0), MlaStatus::SUCCESS);
    --launch_.maskDim2;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 128, 1, 0), MlaStatus::INVALID_PATCH);
    launch_.pageAttention = 1;
    launch_.maskDim2 = 17;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 32, 1, 0), MlaStatus::INVALID_PATCH);
    launch_.maskDim2 = 32;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 32, 1, 0), MlaStatus::SUCCESS);
    launch_.maskDim0 = 1;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 32, 1, 0), MlaStatus::SUCCESS);
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 32, 0, 0), MlaStatus::INVALID_PATCH);
}

TEST_F(FlashMlaMetadataProtocolTest, SparseAndTreeMaskContractsDifferByFamily)
{
    launch_.maskRank = 2;
    launch_.maskDim0 = 2048;
    launch_.maskDim1 = 2048;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 128, 1, 3), MlaStatus::SUCCESS);
    launch_.maskDim1 = 2047;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 128, 1, 3), MlaStatus::INVALID_PATCH);
    launch_.treeMask = 1;
    launch_.maskRank = 1;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 128, 1, 9), MlaStatus::SUCCESS);
    launch_.maskRank = 2;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 4, 128, 1, 9), MlaStatus::INVALID_PATCH);
    launch_.family = MlaFamily::LEGACY_MLA;
    launch_.layoutQ = MlaLayout::TND;
    launch_.maskRank = 1;
    launch_.maskDim0 = 10;
    metadata_.header.qSeqSquareSum = 10; // Packed query lengths [1, 3].
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 3, 128, 1, 9), MlaStatus::SUCCESS);
    --launch_.maskDim0;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 3, 128, 1, 9), MlaStatus::INVALID_PATCH);
    launch_.layoutQ = MlaLayout::BSH;
    launch_.maskRank = 3;
    launch_.maskDim0 = 2;
    launch_.maskDim1 = 3;
    launch_.maskDim2 = 3;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 3, 128, 1, 9), MlaStatus::SUCCESS);
    launch_.maskDim2 = 4;
    EXPECT_EQ(FinalizeMlaMask(launch_, metadata_.header, 2, 3, 128, 1, 9), MlaStatus::INVALID_PATCH);
}

TEST_F(FlashMlaMetadataProtocolTest, NonQuantFdChecksReductionSourcesAndEndpoints)
{
    MlaNonQuantTiling tiling{};
    tiling.baseParams.bSize = 2;
    tiling.baseParams.s1Size = 1;
    tiling.baseParams.gSize = 8;
    tiling.innerSplitParams.mBaseSize = 8;
    tiling.workspaceParams.fdAccumOutSize = 8192;
    tiling.workspaceParams.fdLogSumExpSize = 256;
    tiling.fdParams.numOfFdHead = 1;
    tiling.fdParams.usedVecNumOfFd = 1;
    tiling.fdParams.gS1BaseSizeOfFd = 8;
    tiling.fdParams.s2SplitNumOfFdHead[0] = 2;
    tiling.fdParams.gS1SplitNumOfFdHead[0] = 1;
    tiling.fdParams.gS1LastPartSizeOfFdHead[0] = 8;
    ASSERT_EQ(ValidateMlaNonQuantFd(launch_, tiling), MlaStatus::SUCCESS);
    const std::pair<const char *, std::function<void(MlaNonQuantTiling &)>> invalid[] = {
        {"one_kv_part", [](auto &t) { t.fdParams.s2SplitNumOfFdHead[0] = 1; }},
        {"row_buffer_overrun", [](auto &t) { t.fdParams.gS1BaseSizeOfFd = 9; }},
        {"batch_overrun", [](auto &t) { t.fdParams.bN2IdxOfFdHead[0] = 2; }},
        {"output_overrun", [](auto &t) { t.fdParams.gS1IdxOfFdHead[0] = 1; }},
        {"accum_source_overrun", [](auto &t) { --t.workspaceParams.fdAccumOutSize; }},
        {"lse_source_overrun", [](auto &t) { --t.workspaceParams.fdLogSumExpSize; }},
        {"task_endpoint", [](auto &t) { t.fdParams.gS1IdxEndOfFdHead[0] = 1; }},
        {"split_endpoint", [](auto &t) { t.fdParams.gS1IdxEndOfFdHeadSplit[0] = 1; }},
        {"duplicate_endpoint", [](auto &t) { t.fdParams.usedVecNumOfFd = 2; }}};
    for (const auto &test : invalid) {
        SCOPED_TRACE(test.first);
        auto invalidTiling = tiling;
        test.second(invalidTiling);
        EXPECT_EQ(ValidateMlaNonQuantFd(launch_, invalidTiling), MlaStatus::INVALID_PATCH);
    }
}

TEST_F(FlashMlaMetadataProtocolTest, LegacyFdChecksReductionSourcesAndEndpoints)
{
    Reset(MlaFamily::LEGACY_MLA);
    MlaLegacyTiling tiling{};
    tiling.baseParams.qSeqSize = 1;
    tiling.baseParams.nNumOfQInOneGroup = 8;
    tiling.increFlashAttentionSingleCoreParams.s1SplitSize = 1;
    tiling.splitKVParams.accumOutSize = 8192;
    tiling.splitKVParams.logSumExpSize = 256;
    tiling.tndSplitCoreParams.tndFDCoreArrLen = 1;
    tiling.tndSplitCoreParams.usedVecNumOfFd = 1;
    tiling.tndSplitCoreParams.balanceFDCoreKVSplitArr[0] = 2;
    tiling.tndSplitCoreParams.gS1SplitNumOfFdHeadMla[0] = 1;
    tiling.tndSplitCoreParams.gS1LastPartSizeOfFdHead[0] = 8;
    ASSERT_EQ(ValidateMlaLegacyFd(launch_, tiling), MlaStatus::SUCCESS);
    const std::pair<const char *, std::function<void(MlaLegacyTiling &)>> invalid[] = {
        {"missing_vector", [](auto &t) { t.tndSplitCoreParams.usedVecNumOfFd = 0; }},
        {"one_kv_part", [](auto &t) { t.tndSplitCoreParams.balanceFDCoreKVSplitArr[0] = 1; }},
        {"row_buffer_overrun", [](auto &t) { t.tndSplitCoreParams.gS1LastPartSizeOfFdHead[0] = 9; }},
        {"output_overrun", [](auto &t) { t.tndSplitCoreParams.balanceFDCoreS1Arr[0] = 1; }},
        {"accum_source_overrun", [](auto &t) { --t.splitKVParams.accumOutSize; }},
        {"lse_source_overrun", [](auto &t) { --t.splitKVParams.logSumExpSize; }},
        {"task_endpoint", [](auto &t) { t.tndSplitCoreParams.gS1IdxEndOfFdHead[0] = 1; }},
        {"split_endpoint", [](auto &t) { t.tndSplitCoreParams.gS1IdxEndOfFdHeadSplit[0] = 1; }},
        {"duplicate_endpoint", [](auto &t) { t.tndSplitCoreParams.usedVecNumOfFd = 2; }}};
    for (const auto &test : invalid) {
        SCOPED_TRACE(test.first);
        auto invalidTiling = tiling;
        test.second(invalidTiling);
        EXPECT_EQ(ValidateMlaLegacyFd(launch_, invalidTiling), MlaStatus::INVALID_PATCH);
    }
}
} // namespace
