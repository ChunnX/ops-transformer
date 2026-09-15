/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "aclnn/acl_meta.h"
#include "opdev/common_types.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

namespace FlashMlaTest {
void SetPlatformResources(fe::PlatFormInfos *platformInfos, uint32_t cubeCoreNum, uint32_t vectorCoreNum);
void ResetPlatformResources();

using Tensor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

inline Tensor MakeTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *data = nullptr,
                         const std::vector<int64_t> &strides = {}, int64_t offset = 0,
                         const std::vector<int64_t> &storageShape = {})
{
    const auto &storage = storageShape.empty() ? shape : storageShape;
    return Tensor(aclCreateTensor(shape.data(), shape.size(), dtype, strides.empty() ? nullptr : strides.data(), offset,
                                  ACL_FORMAT_ND, storage.data(), storage.size(), data),
                  aclDestroyTensor);
}

class ApiTest : public testing::Test {
protected:
    void SetUp() override
    {
        const auto &platform = op::GetCurrentPlatformInfo();
        previousSoc_ = platform.GetSocVersion();
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        SetPlatformResources(&platformInfos_, 24, 48);
    }

    void TearDown() override
    {
        ResetPlatformResources();
        op::SetPlatformSocVersion(previousSoc_);
    }

    Tensor MakeMetadata()
    {
        return MakeTensor({480}, ACL_INT32, metadataData_.data());
    }

    alignas(64) std::array<int32_t, 480> metadataData_{};
    fe::PlatFormInfos platformInfos_;

private:
    op::SocVersion previousSoc_ = op::SocVersion::ASCEND910B;
};
} // namespace FlashMlaTest

// The AICPU entry points are internal and do not need a separate aclnn header.
extern "C" {
aclnnStatus aclnnInnerFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *actualSeqLengthsQOptional, const aclTensor *actualSeqLengthsKvOptional, int64_t numHeads,
    int64_t batchSize, int64_t qSeqSize, int64_t kvSeqSize, const char *inputLayer,
    int64_t blockSize, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool hasAttenMask,
    int64_t quantMode, const aclTensor *metadataOut, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnInnerFlashMlaMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);
}

namespace {
using namespace FlashMlaTest;

class FlashMlaMetadataApiTest : public ApiTest {
protected:
    void SetUp() override
    {
        ApiTest::SetUp();
        metadata = MakeMetadata();
    }

    aclnnStatus Call(bool nullWorkspace = false, bool nullExecutor = false)
    {
        uint64_t workspaceSize = std::numeric_limits<uint64_t>::max();
        aclOpExecutor *executor = nullptr;
        const auto ret = aclnnInnerFlashMlaMetadataGetWorkspaceSize(
            seqQ.get(), seqKv.get(), numHeads, batchSize, qSeqSize, kvSeqSize, inputLayer, blockSize, sparseMode,
            preTokens, nextTokens, hasMask, quantMode, metadata.get(), nullWorkspace ? nullptr : &workspaceSize,
            nullExecutor ? nullptr : &executor);
        if (ret == ACLNN_SUCCESS) {
            EXPECT_NE(executor, nullptr);
            EXPECT_NE(workspaceSize, std::numeric_limits<uint64_t>::max());
        }
        delete executor;
        return ret;
    }

    void SetQuant()
    {
        quantMode = 1;
        blockSize = 128;
        seqKv = MakeTensor({2}, ACL_INT64);
    }

    Tensor seqQ{nullptr, aclDestroyTensor};
    Tensor seqKv{nullptr, aclDestroyTensor};
    Tensor metadata{nullptr, aclDestroyTensor};
    int64_t numHeads = 8;
    int64_t batchSize = 2;
    int64_t qSeqSize = 1;
    int64_t kvSeqSize = 128;
    int64_t blockSize = 0;
    int64_t sparseMode = 0;
    int64_t preTokens = INT32_MAX;
    int64_t nextTokens = INT32_MAX;
    int64_t quantMode = 0;
    bool hasMask = false;
    const char *inputLayer = "BSH";
};

TEST_F(FlashMlaMetadataApiTest, RequiredOutputLayoutAndWorkspacePointers)
{
    metadata.reset();
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_NULLPTR);
    metadata = MakeMetadata();
    inputLayer = nullptr;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_NULLPTR);
    inputLayer = "BSH";
    EXPECT_EQ(Call(true), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(Call(false, true), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(FlashMlaMetadataApiTest, ExecutionRequiresExecutor)
{
    EXPECT_EQ(aclnnInnerFlashMlaMetadata(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(FlashMlaMetadataApiTest, NonQuantModeAndSupportedSocs)
{
    for (auto soc : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SetPlatformSocVersion(soc);
        EXPECT_EQ(Call(), ACLNN_SUCCESS);
    }
}

TEST_F(FlashMlaMetadataApiTest, HeadAndBatchBounds)
{
    for (int64_t heads : {-1, 0, 129}) {
        numHeads = heads;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << heads;
    }
    numHeads = 1;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    numHeads = 128;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    batchSize = -1;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    batchSize = 65537;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    batchSize = 65536;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataApiTest, SequenceCapacityBounds)
{
    for (auto *size : {&qSeqSize, &kvSeqSize}) {
        const auto saved = *size;
        const int64_t maxSize = size == &qSeqSize ? UINT32_MAX : INT32_MAX;
        *size = -1;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *size = maxSize;
        EXPECT_EQ(Call(), ACLNN_SUCCESS);
        *size = maxSize + 1;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *size = static_cast<int64_t>(UINT32_MAX) + 1;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *size = saved;
    }
}

TEST_F(FlashMlaMetadataApiTest, InputLayerNamesAndOutputSuffixes)
{
    seqQ = MakeTensor({2}, ACL_INT64);
    seqKv = MakeTensor({2}, ACL_INT64);
    for (const char *name : {"BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"}) {
        inputLayer = name;
        EXPECT_EQ(Call(), ACLNN_SUCCESS) << name;
    }
    for (const char *name : {"", "NZ", "NBSD", "BSH_NTD", "TND_NBSD", "BSH_", "TND_NTD_extra", "bsh"}) {
        inputLayer = name;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << name;
    }
}

TEST_F(FlashMlaMetadataApiTest, UnsupportedQuantModes)
{
    for (int64_t mode : {-1, 2, 27, INT32_MAX}) {
        quantMode = mode;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << mode;
    }
}

TEST_F(FlashMlaMetadataApiTest, PaddedSequenceAbsentEmptyBroadcastAndPerBatch)
{
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    for (int64_t count : {0, 1, 2, 3, 65537}) {
        seqQ = MakeTensor({count}, ACL_INT64);
        seqKv = MakeTensor({count}, ACL_INT64);
        EXPECT_EQ(Call(), ACLNN_SUCCESS) << count;
    }
}

TEST_F(FlashMlaMetadataApiTest, PaddedSequenceCountCannotTruncateLogicalBatch)
{
    batchSize = 4;
    seqQ = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqQ.reset();
    seqKv = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FlashMlaMetadataApiTest, SequenceDtypeRankStrideFormatAndPlacement)
{
    for (auto *seq : {&seqQ, &seqKv}) {
        *seq = MakeTensor({2}, ACL_INT32);
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *seq = MakeTensor({1, 2}, ACL_INT64);
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *seq = MakeTensor({2}, ACL_INT64, nullptr, {2}, 0, {4});
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *seq = MakeTensor({2}, ACL_INT64);
        (*seq)->SetViewFormat(op::Format::FORMAT_NCHW);
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        *seq = MakeTensor({2}, ACL_INT64);
        (*seq)->GetTensor()->SetPlacement(gert::kOnHost);
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
        seq->reset();
    }
}

TEST_F(FlashMlaMetadataApiTest, ContiguousSequenceSlicesPreserveDescriptorsAndContents)
{
    std::array<int64_t, 4> qLengths{99, 1, 1, 99};
    std::array<int64_t, 4> kvLengths{99, 128, 64, 99};
    seqQ = MakeTensor({2}, ACL_INT64, qLengths.data(), {1}, 1, {4});
    seqKv = MakeTensor({2}, ACL_INT64, kvLengths.data(), {1}, 1, {4});
    const auto *qAddress = seqQ->GetData();
    const auto *kvAddress = seqKv->GetData();
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    EXPECT_EQ(seqQ->GetData(), qAddress);
    EXPECT_EQ(seqKv->GetData(), kvAddress);
    EXPECT_EQ(seqQ->GetStorageShape(), op::Shape({4}));
    EXPECT_EQ(seqKv->GetStorageShape(), op::Shape({4}));
    EXPECT_EQ(seqQ->GetViewOffset(), 1);
    EXPECT_EQ(seqKv->GetViewOffset(), 1);
    EXPECT_EQ(qLengths, (std::array<int64_t, 4>{99, 1, 1, 99}));
    EXPECT_EQ(kvLengths, (std::array<int64_t, 4>{99, 128, 64, 99}));
    EXPECT_EQ(metadataData_, (std::array<int32_t, 480>{}));
}

TEST_F(FlashMlaMetadataApiTest, TndRequiresBothExactSequenceCounts)
{
    inputLayer = "TND";
    qSeqSize = 2;
    kvSeqSize = 256;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqKv = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqQ = MakeTensor({1}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqQ = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    for (int64_t count : {0, 1, 3}) {
        seqKv = MakeTensor({count}, ACL_INT64);
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << count;
    }
}

TEST_F(FlashMlaMetadataApiTest, TndPagedKvLengthsStillRequireExactCount)
{
    inputLayer = "TND";
    blockSize = 128;
    qSeqSize = 2;
    seqQ = MakeTensor({2}, ACL_INT64);
    seqKv = MakeTensor({1}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqKv = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataApiTest, TndOutputSuffixStillRequiresExactSequenceCounts)
{
    inputLayer = "TND_NTD";
    seqQ = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqKv = MakeTensor({1}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqKv = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataApiTest, PagedAttentionRequiresKvSequence)
{
    blockSize = 128;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqKv = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataApiTest, PagedBlockSizeBounds)
{
    seqKv = MakeTensor({2}, ACL_INT64);
    for (int64_t block : {-1, 1, 17, 1040}) {
        blockSize = block;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << block;
    }
    blockSize = static_cast<int64_t>(INT32_MAX) + 1;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    for (int64_t block : {16, 128, 1024}) {
        blockSize = block;
        EXPECT_EQ(Call(), ACLNN_SUCCESS) << block;
    }
}

TEST_F(FlashMlaMetadataApiTest, Int8RequiresPaged128AndPowerOfTwoHeads)
{
    quantMode = 1;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    SetQuant();
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    blockSize = 64;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    blockSize = 128;
    numHeads = 3;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    for (int64_t heads : {1, 2, 4, 8, 16, 32, 64, 128}) {
        numHeads = heads;
        EXPECT_EQ(Call(), ACLNN_SUCCESS) << heads;
    }
}

TEST_F(FlashMlaMetadataApiTest, Int8QueryLayoutsAndStorageBounds)
{
    SetQuant();
    for (const char *name : {"BNSD", "BNSD_NBSD"}) {
        inputLayer = name;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    }
    for (const char *name : {"BSH", "BSH_NBSD", "BSND", "BSND_NBSD"}) {
        inputLayer = name;
        EXPECT_EQ(Call(), ACLNN_SUCCESS);
    }
    qSeqSize = 33;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    qSeqSize = 32;
    sparseMode = 3;
    hasMask = true;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    inputLayer = "TND";
    seqQ = MakeTensor({2}, ACL_INT64);
    qSeqSize = 1048576;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    qSeqSize = 1048577;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FlashMlaMetadataApiTest, Int8PaddedQueryIgnoresOnlyQuerySequenceCount)
{
    SetQuant();
    batchSize = 4;
    seqQ = MakeTensor({2}, ACL_INT64);
    seqKv = MakeTensor({4}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    seqKv = MakeTensor({2}, ACL_INT64);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    seqKv = MakeTensor({4}, ACL_INT64);
    seqQ = MakeTensor({2}, ACL_INT32);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FlashMlaMetadataApiTest, NonQuantSparseMaskCombinations)
{
    for (int64_t mode : {-1, 1, 2, 5, 8, 10}) {
        sparseMode = mode;
        hasMask = true;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << mode;
    }
    for (int64_t mode : {3, 4, 9}) {
        sparseMode = mode;
        hasMask = false;
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID) << mode;
        hasMask = true;
        EXPECT_EQ(Call(), ACLNN_SUCCESS) << mode;
    }
    sparseMode = 0;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
}

TEST_F(FlashMlaMetadataApiTest, Int8PaddedSparseMaskDependsOnStoredQueryLength)
{
    SetQuant();
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    hasMask = true;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    sparseMode = 3;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    qSeqSize = 2;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    sparseMode = 9;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    sparseMode = 4;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    sparseMode = 0;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    hasMask = false;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FlashMlaMetadataApiTest, Int8TndSparseMaskCombinations)
{
    SetQuant();
    inputLayer = "TND";
    seqQ = MakeTensor({2}, ACL_INT64);
    qSeqSize = 2;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    hasMask = true;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    for (int64_t mode : {3, 9}) {
        sparseMode = mode;
        EXPECT_EQ(Call(), ACLNN_SUCCESS) << mode;
    }
}

TEST_F(FlashMlaMetadataApiTest, EffectiveTokenWindowAfterNormalization)
{
    preTokens = nextTokens = std::numeric_limits<int64_t>::min();
    EXPECT_EQ(Call(), ACLNN_SUCCESS); // Unmasked sparse 0 overrides both tokens.
    hasMask = true;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    preTokens = 7;
    nextTokens = -8;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    preTokens = -8;
    nextTokens = 7;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    preTokens = -7;
    nextTokens = 7;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    preTokens = std::numeric_limits<int64_t>::min();
    nextTokens = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(Call(), ACLNN_SUCCESS); // Clamp produces an effective zero-width boundary.
    sparseMode = 4;
    preTokens = -2;
    nextTokens = 1;
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    for (int64_t mode : {3, 9}) {
        sparseMode = mode;
        EXPECT_EQ(Call(), ACLNN_SUCCESS); // Causal/tree modes replace the raw window.
    }
}

TEST_F(FlashMlaMetadataApiTest, EmptyBatchStillBuildsExecutorAndValidatesOutput)
{
    batchSize = qSeqSize = kvSeqSize = 0;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    inputLayer = "TND";
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    blockSize = 128;
    EXPECT_EQ(Call(), ACLNN_SUCCESS);
    metadata.reset();
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(FlashMlaMetadataApiTest, MetadataDescriptorContract)
{
    metadata = MakeTensor({480}, ACL_FLOAT, metadataData_.data());
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    for (const auto &shape : {std::vector<int64_t>{479}, {481}, {1, 480}, {0}}) {
        metadata = MakeTensor(shape, ACL_INT32, metadataData_.data());
        EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    }
    metadata = MakeMetadata();
    metadata->SetViewFormat(op::Format::FORMAT_NCHW);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    metadata = MakeMetadata();
    metadata->SetStorageFormat(op::Format::FORMAT_NCHW);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    metadata = MakeMetadata();
    metadata->SetViewStrides(op::Strides{2});
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    metadata = MakeMetadata();
    metadata->SetViewOffset(1);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    metadata = MakeMetadata();
    metadata->SetStorageOffset(1);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FlashMlaMetadataApiTest, MetadataAddressAndPlacement)
{
    metadata = MakeTensor({480}, ACL_INT32);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    metadata = MakeTensor({480}, ACL_INT32, metadataData_.data() + 1);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    metadata = MakeMetadata();
    metadata->GetTensor()->SetPlacement(gert::kOnHost);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FlashMlaMetadataApiTest, UnsupportedSocAndMissingPlatformResources)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
    EXPECT_EQ(Call(), ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    SetPlatformResources(nullptr, 24, 48);
    EXPECT_EQ(Call(), ACLNN_ERR_INNER_NULLPTR);
}
} // namespace
