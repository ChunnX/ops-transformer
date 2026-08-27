/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "../../../op_kernel_aicpu/generic_block_sparse_attention_metadata_q_seq_utils.h"
#include "../../../op_kernel_aicpu/generic_block_sparse_attention_metadata_scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>

using namespace aicpu::generic_block_sparse_attention_metadata;
using namespace optiling::generic_block_sparse_attention_metadata;

namespace {

class GenericBlockSparseAttentionMetadataSchedulerTest : public testing::Test {};

ScheduleInput MakeInput()
{
    ScheduleInput input;
    input.batchSize = 2;
    input.numQHeads = 8;
    input.numKvHeads = 2;
    input.maxQSeqLen = 3;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 6;
    input.isPackedGQA = 1;
    input.aicCoreNum = 8;
    input.qSeqLens = {2, 1};
    input.validBlockNums = {0, 0, 0, 0, 0, 0};
    return input;
}

uint32_t GetHostFdPartialCapacity(uint32_t aicCoreNum)
{
    const uint32_t maxNonEmptyBaseTaskNum =
        aicCoreNum == 0U ? 0U : std::min<uint32_t>(MAX_COMBINE_TASK_NUM, (aicCoreNum * 3U - 1U) / 10U);
    const uint32_t maxActiveCoreNum = std::min<uint32_t>(aicCoreNum, MAX_FD_ACTIVE_CORE_NUM);
    return maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ? 0U : maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PrefillSchedule)
{
    ScheduleInput input = MakeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.totalQTokenNum, 3);
    ASSERT_EQ(result.saTotalTaskNum, 6);
    ASSERT_EQ(result.saUsedCoreNum, 6);
    ASSERT_EQ(result.sparseHeadNum, 2);
    ASSERT_EQ(result.groupSize, 4);
    ASSERT_EQ(result.fdActiveCoreNum, 0);

    TaskInfo task;
    ASSERT_EQ(DecodeTask(input, result, 4, task), ScheduleStatus::SUCCESS);
    ASSERT_EQ(task.qUnit, 2);
    ASSERT_EQ(task.batchIdx, 1);
    ASSERT_EQ(task.qTokenInBatch, 0);
    ASSERT_EQ(task.kvHeadIdx, 0);
    ASSERT_EQ(task.qHeadStart, 0);
    ASSERT_EQ(task.qHeadCount, 4);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ActualBlockPrefixDecode)
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.blockPrefix[0], 0);
    ASSERT_EQ(result.blockPrefix[1], 12);
    ASSERT_EQ(result.blockPrefix[2], 15);
    ASSERT_EQ(result.fdActiveCoreNum, 5);
    ASSERT_EQ(result.decodePerCoreTaskNum, 3);

    ASSERT_EQ(result.decodeSchedules[0].baseTaskStart, 0);
    ASSERT_EQ(result.decodeSchedules[0].baseTaskEnd, 1);
    ASSERT_EQ(result.decodeSchedules[0].firstBlockStart, 0);
    ASSERT_EQ(result.decodeSchedules[0].lastBlockEnd, 3);
    ASSERT_EQ(result.decodeSchedules[3].firstBlockStart, 9);
    ASSERT_EQ(result.decodeSchedules[3].lastBlockEnd, 12);
    ASSERT_EQ(result.decodeSchedules[4].baseTaskStart, 1);
    ASSERT_EQ(result.decodeSchedules[4].baseTaskEnd, 2);
    ASSERT_EQ(result.decodeSchedules[4].firstBlockStart, 0);
    ASSERT_EQ(result.decodeSchedules[4].lastBlockEnd, 3);
    ASSERT_EQ(result.decodeSchedules[5].baseTaskEnd, 0);

    ASSERT_EQ(result.combineTaskNum, 1);
    ASSERT_EQ(result.combineSchedules[0].baseTask, 0);
    ASSERT_EQ(result.combineSchedules[0].firstCore, 0);
    ASSERT_EQ(result.combineSchedules[0].partialStart, 0);
    ASSERT_EQ(result.combineSchedules[0].partialCount, 4);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, DecodeShapeConstraints)
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};

    ScheduleResult result;
    input.blockShapeY = 64;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, 0);

    input.blockShapeY = 128;
    input.headDim = 64;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ZeroCountsAndRepeatedPrefix)
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 3;
    input.numQHeads = 1;
    input.numKvHeads = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {3};
    input.validBlockNums = {0, 12, 3};
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.blockPrefix[0], 0);
    ASSERT_EQ(result.blockPrefix[1], 0);
    ASSERT_EQ(result.blockPrefix[2], 12);
    ASSERT_EQ(result.blockPrefix[3], 15);
    ASSERT_EQ(result.decodeSchedules[0].baseTaskStart, 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, SparseBlockCapacityGate)
{
    ScheduleInput lhs;
    lhs.batchSize = 1;
    lhs.numQHeads = 4;
    lhs.numKvHeads = 1;
    lhs.maxQSeqLen = 2;
    lhs.headDim = 128;
    lhs.blockShapeX = 1;
    lhs.blockShapeY = 128;
    lhs.blockIndexStride = 12;
    lhs.qBlockStorageNum = 2;
    lhs.isPackedGQA = 1;
    lhs.aicCoreNum = 32;
    lhs.qSeqLens = {2};
    lhs.validBlockNums = {12, 3};
    ScheduleInput rhs = lhs;
    rhs.blockIndexStride = 17;
    ScheduleResult lhsResult;
    ScheduleResult rhsResult;
    ASSERT_EQ(BuildSchedule(lhs, lhsResult), ScheduleStatus::SUCCESS);
    ASSERT_EQ(BuildSchedule(rhs, rhsResult), ScheduleStatus::SUCCESS);
    ASSERT_TRUE(lhsResult.fdActiveCoreNum > 0);
    ASSERT_EQ(rhsResult.fdActiveCoreNum, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ActiveCoreCapacityAcrossBaseTasks)
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 8;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 8;
    input.isPackedGQA = 1;
    input.aicCoreNum = 28;
    input.qSeqLens = {8};
    input.validBlockNums.assign(8, 16);

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_TRUE(result.fdActiveCoreNum > 16);
    ASSERT_TRUE(result.fdActiveCoreNum <= input.aicCoreNum);
    ASSERT_EQ(result.combineTaskNum, 8);
    int64_t partialTaskNum = 0;
    for (int64_t task = 0; task < result.combineTaskNum; ++task) {
        const CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(task)];
        ASSERT_TRUE(combine.partialCount > 1);
        ASSERT_TRUE(combine.partialCount <= static_cast<int64_t>(MAX_FD_PARTIAL_PER_BASE_TASK));
        ASSERT_TRUE(combine.firstCore + combine.partialCount <= result.fdActiveCoreNum);
        partialTaskNum += combine.partialCount;
    }
    ASSERT_EQ(result.fdPartialTaskNum, partialTaskNum);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PartialTaskNumCanExceedActiveCoreNum)
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 7;
    input.qSeqLens = {2};
    input.validBlockNums = {3, 3};

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, 3);
    ASSERT_EQ(result.combineTaskNum, 2);
    ASSERT_EQ(result.fdPartialTaskNum, 4);
    ASSERT_TRUE(result.fdPartialTaskNum > result.fdActiveCoreNum);
    ASSERT_EQ(result.fdPartialTaskNum, static_cast<int64_t>(input.validBlockNums.size()) + result.fdActiveCoreNum - 1);
    ASSERT_TRUE(result.fdPartialTaskNum <= GetHostFdPartialCapacity(static_cast<uint32_t>(input.aicCoreNum)));
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PartialWorkspaceUpperBound)
{
    ASSERT_EQ(GetHostFdPartialCapacity(0U), 0U);
    ASSERT_EQ(GetHostFdPartialCapacity(28U), 35U);
    for (uint32_t aicCoreNum = 1U; aicCoreNum <= MAX_FD_ACTIVE_CORE_NUM; ++aicCoreNum) {
        const uint32_t maxNonEmptyBaseTaskNum = (aicCoreNum * 3U - 1U) / 10U;
        const uint32_t hostCapacity = GetHostFdPartialCapacity(aicCoreNum);
        for (uint32_t baseTaskNum = 1U; baseTaskNum <= maxNonEmptyBaseTaskNum; ++baseTaskNum) {
            for (int64_t validBlockNum = 1; validBlockNum <= 16; ++validBlockNum) {
                ScheduleInput input;
                input.batchSize = 1;
                input.numQHeads = 4;
                input.numKvHeads = 1;
                input.maxQSeqLen = baseTaskNum;
                input.headDim = 128;
                input.blockShapeX = 1;
                input.blockShapeY = 128;
                input.blockIndexStride = 16;
                input.qBlockStorageNum = baseTaskNum;
                input.isPackedGQA = 1;
                input.aicCoreNum = aicCoreNum;
                input.qSeqLens = {baseTaskNum};
                input.validBlockNums.assign(baseTaskNum, validBlockNum);

                ScheduleResult result;
                ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
                if ((static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED) == 0U) {
                    continue;
                }
                const int64_t actualIntersectionUpperBound =
                    static_cast<int64_t>(baseTaskNum) + result.fdActiveCoreNum - 1;
                ASSERT_TRUE(result.fdPartialTaskNum <= actualIntersectionUpperBound);
                ASSERT_TRUE(result.fdPartialTaskNum <= hostCapacity);
                for (int64_t combineIdx = 0; combineIdx < result.combineTaskNum; ++combineIdx) {
                    const CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(combineIdx)];
                    ASSERT_TRUE(combine.partialStart >= 0);
                    ASSERT_TRUE(combine.partialStart + combine.partialCount <= result.fdPartialTaskNum);
                }
            }
        }
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, UnpackedInternalTaskModel)
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 1;
    input.qSeqLens = {1};
    input.isPackedGQA = 0;
    input.validBlockNums.assign(8, 0);
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    ASSERT_EQ(result.sparseHeadNum, 8);
    ASSERT_EQ(result.saTotalTaskNum, 8);

    TaskInfo task;
    ASSERT_EQ(DecodeTask(input, result, 5, task), ScheduleStatus::SUCCESS);
    ASSERT_EQ(task.sparseHeadIdx, 5);
    ASSERT_EQ(task.qHeadStart, 5);
    ASSERT_EQ(task.qHeadCount, 1);
    ASSERT_EQ(task.kvHeadIdx, 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, InvalidBlockCount)
{
    ScheduleInput input = MakeInput();
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::INVALID_PARAM);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndSeqUsedUsesStorageOffsets)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(BuildTndQSeqLayout({0, 4, 8}, {2, 3}, true, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    ASSERT_EQ(actualQSeqLens[0], 2);
    ASSERT_EQ(actualQSeqLens[1], 3);
    ASSERT_EQ(qStorageBlockStarts[0], 0);
    ASSERT_EQ(qStorageBlockStarts[1], 4);

    const int32_t counts[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> validBlockNums;
    ASSERT_TRUE(GatherTndValidBlockNums(counts, 1, 8, 1, 8, actualQSeqLens, qStorageBlockStarts, validBlockNums));
    ASSERT_EQ(validBlockNums.size(), 5U);
    ASSERT_EQ(validBlockNums[0], 1);
    ASSERT_EQ(validBlockNums[1], 2);
    ASSERT_EQ(validBlockNums[2], 5);
    ASSERT_EQ(validBlockNums[3], 6);
    ASSERT_EQ(validBlockNums[4], 7);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndQSeqValidation)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(!BuildTndQSeqLayout({0, 4, 8}, {5, 3}, true, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({0, 4, 3}, {}, false, 4, 1, 4, actualQSeqLens, qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({1, 4, 8}, {}, false, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({0, 4, 8}, {}, false, 4, 1, 7, actualQSeqLens, qStorageBlockStarts));
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, MetadataEncoding)
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    std::array<MetadataType, METADATA_TOTAL_SIZE> metadata{};
    ASSERT_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), ScheduleStatus::SUCCESS);
    ASSERT_EQ(metadata[MAGIC_INDEX], METADATA_MAGIC);
    ASSERT_EQ(metadata[VERSION_INDEX], METADATA_VERSION);
    ASSERT_EQ(metadata[METADATA_USED_SIZE_INDEX], static_cast<int32_t>(METADATA_USED_SIZE));
    ASSERT_EQ(metadata[SA_USED_CORE_NUM_INDEX], 2);
    ASSERT_EQ(metadata[SA_TOTAL_TASK_NUM_INDEX], 2);
    ASSERT_EQ(metadata[FD_ACTIVE_CORE_NUM_INDEX], 5);
    ASSERT_EQ(metadata[DECODE_PER_CORE_TASK_NUM_INDEX], 3);
    ASSERT_EQ(metadata[COMBINE_TASK_NUM_INDEX], 1);
    ASSERT_EQ(metadata[FD_SCHEDULE_FLAGS_INDEX], static_cast<int32_t>(FD_SCHEDULE_ENABLED | FD_ACTUAL_BLOCK_PREFIX));
    ASSERT_EQ(metadata[FD_TOTAL_FLAT_TASK_NUM_INDEX], 15);
    ASSERT_EQ(metadata[FD_PARTIAL_TASK_NUM_INDEX], 4);
    ASSERT_EQ(metadata[CONFIG_SIGNATURE_INDEX], static_cast<int32_t>(result.configSignature));
    ASSERT_EQ(metadata[GetDecodeScheduleIndex(4, DECODE_BASE_TASK_START_INDEX)], 1);
    ASSERT_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_BASE_TASK_INDEX)], 0);
    ASSERT_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_PARTIAL_COUNT_INDEX)], 4);
    ASSERT_EQ(metadata[METADATA_USED_SIZE], 0);
}

} // namespace
