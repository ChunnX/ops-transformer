/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "../../op_kernel_aicpu/sparse_attention_score_metadata_q_seq_utils.h"
#include "../../op_kernel_aicpu/sparse_attention_score_metadata_scheduler.h"

#include <array>
#include <cstdlib>
#include <iostream>

using namespace aicpu::sparse_attention_score_metadata;
using namespace optiling::generic_block_sparse_attention_metadata;

#define CHECK_TRUE(expr)                                                                                               \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " << #expr << std::endl;                       \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_EQ(lhs, rhs)                                                                                             \
    do {                                                                                                               \
        const auto lhsValue = (lhs);                                                                                   \
        const auto rhsValue = (rhs);                                                                                   \
        if (lhsValue != rhsValue) {                                                                                    \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " << #lhs << " != " << #rhs << std::endl;        \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

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
    input.isPackedGQA = 1;
    input.aicCoreNum = 8;
    input.qSeqLens = {2, 1};
    input.validBlockNums = {0, 0, 0, 0, 0, 0};
    return input;
}

void TestPrefillSchedule()
{
    ScheduleInput input = MakeInput();
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.totalQTokenNum, 3);
    CHECK_EQ(result.saTotalTaskNum, 6);
    CHECK_EQ(result.saUsedCoreNum, 6);
    CHECK_EQ(result.sparseHeadNum, 2);
    CHECK_EQ(result.groupSize, 4);
    CHECK_EQ(result.decodeLaunchCoreNum, 0);

    TaskInfo task;
    CHECK_EQ(DecodeTask(input, result, 4, task), ScheduleStatus::SUCCESS);
    CHECK_EQ(task.qUnit, 2);
    CHECK_EQ(task.batchIdx, 1);
    CHECK_EQ(task.qTokenInBatch, 0);
    CHECK_EQ(task.kvHeadIdx, 0);
    CHECK_EQ(task.qHeadStart, 0);
    CHECK_EQ(task.qHeadCount, 4);
}

void TestActualBlockPrefixDecode()
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 64;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};

    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.blockPrefix[0], 0);
    CHECK_EQ(result.blockPrefix[1], 12);
    CHECK_EQ(result.blockPrefix[2], 15);
    CHECK_EQ(result.decodeLaunchCoreNum, 7);
    CHECK_EQ(result.decodePerCoreTaskNum, 3);

    CHECK_EQ(result.decodeSchedules[0].baseTaskStart, 0);
    CHECK_EQ(result.decodeSchedules[0].baseTaskEnd, 1);
    CHECK_EQ(result.decodeSchedules[0].firstBlockStart, 0);
    CHECK_EQ(result.decodeSchedules[0].lastBlockEnd, 3);
    CHECK_EQ(result.decodeSchedules[3].firstBlockStart, 9);
    CHECK_EQ(result.decodeSchedules[3].lastBlockEnd, 12);
    CHECK_EQ(result.decodeSchedules[4].baseTaskStart, 1);
    CHECK_EQ(result.decodeSchedules[4].baseTaskEnd, 2);
    CHECK_EQ(result.decodeSchedules[4].firstBlockStart, 0);
    CHECK_EQ(result.decodeSchedules[4].lastBlockEnd, 3);
    CHECK_EQ(result.decodeSchedules[5].baseTaskEnd, 0);

    CHECK_EQ(result.combineTaskNum, 1);
    CHECK_EQ(result.combineSchedules[0].baseTask, 0);
    CHECK_EQ(result.combineSchedules[0].firstCore, 0);
    CHECK_EQ(result.combineSchedules[0].partialStart, 0);
    CHECK_EQ(result.combineSchedules[0].partialCount, 4);
}

void TestDecodeShapeConstraints()
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
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};

    ScheduleResult result;
    input.blockShapeY = 64;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.decodeLaunchCoreNum, 0);

    input.blockShapeY = 128;
    input.headDim = 64;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.decodeLaunchCoreNum, 0);
}

void TestZeroCountsAndRepeatedPrefix()
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
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.blockPrefix[0], 0);
    CHECK_EQ(result.blockPrefix[1], 0);
    CHECK_EQ(result.blockPrefix[2], 12);
    CHECK_EQ(result.blockPrefix[3], 15);
    CHECK_EQ(result.decodeSchedules[0].baseTaskStart, 1);
}

void TestBlockCapacityDoesNotAffectSchedule()
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
    lhs.isPackedGQA = 1;
    lhs.aicCoreNum = 32;
    lhs.qSeqLens = {2};
    lhs.validBlockNums = {12, 3};
    ScheduleInput rhs = lhs;
    rhs.blockIndexStride = 128;
    ScheduleResult lhsResult;
    ScheduleResult rhsResult;
    CHECK_EQ(BuildSchedule(lhs, lhsResult), ScheduleStatus::SUCCESS);
    CHECK_EQ(BuildSchedule(rhs, rhsResult), ScheduleStatus::SUCCESS);
    CHECK_EQ(lhsResult.decodeLaunchCoreNum, rhsResult.decodeLaunchCoreNum);
    CHECK_EQ(lhsResult.decodePerCoreTaskNum, rhsResult.decodePerCoreTaskNum);
    CHECK_EQ(lhsResult.combineTaskNum, rhsResult.combineTaskNum);
    for (size_t core = 0; core < lhsResult.decodeSchedules.size(); ++core) {
        CHECK_EQ(lhsResult.decodeSchedules[core].baseTaskStart, rhsResult.decodeSchedules[core].baseTaskStart);
        CHECK_EQ(lhsResult.decodeSchedules[core].lastBlockEnd, rhsResult.decodeSchedules[core].lastBlockEnd);
    }
}

void TestUnpackedInternalTaskModel()
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 1;
    input.qSeqLens = {1};
    input.isPackedGQA = 0;
    input.validBlockNums.assign(8, 0);
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.sparseHeadNum, 8);
    CHECK_EQ(result.saTotalTaskNum, 8);

    TaskInfo task;
    CHECK_EQ(DecodeTask(input, result, 5, task), ScheduleStatus::SUCCESS);
    CHECK_EQ(task.sparseHeadIdx, 5);
    CHECK_EQ(task.qHeadStart, 5);
    CHECK_EQ(task.qHeadCount, 1);
    CHECK_EQ(task.kvHeadIdx, 1);
}

void TestInvalidBlockCount()
{
    ScheduleInput input = MakeInput();
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::INVALID_PARAM);
}

void TestTndSeqUsedUsesStorageOffsets()
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    CHECK_TRUE(BuildTndQSeqLayout({0, 4, 8}, {2, 3}, true, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    CHECK_EQ(actualQSeqLens[0], 2);
    CHECK_EQ(actualQSeqLens[1], 3);
    CHECK_EQ(qStorageBlockStarts[0], 0);
    CHECK_EQ(qStorageBlockStarts[1], 4);

    const int32_t counts[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> validBlockNums;
    CHECK_TRUE(GatherTndValidBlockNums(counts, 1, 8, 1, 8, actualQSeqLens, qStorageBlockStarts, validBlockNums));
    CHECK_EQ(validBlockNums.size(), 5U);
    CHECK_EQ(validBlockNums[0], 1);
    CHECK_EQ(validBlockNums[1], 2);
    CHECK_EQ(validBlockNums[2], 5);
    CHECK_EQ(validBlockNums[3], 6);
    CHECK_EQ(validBlockNums[4], 7);
}

void TestTndQSeqValidation()
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    CHECK_TRUE(!BuildTndQSeqLayout({0, 4, 8}, {5, 3}, true, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    CHECK_TRUE(!BuildTndQSeqLayout({0, 4, 3}, {}, false, 4, 1, 4, actualQSeqLens, qStorageBlockStarts));
    CHECK_TRUE(!BuildTndQSeqLayout({1, 4, 8}, {}, false, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    CHECK_TRUE(!BuildTndQSeqLayout({0, 4, 8}, {}, false, 4, 1, 7, actualQSeqLens, qStorageBlockStarts));
}

void TestMetadataEncoding()
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
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    std::array<MetadataType, METADATA_TOTAL_SIZE> metadata{};
    CHECK_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), ScheduleStatus::SUCCESS);
    CHECK_EQ(metadata[MAGIC_INDEX], METADATA_MAGIC);
    CHECK_EQ(metadata[VERSION_INDEX], METADATA_VERSION);
    CHECK_EQ(metadata[METADATA_USED_SIZE_INDEX], static_cast<int32_t>(METADATA_USED_SIZE));
    CHECK_EQ(metadata[SA_USED_CORE_NUM_INDEX], 2);
    CHECK_EQ(metadata[SA_TOTAL_TASK_NUM_INDEX], 2);
    CHECK_EQ(metadata[DECODE_LAUNCH_CORE_NUM_INDEX], 7);
    CHECK_EQ(metadata[DECODE_PER_CORE_TASK_NUM_INDEX], 3);
    CHECK_EQ(metadata[COMBINE_TASK_NUM_INDEX], 1);
    CHECK_EQ(metadata[GetDecodeScheduleIndex(4, DECODE_BASE_TASK_START_INDEX)], 1);
    CHECK_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_BASE_TASK_INDEX)], 0);
    CHECK_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_PARTIAL_COUNT_INDEX)], 4);
    CHECK_EQ(metadata[METADATA_USED_SIZE], 0);
}

int main()
{
    TestPrefillSchedule();
    TestActualBlockPrefixDecode();
    TestDecodeShapeConstraints();
    TestZeroCountsAndRepeatedPrefix();
    TestBlockCapacityDoesNotAffectSchedule();
    TestUnpackedInternalTaskModel();
    TestInvalidBlockCount();
    TestTndSeqUsedUsesStorageOffsets();
    TestTndQSeqValidation();
    TestMetadataEncoding();
    std::cout << "All sparse attention score metadata scheduler tests passed." << std::endl;
    return 0;
}
