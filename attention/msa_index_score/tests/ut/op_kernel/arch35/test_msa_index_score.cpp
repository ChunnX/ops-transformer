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
 * \file test_msa_index_score.cpp
 * \brief MsaIndexScore op_kernel UT (arch35 / Ascend 950).
 *
 * 整 batch q_len=0 是算子支持的入口：scheduler 任务数为 0，kernel 完成 Init 后返回。
 * 覆盖 fp16 / bf16 / fp16+int8 三个 TilingKey。
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void msa_index_score(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale,
                                                      GM_ADDR attenMask, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKlen,
                                                      GM_ADDR startLoc, GM_ADDR score, GM_ADDR workspace,
                                                      GM_ADDR tiling);

namespace {
constexpr uint32_t MSA_TILING_KEY_BF16 = 0;
constexpr uint32_t MSA_TILING_KEY_FP16 = 1;
constexpr uint32_t MSA_TILING_KEY_FP16_INT8 = 3;
constexpr uint32_t MSA_KEY_LAYOUT_BBND = 0;
constexpr int32_t HEAD_DIM = 128;
constexpr int32_t BLOCK_SIZE = 128;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;

void FillTiling(MsaIndexScoreTilingData *td, uint32_t isQuant)
{
    td->batch = 1;
    td->totalQ = 0;
    td->numQHeads = 1;
    td->numKvHeads = 1;
    td->gqaGroup = 1;
    td->headDim = HEAD_DIM;
    td->blockSize = BLOCK_SIZE;
    td->maxBlocksPerBatch = 1;
    td->scoreBlockStride = 16;
    td->usedCoreNum = 1;
    td->isQuant = isQuant;
    td->sparseMode = 0;
    td->initBlocks = 0;
    td->localBlocks = 0;
    td->numPages = 1;
    td->strideQt = HEAD_DIM;
    td->strideQn = HEAD_DIM;
    td->strideKvBlock = BLOCK_SIZE * HEAD_DIM;
    td->strideKvToken = HEAD_DIM;
    td->strideScalePage = BLOCK_SIZE;
    td->strideScaleHead = BLOCK_SIZE;
    td->strideOutHead = 0;
    td->strideOutToken = 16;
    td->kScratchOffsetElems = 0;
    td->keyLayout = MSA_KEY_LAYOUT_BBND;
    td->totalK = 0;
    td->kvChunks = 1;
}

void RunEmptyQuery(uint32_t tilingKey, bool isQuant)
{
    const int64_t keyElems = static_cast<int64_t>(BLOCK_SIZE) * HEAD_DIM;
    const size_t queryBytes = 256;
    const size_t keyBytes = static_cast<size_t>(keyElems) * (isQuant ? sizeof(int8_t) : sizeof(half));
    const size_t scaleBytes = static_cast<size_t>(BLOCK_SIZE) * sizeof(float);
    const size_t tableBytes = sizeof(int32_t);
    const size_t qlenBytes = 2 * sizeof(int32_t);
    const size_t klenBytes = sizeof(int32_t);
    const size_t scoreBytes = 16 * sizeof(float);
    const size_t tilingBytes = sizeof(MsaIndexScoreTilingData);

    uint8_t *queryGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(queryBytes));
    uint8_t *keyGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(keyBytes));
    uint8_t *tableGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tableBytes));
    uint8_t *scaleGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(scaleBytes));
    uint8_t *qlenGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(qlenBytes));
    uint8_t *klenGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(klenBytes));
    uint8_t *startGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(klenBytes));
    uint8_t *scoreGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(scoreBytes));
    uint8_t *workspaceGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(WORKSPACE_SIZE));
    uint8_t *tilingGm = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tilingBytes));

    ASSERT_NE(queryGm, nullptr);
    ASSERT_NE(keyGm, nullptr);
    ASSERT_NE(qlenGm, nullptr);
    ASSERT_NE(tilingGm, nullptr);

    std::memset(queryGm, 0, queryBytes);
    std::memset(keyGm, 0, keyBytes);
    std::memset(scaleGm, 0, scaleBytes);
    std::memset(scoreGm, 0, scoreBytes);
    std::memset(workspaceGm, 0, WORKSPACE_SIZE);
    reinterpret_cast<int32_t *>(tableGm)[0] = 0;
    reinterpret_cast<int32_t *>(qlenGm)[0] = 0;
    reinterpret_cast<int32_t *>(qlenGm)[1] = 0;
    reinterpret_cast<int32_t *>(klenGm)[0] = 0;
    reinterpret_cast<int32_t *>(startGm)[0] = 0;

    auto *td = reinterpret_cast<MsaIndexScoreTilingData *>(tilingGm);
    FillTiling(td, isQuant ? 1U : 0U);

    ICPU_SET_TILING_KEY(tilingKey);
    // kernel 使用 INTRA_GROUP_MODE，CPU 仿真要求 MIX。AIC 在 CPU 仿真的 Process() 里直接返回。
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(msa_index_score, 1, queryGm, keyGm, tableGm, scaleGm, nullptr, qlenGm, klenGm, startGm, scoreGm,
                workspaceGm, tilingGm);

    AscendC::GmFree(queryGm);
    AscendC::GmFree(keyGm);
    AscendC::GmFree(tableGm);
    AscendC::GmFree(scaleGm);
    AscendC::GmFree(qlenGm);
    AscendC::GmFree(klenGm);
    AscendC::GmFree(startGm);
    AscendC::GmFree(scoreGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}
} // namespace

class MsaIndexScoreKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MsaIndexScoreKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MsaIndexScoreKernelTest TearDown" << endl;
    }
};

TEST_F(MsaIndexScoreKernelTest, EmptyQuery_BBND_FP16)
{
    RunEmptyQuery(MSA_TILING_KEY_FP16, false);
}

TEST_F(MsaIndexScoreKernelTest, EmptyQuery_BBND_BF16)
{
    RunEmptyQuery(MSA_TILING_KEY_BF16, false);
}

TEST_F(MsaIndexScoreKernelTest, EmptyQuery_BBND_FP16_INT8)
{
    RunEmptyQuery(MSA_TILING_KEY_FP16_INT8, true);
}
