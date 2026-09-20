/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <cstring>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include <iostream>
#endif

#include "../../../../op_kernel/generic_lightning_indexer_template_tiling_key.h"

template <int DT_Q, int DT_K, int DT_OUT, int LAYOUT_Q, int LAYOUT_K, int SPARSE_MODE, int QUANT_MODE, int RETURN_VALUE,
          int HEAD_REDUCE, int EXPAND>
__global__ __aicore__ void generic_lightning_indexer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights, __gm__ uint8_t *actualSeqQ,
    __gm__ uint8_t *actualSeqK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *qDescale, __gm__ uint8_t *kDescale, __gm__ uint8_t *outputIdxOffset,
    __gm__ uint8_t *metadata, __gm__ uint8_t *startLoc, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling);

using namespace std;

namespace {
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t N2 = 1;
constexpr uint32_t S1_BASE_SIZE = 4;
constexpr uint32_t S2_BASE_SIZE = 128;
constexpr uint32_t M_BASE_SIZE = 256;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

void SetTilingData(GenericLightningIndexerTilingData *td, uint32_t b, uint32_t n1, uint32_t s1, uint32_t s2,
                   uint32_t topk, uint32_t cmpRatio, uint32_t sparseMode, uint32_t expandMode, uint32_t hasWeighted,
                   bool returnValue)
{
    td->bSize = b;
    td->gSize = n1;
    td->s1Size = s1;
    td->s2Size = static_cast<int64_t>(s2);
    td->sparseCount = topk;
    td->topk = topk;
    td->poolSize = cmpRatio;
    td->usedCoreNum = 1;
    td->s1BaseSize = S1_BASE_SIZE;
    td->mBaseSize = M_BASE_SIZE;
    td->mBaseSizeMax = M_BASE_SIZE;
    td->s2BaseSize = S2_BASE_SIZE;
    td->trunkLen = TRUNK_LEN_8K;
    td->maskMode = sparseMode;
    td->quantMode = 0;
    td->returnValue = returnValue ? 1 : 0;
    td->layoutQ = 0;
    td->layoutK = 0;
    td->blockSize = 0;
    td->maxBlockNumPerBatch = 0;
    td->keyStride0 = s2 * N2 * HEAD_DIM;
    td->keyDequantScaleStride0 = 0;
    td->wsOffScore = 0;
    td->wsOffLdScore = 0;
    td->wsOffLdIdx = 0;
    td->qkScale = 1.0f;
    td->expandMode = expandMode;
    td->hasWeightedSum = hasWeighted;
    td->actType = 0;
    td->selectMode = 0;
    td->keyReduceMode = 0;
    td->initBlocks = 0;
    td->localBlocks = 0;
    td->hasOutputIdxOffset = 0;
    td->maxSeqlenQ = -1;
    td->seqQCuFormat = 0;
    td->seqKCuFormat = 0;
    td->seqDtypeFlags = 0;
    td->msaScoreNT = 0;
    td->wsOffKCast = 0;
}
} // namespace

class GenericLightningIndexerKernelTest : public testing::Test {};

TEST_F(GenericLightningIndexerKernelTest, P_LI_BSND_FP16)
{
    const uint32_t b = 1;
    const uint32_t n1 = 8;
    const uint32_t s1 = 8;
    const uint32_t s2 = 32;
    const uint32_t topk = 16;
    int64_t qSize = b * s1 * n1 * HEAD_DIM;
    int64_t kSize = b * s2 * N2 * HEAD_DIM;
    int64_t wSize = b * s1 * n1;
    int64_t idxSize = b * s1 * topk;
    int64_t valSize = b * s1 * topk;

    uint8_t *queryGm = (uint8_t *)AscendC::GmAlloc(qSize * sizeof(half));
    uint8_t *keyGm = (uint8_t *)AscendC::GmAlloc(kSize * sizeof(half));
    uint8_t *weightsGm = (uint8_t *)AscendC::GmAlloc(wSize * sizeof(half));
    uint8_t *indicesGm = (uint8_t *)AscendC::GmAlloc(idxSize * sizeof(int32_t));
    uint8_t *valuesGm = (uint8_t *)AscendC::GmAlloc(valSize * sizeof(float));
    uint8_t *workspaceGm = (uint8_t *)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t *tilingGm = (uint8_t *)AscendC::GmAlloc(sizeof(GenericLightningIndexerTilingData));
    ASSERT_NE(queryGm, nullptr);

    vector<half> q(qSize, static_cast<half>(0.5f));
    vector<half> k(kSize, static_cast<half>(0.5f));
    vector<half> w(wSize, static_cast<half>(1.0f));
    memcpy(queryGm, q.data(), qSize * sizeof(half));
    memcpy(keyGm, k.data(), kSize * sizeof(half));
    memcpy(weightsGm, w.data(), wSize * sizeof(half));
    SetTilingData(reinterpret_cast<GenericLightningIndexerTilingData *>(tilingGm), b, n1, s1, s2, topk, 1, 3, 0, 1,
                  true);

    auto kernelFn = [&](GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR actQ, GM_ADDR actK, GM_ADDR sequsedQ,
                        GM_ADDR sequsedK, GM_ADDR residual, GM_ADDR blockTable, GM_ADDR qDescale, GM_ADDR kDescale,
                        GM_ADDR offset, GM_ADDR metadata, GM_ADDR startLoc, GM_ADDR indices, GM_ADDR values,
                        GM_ADDR workspace, GM_ADDR tiling) {
        generic_lightning_indexer<1, 1, 3, 0, 0, 3, 0, 1, 0, 0>(query, key, weights, actQ, actK, sequsedQ, sequsedK,
                                                                residual, blockTable, qDescale, kDescale, offset,
                                                                metadata, startLoc, indices, values, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFn, 1, queryGm, keyGm, weightsGm, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, indicesGm, valuesGm, workspaceGm, tilingGm);

    int32_t *indicesOut = reinterpret_cast<int32_t *>(indicesGm);
    for (int64_t i = 0; i < idxSize; ++i) {
        EXPECT_GE(indicesOut[i], -1);
    }
    AscendC::GmFree(queryGm);
    AscendC::GmFree(keyGm);
    AscendC::GmFree(weightsGm);
    AscendC::GmFree(indicesGm);
    AscendC::GmFree(valuesGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}

TEST_F(GenericLightningIndexerKernelTest, P_QWEN_sum_expand)
{
    const uint32_t b = 1;
    const uint32_t n1 = 8;
    const uint32_t s1 = 8;
    const uint32_t s2 = 8;
    const uint32_t topk = 4;
    const uint32_t r = 4;
    const uint32_t outLen = topk * r + (r - 1);
    int64_t qSize = b * s1 * n1 * HEAD_DIM;
    int64_t kSize = b * s2 * N2 * HEAD_DIM;
    int64_t idxSize = b * s1 * outLen;
    int64_t valSize = b * s1 * topk;

    uint8_t *queryGm = (uint8_t *)AscendC::GmAlloc(qSize * sizeof(half));
    uint8_t *keyGm = (uint8_t *)AscendC::GmAlloc(kSize * sizeof(half));
    uint8_t *residualGm = (uint8_t *)AscendC::GmAlloc(b * sizeof(int32_t));
    uint8_t *indicesGm = (uint8_t *)AscendC::GmAlloc(idxSize * sizeof(int32_t));
    uint8_t *valuesGm = (uint8_t *)AscendC::GmAlloc(valSize * sizeof(float));
    uint8_t *workspaceGm = (uint8_t *)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t *tilingGm = (uint8_t *)AscendC::GmAlloc(sizeof(GenericLightningIndexerTilingData));
    ASSERT_NE(queryGm, nullptr);

    vector<half> q(qSize, static_cast<half>(0.25f));
    vector<half> k(kSize, static_cast<half>(0.25f));
    int32_t residual = 1;
    memcpy(queryGm, q.data(), qSize * sizeof(half));
    memcpy(keyGm, k.data(), kSize * sizeof(half));
    memcpy(residualGm, &residual, sizeof(int32_t));
    SetTilingData(reinterpret_cast<GenericLightningIndexerTilingData *>(tilingGm), b, n1, s1, s2, topk, r, 4, 2, 0,
                  true);

    auto kernelFn = [&](GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR actQ, GM_ADDR actK, GM_ADDR sequsedQ,
                        GM_ADDR sequsedK, GM_ADDR residualAddr, GM_ADDR blockTable, GM_ADDR qDescale, GM_ADDR kDescale,
                        GM_ADDR offset, GM_ADDR metadata, GM_ADDR startLoc, GM_ADDR indices, GM_ADDR values,
                        GM_ADDR workspace, GM_ADDR tiling) {
        generic_lightning_indexer<1, 1, 3, 0, 0, 4, 0, 1, 1, 2>(query, key, weights, actQ, actK, sequsedQ, sequsedK,
                                                                residualAddr, blockTable, qDescale, kDescale, offset,
                                                                metadata, startLoc, indices, values, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFn, 1, queryGm, keyGm, nullptr, nullptr, nullptr, nullptr, nullptr, residualGm, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, indicesGm, valuesGm, workspaceGm, tilingGm);

    int32_t *indicesOut = reinterpret_cast<int32_t *>(indicesGm);
    for (int64_t i = 0; i < idxSize; ++i) {
        EXPECT_GE(indicesOut[i], -1);
    }
    AscendC::GmFree(queryGm);
    AscendC::GmFree(keyGm);
    AscendC::GmFree(residualGm);
    AscendC::GmFree(indicesGm);
    AscendC::GmFree(valuesGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}
