/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "securec.h"
#include "dense_lightning_indexer_softmax_lse_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void dense_lightning_indexer_softmax_lse(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR actualSeqLengthsQ,
    GM_ADDR actualSeqLengths, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GM_ADDR workspace, GM_ADDR tiling);

class dense_lightning_indexer_softmax_lse_kernel_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
        cout << "dense_lightning_indexer_softmax_lse_kernel_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "dense_lightning_indexer_softmax_lse_kernel_test TearDown" << endl;
    }
};

TEST_F(dense_lightning_indexer_softmax_lse_kernel_test, bsnd_f16_small_shape)
{
    int64_t b = 1, s1 = 128, s2 = 256, n1 = 8, n2 = 1, d = 128;
    int64_t g = n1 / n2;

    size_t querySize = b * s1 * n1 * d * sizeof(uint16_t);
    size_t keySize = b * s2 * n2 * d * sizeof(uint16_t);
    size_t weightSize = b * s1 * n1 * sizeof(uint16_t);
    size_t softmaxMaxSize = b * n2 * s1 * sizeof(float);
    size_t softmaxSumSize = b * n2 * s1 * sizeof(float);
    size_t workspaceSize = b * n2 * s1 * s2 * sizeof(float) * 2 + 65536;
    size_t tilingSize = sizeof(DenseLISoftmaxLseTilingData);

    uint8_t *query = (uint8_t *)AscendC::GmAlloc(querySize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(keySize);
    uint8_t *weights = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *softmaxMax = (uint8_t *)AscendC::GmAlloc(softmaxMaxSize);
    uint8_t *softmaxSum = (uint8_t *)AscendC::GmAlloc(softmaxSumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    DenseLISoftmaxLseTilingData *tilingData = reinterpret_cast<DenseLISoftmaxLseTilingData *>(tiling);
    tilingData->bSize = b;
    tilingData->n2Size = n2;
    tilingData->gSize = g;
    tilingData->s1Size = s1;
    tilingData->s2Size = s2;
    tilingData->usedCoreNum = 1;
    tilingData->sparseMode = 3;
    tilingData->preTokens = INT64_MAX;
    tilingData->nextTokens = INT64_MAX;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(dense_lightning_indexer_softmax_lse, 24, query, key, weights,
                nullptr, nullptr, softmaxMax, softmaxSum, workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(weights);
    AscendC::GmFree(softmaxMax);
    AscendC::GmFree(softmaxSum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(dense_lightning_indexer_softmax_lse_kernel_test, bsnd_f16_multi_core)
{
    int64_t b = 2, s1 = 512, s2 = 1024, n1 = 16, n2 = 1, d = 128;
    int64_t g = n1 / n2;

    size_t querySize = b * s1 * n1 * d * sizeof(uint16_t);
    size_t keySize = b * s2 * n2 * d * sizeof(uint16_t);
    size_t weightSize = b * s1 * n1 * sizeof(uint16_t);
    size_t softmaxMaxSize = b * n2 * s1 * sizeof(float);
    size_t softmaxSumSize = b * n2 * s1 * sizeof(float);
    size_t workspaceSize = b * n2 * s1 * s2 * sizeof(float) * 2 + 131072;
    size_t tilingSize = sizeof(DenseLISoftmaxLseTilingData);

    uint8_t *query = (uint8_t *)AscendC::GmAlloc(querySize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(keySize);
    uint8_t *weights = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *softmaxMax = (uint8_t *)AscendC::GmAlloc(softmaxMaxSize);
    uint8_t *softmaxSum = (uint8_t *)AscendC::GmAlloc(softmaxSumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    DenseLISoftmaxLseTilingData *tilingData = reinterpret_cast<DenseLISoftmaxLseTilingData *>(tiling);
    tilingData->bSize = b;
    tilingData->n2Size = n2;
    tilingData->gSize = g;
    tilingData->s1Size = s1;
    tilingData->s2Size = s2;
    tilingData->usedCoreNum = 8;
    tilingData->sparseMode = 3;
    tilingData->preTokens = INT64_MAX;
    tilingData->nextTokens = INT64_MAX;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(dense_lightning_indexer_softmax_lse, 24, query, key, weights,
                nullptr, nullptr, softmaxMax, softmaxSum, workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(weights);
    AscendC::GmFree(softmaxMax);
    AscendC::GmFree(softmaxSum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(dense_lightning_indexer_softmax_lse_kernel_test, bsnd_bf16_small_shape)
{
    int64_t b = 1, s1 = 64, s2 = 128, n1 = 32, n2 = 1, d = 128;
    int64_t g = n1 / n2;

    size_t querySize = b * s1 * n1 * d * sizeof(uint16_t);
    size_t keySize = b * s2 * n2 * d * sizeof(uint16_t);
    size_t weightSize = b * s1 * n1 * sizeof(uint16_t);
    size_t softmaxMaxSize = b * n2 * s1 * sizeof(float);
    size_t softmaxSumSize = b * n2 * s1 * sizeof(float);
    size_t workspaceSize = b * n2 * s1 * s2 * sizeof(float) * 2 + 65536;
    size_t tilingSize = sizeof(DenseLISoftmaxLseTilingData);

    uint8_t *query = (uint8_t *)AscendC::GmAlloc(querySize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(keySize);
    uint8_t *weights = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *softmaxMax = (uint8_t *)AscendC::GmAlloc(softmaxMaxSize);
    uint8_t *softmaxSum = (uint8_t *)AscendC::GmAlloc(softmaxSumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    DenseLISoftmaxLseTilingData *tilingData = reinterpret_cast<DenseLISoftmaxLseTilingData *>(tiling);
    tilingData->bSize = b;
    tilingData->n2Size = n2;
    tilingData->gSize = g;
    tilingData->s1Size = s1;
    tilingData->s2Size = s2;
    tilingData->usedCoreNum = 1;
    tilingData->sparseMode = 3;
    tilingData->preTokens = INT64_MAX;
    tilingData->nextTokens = INT64_MAX;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(dense_lightning_indexer_softmax_lse, 24, query, key, weights,
                nullptr, nullptr, softmaxMax, softmaxSum, workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(weights);
    AscendC::GmFree(softmaxMax);
    AscendC::GmFree(softmaxSum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(dense_lightning_indexer_softmax_lse_kernel_test, tnd_f16_small_shape)
{
    int64_t b = 1, t1 = 128, t2 = 256, n1 = 8, n2 = 1, d = 128;
    int64_t g = n1 / n2;

    int64_t actualSeqQueryData[1] = {t1};
    int64_t actualSeqKeyData[1] = {t2};
    size_t actualSeqQSize = b * sizeof(int64_t);
    size_t actualSeqKSize = b * sizeof(int64_t);

    size_t querySize = t1 * n1 * d * sizeof(uint16_t);
    size_t keySize = t2 * n2 * d * sizeof(uint16_t);
    size_t weightSize = t1 * n1 * sizeof(uint16_t);
    size_t softmaxMaxSize = n2 * t1 * sizeof(float);
    size_t softmaxSumSize = n2 * t1 * sizeof(float);
    size_t workspaceSize = n2 * t1 * t2 * sizeof(float) * 2 + 65536;
    size_t tilingSize = sizeof(DenseLISoftmaxLseTilingData);

    uint8_t *query = (uint8_t *)AscendC::GmAlloc(querySize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(keySize);
    uint8_t *weights = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *actualSeqQ = (uint8_t *)AscendC::GmAlloc(actualSeqQSize);
    uint8_t *actualSeqK = (uint8_t *)AscendC::GmAlloc(actualSeqKSize);
    uint8_t *softmaxMax = (uint8_t *)AscendC::GmAlloc(softmaxMaxSize);
    uint8_t *softmaxSum = (uint8_t *)AscendC::GmAlloc(softmaxSumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memcpy_s(actualSeqQ, actualSeqQSize, actualSeqQueryData, actualSeqQSize);
    memcpy_s(actualSeqK, actualSeqKSize, actualSeqKeyData, actualSeqKSize);

    DenseLISoftmaxLseTilingData *tilingData = reinterpret_cast<DenseLISoftmaxLseTilingData *>(tiling);
    tilingData->bSize = b;
    tilingData->n2Size = n2;
    tilingData->gSize = g;
    tilingData->s1Size = t1;
    tilingData->s2Size = t2;
    tilingData->usedCoreNum = 1;
    tilingData->sparseMode = 3;
    tilingData->preTokens = INT64_MAX;
    tilingData->nextTokens = INT64_MAX;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(dense_lightning_indexer_softmax_lse, 24, query, key, weights,
                actualSeqQ, actualSeqK, softmaxMax, softmaxSum, workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(weights);
    AscendC::GmFree(actualSeqQ);
    AscendC::GmFree(actualSeqK);
    AscendC::GmFree(softmaxMax);
    AscendC::GmFree(softmaxSum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(dense_lightning_indexer_softmax_lse_kernel_test, tnd_f16_multi_batch)
{
    int64_t b = 4, t1 = 256, t2 = 512, n1 = 16, n2 = 1, d = 128;
    int64_t g = n1 / n2;

    int64_t actualSeqQueryData[4] = {64, 128, 192, 256};
    int64_t actualSeqKeyData[4] = {128, 256, 384, 512};
    size_t actualSeqQSize = b * sizeof(int64_t);
    size_t actualSeqKSize = b * sizeof(int64_t);

    size_t querySize = t1 * n1 * d * sizeof(uint16_t);
    size_t keySize = t2 * n2 * d * sizeof(uint16_t);
    size_t weightSize = t1 * n1 * sizeof(uint16_t);
    size_t softmaxMaxSize = n2 * t1 * sizeof(float);
    size_t softmaxSumSize = n2 * t1 * sizeof(float);
    size_t workspaceSize = n2 * t1 * t2 * sizeof(float) * 2 + 131072;
    size_t tilingSize = sizeof(DenseLISoftmaxLseTilingData);

    uint8_t *query = (uint8_t *)AscendC::GmAlloc(querySize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(keySize);
    uint8_t *weights = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *actualSeqQ = (uint8_t *)AscendC::GmAlloc(actualSeqQSize);
    uint8_t *actualSeqK = (uint8_t *)AscendC::GmAlloc(actualSeqKSize);
    uint8_t *softmaxMax = (uint8_t *)AscendC::GmAlloc(softmaxMaxSize);
    uint8_t *softmaxSum = (uint8_t *)AscendC::GmAlloc(softmaxSumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memcpy_s(actualSeqQ, actualSeqQSize, actualSeqQueryData, actualSeqQSize);
    memcpy_s(actualSeqK, actualSeqKSize, actualSeqKeyData, actualSeqKSize);

    DenseLISoftmaxLseTilingData *tilingData = reinterpret_cast<DenseLISoftmaxLseTilingData *>(tiling);
    tilingData->bSize = b;
    tilingData->n2Size = n2;
    tilingData->gSize = g;
    tilingData->s1Size = t1;
    tilingData->s2Size = t2;
    tilingData->usedCoreNum = 8;
    tilingData->sparseMode = 3;
    tilingData->preTokens = INT64_MAX;
    tilingData->nextTokens = INT64_MAX;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(dense_lightning_indexer_softmax_lse, 24, query, key, weights,
                actualSeqQ, actualSeqK, softmaxMax, softmaxSum, workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(weights);
    AscendC::GmFree(actualSeqQ);
    AscendC::GmFree(actualSeqK);
    AscendC::GmFree(softmaxMax);
    AscendC::GmFree(softmaxSum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(dense_lightning_indexer_softmax_lse_kernel_test, bsnd_f16_large_shape)
{
    int64_t b = 1, s1 = 4096, s2 = 8192, n1 = 64, n2 = 1, d = 128;
    int64_t g = n1 / n2;

    size_t querySize = b * s1 * n1 * d * sizeof(uint16_t);
    size_t keySize = b * s2 * n2 * d * sizeof(uint16_t);
    size_t weightSize = b * s1 * n1 * sizeof(uint16_t);
    size_t softmaxMaxSize = b * n2 * s1 * sizeof(float);
    size_t softmaxSumSize = b * n2 * s1 * sizeof(float);
    size_t workspaceSize = b * n2 * s1 * s2 * sizeof(float) * 2 + 262144;
    size_t tilingSize = sizeof(DenseLISoftmaxLseTilingData);

    uint8_t *query = (uint8_t *)AscendC::GmAlloc(querySize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(keySize);
    uint8_t *weights = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *softmaxMax = (uint8_t *)AscendC::GmAlloc(softmaxMaxSize);
    uint8_t *softmaxSum = (uint8_t *)AscendC::GmAlloc(softmaxSumSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    DenseLISoftmaxLseTilingData *tilingData = reinterpret_cast<DenseLISoftmaxLseTilingData *>(tiling);
    tilingData->bSize = b;
    tilingData->n2Size = n2;
    tilingData->gSize = g;
    tilingData->s1Size = s1;
    tilingData->s2Size = s2;
    tilingData->usedCoreNum = 24;
    tilingData->sparseMode = 3;
    tilingData->preTokens = INT64_MAX;
    tilingData->nextTokens = INT64_MAX;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(dense_lightning_indexer_softmax_lse, 24, query, key, weights,
                nullptr, nullptr, softmaxMax, softmaxSum, workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(weights);
    AscendC::GmFree(softmaxMax);
    AscendC::GmFree(softmaxSum);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
