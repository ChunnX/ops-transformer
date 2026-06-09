/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_kernel/lightning_indexer_v2.cpp"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void lightning_indexer_v2(
    GM_ADDR q, GM_ADDR k, GM_ADDR w,
    GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK,
    GM_ADDR sequsedQ, GM_ADDR sequsedK,
    GM_ADDR cmpResidualK, GM_ADDR blocktable,
    GM_ADDR outputIdxOffset, GM_ADDR metadata,
    GM_ADDR sparseIndices, GM_ADDR sparseValues,
    GM_ADDR workspace, GM_ADDR tiling);

static void RunFp16TndTnd(GM_ADDR q, GM_ADDR k, GM_ADDR w,
    GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK,
    GM_ADDR cmpResidualK, GM_ADDR blocktable, GM_ADDR outputIdxOffset, GM_ADDR metadata,
    GM_ADDR sparseIndices, GM_ADDR sparseValues, GM_ADDR workspace, GM_ADDR tiling)
{
    lightning_indexer_v2<1, 1, 3, 0, 1, 1, 0>(q, k, w, cuSeqlensQ, cuSeqlensK,
        sequsedQ, sequsedK, cmpResidualK, blocktable, outputIdxOffset, metadata,
        sparseIndices, sparseValues, workspace, tiling);
}

class LightningIndexerV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "LightningIndexerV2KernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "LightningIndexerV2KernelTest TearDown" << endl;
    }
};

// FP16 TND/TND: B=36, T_q=4096, T_kc=1010, N1=64, N2=1, D=128, topk=512, mask_mode=3, cmp_ratio=4, return_value=1
TEST_F(LightningIndexerV2KernelTest, kernel_fp16_tnd_tnd_0)
{
    uint32_t blockDim = 24;
    size_t qSize = 4096 * 64 * 128 * sizeof(uint16_t);
    size_t kSize = 1010 * 1 * 128 * sizeof(uint16_t);
    size_t wSize = 4096 * 64 * sizeof(float);
    size_t outSize = 4096 * 1 * 512 * sizeof(int32_t);
    size_t outValSize = 4096 * 1 * 512 * sizeof(float);
    size_t wsSize = 256 * 1024 * 1024;
    size_t tilingSize = sizeof(LIV2TilingData);

    uint8_t *q = (uint8_t *)AscendC::GmAlloc(qSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(kSize);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(wSize);
    uint8_t *cuSeqlensQ = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *cuSeqlensK = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *sequsedQ = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *sequsedK = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *cmpResidualK = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *blocktable = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *outputIdxOffset = (uint8_t *)AscendC::GmAlloc(256);
    uint8_t *metadata = (uint8_t *)AscendC::GmAlloc(4096);
    uint8_t *sparseIndices = (uint8_t *)AscendC::GmAlloc(outSize);
    uint8_t *sparseValues = (uint8_t *)AscendC::GmAlloc(outValSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(wsSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memset(q, 0, qSize);
    memset(k, 0, kSize);
    memset(w, 0, wSize);
    memset(sparseIndices, 0, outSize);
    memset(sparseValues, 0, outValSize);

    int32_t cuSeqlensQData[] = {0, 49, 95, 344, 646, 703, 805, 990, 1037, 1172, 1238, 1344, 1452,
        1630, 1684, 1779, 1897, 2043, 2096, 2206, 2348, 2481, 2702, 2804, 2873,
        2969, 3022, 3130, 3238, 3279, 3486, 3592, 3664, 3749, 3817, 3892, 4062, 4096};
    int32_t cuSeqlensKData[] = {0, 12, 23, 85, 160, 174, 199, 245, 256, 289, 305, 331, 358,
        402, 415, 438, 467, 503, 516, 543, 578, 611, 666, 691, 708,
        732, 745, 772, 799, 809, 860, 886, 904, 925, 942, 960, 1002, 1010};
    int32_t cmpResidualKData[] = {1, 2, 1, 2, 1, 2, 1, 3, 3, 2, 2, 0, 2, 2, 3, 2, 2, 1, 2, 2, 1, 1, 2, 1, 0, 1, 0, 0, 1, 3, 2, 0, 1, 0, 3, 2};

    memcpy(cuSeqlensQ, cuSeqlensQData, sizeof(cuSeqlensQData));
    memcpy(cuSeqlensK, cuSeqlensKData, sizeof(cuSeqlensKData));
    memcpy(cmpResidualK, cmpResidualKData, sizeof(cmpResidualKData));

    LIV2TilingData *td = reinterpret_cast<LIV2TilingData *>(tiling);
    memset(td, 0, tilingSize);
    td->bSize = 37;
    td->s1Size = 4096;
    td->s2Size = 1010;
    td->topk = 512;
    td->gSize = 64;
    td->blockSize = 0;
    td->maxBlockNumPerBatch = 0;
    td->maskMode = 3;
    td->preTokens = INT64_MAX;
    td->nextTokens = INT64_MAX;
    td->cmpRatio = 4;
    td->returnValue = 1;
    td->usedCoreNum = blockDim;
    td->batchSupperFlag = 0;

    ICPU_SET_TILING_KEY(570622209);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(RunFp16TndTnd, blockDim,
                q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK,
                cmpResidualK, blocktable, outputIdxOffset, metadata,
                sparseIndices, sparseValues, workspace, tiling);

    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(w);
    AscendC::GmFree(cuSeqlensQ);
    AscendC::GmFree(cuSeqlensK);
    AscendC::GmFree(sequsedQ);
    AscendC::GmFree(sequsedK);
    AscendC::GmFree(cmpResidualK);
    AscendC::GmFree(blocktable);
    AscendC::GmFree(outputIdxOffset);
    AscendC::GmFree(metadata);
    AscendC::GmFree(sparseIndices);
    AscendC::GmFree(sparseValues);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
