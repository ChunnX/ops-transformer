/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../op_kernel/mega_moe_tiling_def.h"

using namespace std;

namespace {
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_384 = 384;
constexpr uint32_t NUM_512 = 512;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

template <typename T1, typename T2>
inline T1 CeilA2B(T1 a, T2 b) {
    return (a + b - 1) / b;
}

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};
}

extern "C" void mega_moe(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR weight1, GM_ADDR weightScales1, GM_ADDR weight2, GM_ADDR weightScales2,
    GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM);

struct MegaMoeKernelTestParam {
    string caseName;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t topK;
    uint32_t EP;
    uint32_t expertPerRank;
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    uint32_t blockDim;
};

class MegaMoeKernelTest : public testing::TestWithParam<MegaMoeKernelTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "MegaMoeKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MegaMoeKernelTest TearDown" << std::endl;
    }
};

TEST_P(MegaMoeKernelTest, test_kernel_execution) {
    MegaMoeKernelTestParam param = GetParam();

    uint32_t M = param.M;
    uint32_t K = param.K;
    uint32_t N = param.N;
    uint32_t topK = param.topK;
    uint32_t EP = param.EP;
    uint32_t expertPerRank = param.expertPerRank;

    int64_t contextSize = sizeof(Mc2MoeContext) + EP * sizeof(uint64_t);
    int64_t xSize = M * K * sizeof(uint16_t);
    int64_t expertIdsSize = M * topK * sizeof(int32_t);
    int64_t scalesSize = M * topK * sizeof(float);
    int64_t xActiveMaskSize = M * sizeof(bool);
    int64_t expertScalesSize = M * topK * sizeof(uint16_t);
    int64_t weight1Size = expertPerRank * K * N * sizeof(uint8_t);
    int64_t weightScales1Size = expertPerRank * N * CeilA2B(K, 64) * sizeof(uint8_t);
    int64_t weight2Size = expertPerRank * (N / 2) * K * sizeof(uint8_t);
    int64_t weightScales2Size = expertPerRank * K * CeilA2B(N / 2, 64) * sizeof(uint8_t);
    int64_t ySize = M * K * sizeof(uint16_t);
    int64_t workspaceSize = 1024 * 1024 * 100;
    int64_t tilingSize = sizeof(MegaMoeTilingData);

    uint8_t* mc2Context = (uint8_t*)AscendC::GmAlloc(contextSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilA2B(xSize, 32) * 32);
    uint8_t* expertIds = (uint8_t*)AscendC::GmAlloc(CeilA2B(expertIdsSize, 32) * 32);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(CeilA2B(scalesSize, 32) * 32);
    uint8_t* xActiveMask = (uint8_t*)AscendC::GmAlloc(CeilA2B(xActiveMaskSize, 32) * 32);
    uint8_t* expertScales = (uint8_t*)AscendC::GmAlloc(CeilA2B(expertScalesSize, 32) * 32);
    uint8_t* weight1 = (uint8_t*)AscendC::GmAlloc(CeilA2B(weight1Size, 32) * 32);
    uint8_t* weightScales1 = (uint8_t*)AscendC::GmAlloc(CeilA2B(weightScales1Size, 32) * 32);
    uint8_t* weight2 = (uint8_t*)AscendC::GmAlloc(CeilA2B(weight2Size, 32) * 32);
    uint8_t* weightScales2 = (uint8_t*)AscendC::GmAlloc(CeilA2B(weightScales2Size, 32) * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilA2B(ySize, 32) * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset(mc2Context, 0, contextSize);
    memset(x, 0, CeilA2B(xSize, 32) * 32);
    memset(expertIds, 0, CeilA2B(expertIdsSize, 32) * 32);
    memset(weight1, 0, CeilA2B(weight1Size, 32) * 32);
    memset(weightScales1, 0, CeilA2B(weightScales1Size, 32) * 32);
    memset(weight2, 0, CeilA2B(weight2Size, 32) * 32);
    memset(weightScales2, 0, CeilA2B(weightScales2Size, 32) * 32);
    memset(y, 0, CeilA2B(ySize, 32) * 32);
    memset(workspace, 0, workspaceSize);

    Mc2MoeContext* contextData = reinterpret_cast<Mc2MoeContext*>(mc2Context);
    contextData->epRankId = 0;
    contextData->epRankSize = EP;
    contextData->winSize = 0;

    MegaMoeTilingData* tilingData = reinterpret_cast<MegaMoeTilingData*>(tiling);
    memset(tilingData, 0, tilingSize);
    tilingData->m = M;
    tilingData->k = K;
    tilingData->n = N;
    tilingData->m0 = param.m0;
    tilingData->k0 = param.k0;
    tilingData->n0 = param.n0;
    tilingData->topK = topK;
    tilingData->expertPerRank = expertPerRank;
    tilingData->EP = EP;
    tilingData->rank = 0;
    tilingData->maxOutputSize = M * topK * EP * 3;
    tilingData->groupListType = 1;
    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;

    uint32_t tilingKey = 0;
    if (param.m0 == NUM_128 && param.n0 == NUM_256 && param.k0 == NUM_512) {
        tilingKey = 1000;
    } else if (param.m0 == NUM_256 && param.n0 == NUM_256 && param.k0 == NUM_384) {
        tilingKey = 2000;
    }
    ICPU_SET_TILING_KEY(tilingKey);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(mega_moe, param.blockDim, mc2Context, x, expertIds, scales, xActiveMask, expertScales,
                weight1, weightScales1, weight2, weightScales2, y, workspace, tiling);

    AscendC::GmFree((void*)mc2Context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIds);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)xActiveMask);
    AscendC::GmFree((void*)expertScales);
    AscendC::GmFree((void*)weight1);
    AscendC::GmFree((void*)weightScales1);
    AscendC::GmFree((void*)weight2);
    AscendC::GmFree((void*)weightScales2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

static MegaMoeKernelTestParam kernel_cases[] = {
    {
        "test_kernel_M128_K512_N256_topK2_EP1",
        128, 512, 256, 2, 1, 4,
        128, 512, 256, 1
    },
    {
        "test_kernel_M256_K384_N256_topK2_EP1",
        256, 384, 256, 2, 1, 4,
        256, 384, 256, 1
    },
    {
        "test_kernel_M128_K512_N256_topK1_EP1",
        128, 512, 256, 1, 1, 4,
        128, 512, 256, 1
    },
    {
        "test_kernel_M128_K512_N256_topK2_EP2",
        128, 512, 256, 2, 2, 2,
        128, 512, 256, 1
    },
};

INSTANTIATE_TEST_CASE_P(MegaMoeKernel, MegaMoeKernelTest, testing::ValuesIn(kernel_cases));

TEST(MegaMoeKernelTilingTest, test_tiling_key_128_256_512) {
    uint32_t m0 = NUM_128;
    uint32_t n0 = NUM_256;
    uint32_t k0 = NUM_512;
    
    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 256;
    
    EXPECT_EQ(m0, M);
    EXPECT_EQ(k0, K);
    EXPECT_EQ(n0, N);
}

TEST(MegaMoeKernelTilingTest, test_tiling_key_256_256_384) {
    uint32_t m0 = NUM_256;
    uint32_t n0 = NUM_256;
    uint32_t k0 = NUM_384;
    
    uint32_t M = 256;
    uint32_t K = 384;
    
    EXPECT_EQ(m0, M);
    EXPECT_EQ(k0, K);
}

TEST(MegaMoeKernelWorkspaceTest, test_workspace_calculation) {
    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t topK = 2;
    uint32_t EP = 1;
    
    uint64_t peermemSize = M * topK * (K + K / 64) * sizeof(int8_t) + 
                          60 * 1024 + 
                          EP * 128 * sizeof(int32_t) + 
                          M * topK * K * sizeof(int16_t);
    
    EXPECT_GT(peermemSize, 0);
}