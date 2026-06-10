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
#include "../../../op_kernel/moe_ep_combine.cpp"
#include "../../../op_kernel/moe_ep_combine_tiling_key.h"

class MoeEpCombineKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeEpCombineKernelTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MoeEpCombineKernelTest TearDown\n" << std::endl;
    }
};

// Basic kernel test: without topk_weights (pure sum)
TEST_F(MoeEpCombineKernelTest, NoTopkWeights)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t allWorkspaceSize = sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpCombineTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpCombineTilingData *tiling_data = reinterpret_cast<MoeEpCombineTilingData*>(tiling);
    tiling_data->moeEpCombineInfo.cfg.epWorldSize = 8;
    tiling_data->moeEpCombineInfo.cfg.epRankId = 0;
    tiling_data->moeEpCombineInfo.cfg.numExperts = 64;
    tiling_data->moeEpCombineInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpCombineInfo.cfg.numTokens = 2048;
    tiling_data->moeEpCombineInfo.cfg.hidden = 7168;
    tiling_data->moeEpCombineInfo.cfg.topK = 8;
    tiling_data->moeEpCombineInfo.cfg.numMaxTokensPerRank = 2048;
    tiling_data->moeEpCombineInfo.cfg.perSlotBytes = 14400;
    tiling_data->moeEpCombineInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpCombineInfo.hasTopkWeights = 0;
    tiling_data->moeEpCombineInfo.aivNum = 48;
    tiling_data->moeEpCombineInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dstSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadata = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRank = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *topkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *bias0 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *bias1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *combinedX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *combinedTopkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));

    auto combineWrapper = [] (GM_ADDR context, GM_ADDR x, GM_ADDR dstSlotIdx,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR topkWeights,
        GM_ADDR bias0, GM_ADDR bias1, GM_ADDR combinedX, GM_ADDR combinedTopkWeights,
        GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_combine<0, 2>(context, x, dstSlotIdx, recvSrcMetadata,
                numRecvPerRank, topkWeights, bias0, bias1,
                combinedX, combinedTopkWeights, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(combineWrapper, 20, context, x, dstSlotIdx, recvSrcMetadata,
                numRecvPerRank, topkWeights, bias0, bias1,
                combinedX, combinedTopkWeights, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dstSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadata);
    AscendC::GmFree((void*)numRecvPerRank);
    AscendC::GmFree((void*)topkWeights);
    AscendC::GmFree((void*)bias0);
    AscendC::GmFree((void*)bias1);
    AscendC::GmFree((void*)combinedX);
    AscendC::GmFree((void*)combinedTopkWeights);
}

// Kernel test: with topk_weights
TEST_F(MoeEpCombineKernelTest, WithTopkWeights)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t allWorkspaceSize = sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpCombineTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpCombineTilingData *tiling_data = reinterpret_cast<MoeEpCombineTilingData*>(tiling);
    tiling_data->moeEpCombineInfo.cfg.epWorldSize = 8;
    tiling_data->moeEpCombineInfo.cfg.epRankId = 0;
    tiling_data->moeEpCombineInfo.cfg.numExperts = 64;
    tiling_data->moeEpCombineInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpCombineInfo.cfg.numTokens = 1024;
    tiling_data->moeEpCombineInfo.cfg.hidden = 7168;
    tiling_data->moeEpCombineInfo.cfg.topK = 8;
    tiling_data->moeEpCombineInfo.cfg.numMaxTokensPerRank = 1024;
    tiling_data->moeEpCombineInfo.cfg.perSlotBytes = 14400;
    tiling_data->moeEpCombineInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpCombineInfo.hasTopkWeights = 1;
    tiling_data->moeEpCombineInfo.aivNum = 48;
    tiling_data->moeEpCombineInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dstSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadata = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRank = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *topkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *bias0 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *bias1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *combinedX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *combinedTopkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));

    auto combineWrapper = [] (GM_ADDR context, GM_ADDR x, GM_ADDR dstSlotIdx,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR topkWeights,
        GM_ADDR bias0, GM_ADDR bias1, GM_ADDR combinedX, GM_ADDR combinedTopkWeights,
        GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_combine<1, 2>(context, x, dstSlotIdx, recvSrcMetadata,
                numRecvPerRank, topkWeights, bias0, bias1,
                combinedX, combinedTopkWeights, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(combineWrapper, 20, context, x, dstSlotIdx, recvSrcMetadata,
                numRecvPerRank, topkWeights, bias0, bias1,
                combinedX, combinedTopkWeights, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dstSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadata);
    AscendC::GmFree((void*)numRecvPerRank);
    AscendC::GmFree((void*)topkWeights);
    AscendC::GmFree((void*)bias0);
    AscendC::GmFree((void*)bias1);
    AscendC::GmFree((void*)combinedX);
    AscendC::GmFree((void*)combinedTopkWeights);
}

// Small shape test
TEST_F(MoeEpCombineKernelTest, SmallShape)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t allWorkspaceSize = sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpCombineTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpCombineTilingData *tiling_data = reinterpret_cast<MoeEpCombineTilingData*>(tiling);
    tiling_data->moeEpCombineInfo.cfg.epWorldSize = 2;
    tiling_data->moeEpCombineInfo.cfg.epRankId = 1;
    tiling_data->moeEpCombineInfo.cfg.numExperts = 16;
    tiling_data->moeEpCombineInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpCombineInfo.cfg.numTokens = 128;
    tiling_data->moeEpCombineInfo.cfg.hidden = 2048;
    tiling_data->moeEpCombineInfo.cfg.topK = 4;
    tiling_data->moeEpCombineInfo.cfg.numMaxTokensPerRank = 128;
    tiling_data->moeEpCombineInfo.cfg.perSlotBytes = 4192;
    tiling_data->moeEpCombineInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpCombineInfo.hasTopkWeights = 0;
    tiling_data->moeEpCombineInfo.aivNum = 40;
    tiling_data->moeEpCombineInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *dstSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadata = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRank = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *topkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *bias0 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *bias1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *combinedX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *combinedTopkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));

    auto combineWrapper = [] (GM_ADDR context, GM_ADDR x, GM_ADDR dstSlotIdx,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR topkWeights,
        GM_ADDR bias0, GM_ADDR bias1, GM_ADDR combinedX, GM_ADDR combinedTopkWeights,
        GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_combine<0, 2>(context, x, dstSlotIdx, recvSrcMetadata,
                numRecvPerRank, topkWeights, bias0, bias1,
                combinedX, combinedTopkWeights, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(combineWrapper, 20, context, x, dstSlotIdx, recvSrcMetadata,
                numRecvPerRank, topkWeights, bias0, bias1,
                combinedX, combinedTopkWeights, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dstSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadata);
    AscendC::GmFree((void*)numRecvPerRank);
    AscendC::GmFree((void*)topkWeights);
    AscendC::GmFree((void*)bias0);
    AscendC::GmFree((void*)bias1);
    AscendC::GmFree((void*)combinedX);
    AscendC::GmFree((void*)combinedTopkWeights);
}
