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
#include "../../../op_kernel/moe_ep_dispatch_epilogue.cpp"
#include "../../../op_kernel/moe_ep_dispatch_epilogue_tiling_key.h"

class MoeEpDispatchEpilogueKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeEpDispatchEpilogueKernelTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MoeEpDispatchEpilogueKernelTest TearDown\n" << std::endl;
    }
};

// Basic kernel test
TEST_F(MoeEpDispatchEpilogueKernelTest, Basic)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t allWorkspaceSize = sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpDispatchEpilogueTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpDispatchEpilogueTilingData *tiling_data =
        reinterpret_cast<MoeEpDispatchEpilogueTilingData*>(tiling);
    tiling_data->moeEpDispatchEpilogueInfo.cfg.epWorldSize = 8;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.epRankId = 0;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numExperts = 64;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numTokens = 2048;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.hidden = 7168;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.topK = 8;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numMaxTokensPerRank = 2048;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.perSlotBytes = 14400;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpDispatchEpilogueInfo.aAlloc = 16384;
    tiling_data->moeEpDispatchEpilogueInfo.numLocalExperts = 8;
    tiling_data->moeEpDispatchEpilogueInfo.aivNum = 48;
    tiling_data->moeEpDispatchEpilogueInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024);
    uint8_t *dstBufferSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadataInternal = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRankHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerExpertHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint64_t));
    uint8_t *recvX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *recvTopkIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvTopkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *recvSrcMetadata = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRank = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerExpert = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint64_t));

    auto epilogueWrapper = [] (GM_ADDR context, GM_ADDR dstBufferSlotIdx,
        GM_ADDR recvSrcMetadataInternal, GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_dispatch_epilogue<2>(context, dstBufferSlotIdx, recvSrcMetadataInternal,
                numRecvPerRankHp, numRecvPerExpertHp,
                recvX, recvTopkIdx, recvTopkWeights,
                recvSrcMetadata, numRecvPerRank, numRecvPerExpert,
                workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(epilogueWrapper, 20, context, dstBufferSlotIdx, recvSrcMetadataInternal,
                numRecvPerRankHp, numRecvPerExpertHp, recvX, recvTopkIdx, recvTopkWeights,
                recvSrcMetadata, numRecvPerRank, numRecvPerExpert, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)dstBufferSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadataInternal);
    AscendC::GmFree((void*)numRecvPerRankHp);
    AscendC::GmFree((void*)numRecvPerExpertHp);
    AscendC::GmFree((void*)recvX);
    AscendC::GmFree((void*)recvTopkIdx);
    AscendC::GmFree((void*)recvTopkWeights);
    AscendC::GmFree((void*)recvSrcMetadata);
    AscendC::GmFree((void*)numRecvPerRank);
    AscendC::GmFree((void*)numRecvPerExpert);
}

// Kernel test: small shape
TEST_F(MoeEpDispatchEpilogueKernelTest, SmallShape)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t allWorkspaceSize = sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpDispatchEpilogueTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpDispatchEpilogueTilingData *tiling_data =
        reinterpret_cast<MoeEpDispatchEpilogueTilingData*>(tiling);
    tiling_data->moeEpDispatchEpilogueInfo.cfg.epWorldSize = 2;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.epRankId = 1;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numExperts = 16;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numTokens = 128;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.hidden = 2048;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.topK = 4;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.numMaxTokensPerRank = 128;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.perSlotBytes = 4192;
    tiling_data->moeEpDispatchEpilogueInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpDispatchEpilogueInfo.aAlloc = 256;
    tiling_data->moeEpDispatchEpilogueInfo.numLocalExperts = 8;
    tiling_data->moeEpDispatchEpilogueInfo.aivNum = 40;
    tiling_data->moeEpDispatchEpilogueInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024);
    uint8_t *dstBufferSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadataInternal = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRankHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerExpertHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint64_t));
    uint8_t *recvX = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *recvTopkIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvTopkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *recvSrcMetadata = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRank = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerExpert = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint64_t));

    auto epilogueWrapper = [] (GM_ADDR context, GM_ADDR dstBufferSlotIdx,
        GM_ADDR recvSrcMetadataInternal, GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_dispatch_epilogue<2>(context, dstBufferSlotIdx, recvSrcMetadataInternal,
                numRecvPerRankHp, numRecvPerExpertHp,
                recvX, recvTopkIdx, recvTopkWeights,
                recvSrcMetadata, numRecvPerRank, numRecvPerExpert,
                workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(epilogueWrapper, 20, context, dstBufferSlotIdx, recvSrcMetadataInternal,
                numRecvPerRankHp, numRecvPerExpertHp, recvX, recvTopkIdx, recvTopkWeights,
                recvSrcMetadata, numRecvPerRank, numRecvPerExpert, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)dstBufferSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadataInternal);
    AscendC::GmFree((void*)numRecvPerRankHp);
    AscendC::GmFree((void*)numRecvPerExpertHp);
    AscendC::GmFree((void*)recvX);
    AscendC::GmFree((void*)recvTopkIdx);
    AscendC::GmFree((void*)recvTopkWeights);
    AscendC::GmFree((void*)recvSrcMetadata);
    AscendC::GmFree((void*)numRecvPerRank);
    AscendC::GmFree((void*)numRecvPerExpert);
}
