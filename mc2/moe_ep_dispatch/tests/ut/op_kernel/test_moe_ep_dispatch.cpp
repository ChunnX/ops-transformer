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
#include "../../../op_kernel/moe_ep_dispatch.cpp"
#include "../../../op_kernel/moe_ep_dispatch_tiling_key.h"

class MoeEpDispatchKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeEpDispatchKernelTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MoeEpDispatchKernelTest TearDown\n" << std::endl;
    }
};

// Basic kernel test: non-cached, doCpuSync, direct network
TEST_F(MoeEpDispatchKernelTest, NonCachedCpuSyncDirect)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpDispatchTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpDispatchTilingData *tiling_data = reinterpret_cast<MoeEpDispatchTilingData*>(tiling);
    tiling_data->moeEpDispatchInfo.cfg.epWorldSize = 8;
    tiling_data->moeEpDispatchInfo.cfg.epRankId = 0;
    tiling_data->moeEpDispatchInfo.cfg.numExperts = 64;
    tiling_data->moeEpDispatchInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpDispatchInfo.cfg.numTokens = 2048;
    tiling_data->moeEpDispatchInfo.cfg.hidden = 7168;
    tiling_data->moeEpDispatchInfo.cfg.topK = 8;
    tiling_data->moeEpDispatchInfo.cfg.numMaxTokensPerRank = 2048;
    tiling_data->moeEpDispatchInfo.cfg.perSlotBytes = 14400;
    tiling_data->moeEpDispatchInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpDispatchInfo.doCpuSync = 1;
    tiling_data->moeEpDispatchInfo.isCached = 0;
    tiling_data->moeEpDispatchInfo.networkMode = 0;
    tiling_data->moeEpDispatchInfo.aivNum = 48;
    tiling_data->moeEpDispatchInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *topkIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *topkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *cachedSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRankHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerExpertHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint64_t));
    uint8_t *dstBufferSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadataInternal = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));

    auto moeEpDispatchWrapper = [] (GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_dispatch<1, 0, 0>(context, x, topkIdx, topkWeights,
                cachedSlotIdx, numRecvPerRankHp, numRecvPerExpertHp,
                dstBufferSlotIdx, recvSrcMetadataInternal, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeEpDispatchWrapper, 20, context, x, topkIdx, topkWeights, cachedSlotIdx,
                numRecvPerRankHp, numRecvPerExpertHp, dstBufferSlotIdx, recvSrcMetadataInternal,
                workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)topkIdx);
    AscendC::GmFree((void*)topkWeights);
    AscendC::GmFree((void*)cachedSlotIdx);
    AscendC::GmFree((void*)numRecvPerRankHp);
    AscendC::GmFree((void*)numRecvPerExpertHp);
    AscendC::GmFree((void*)dstBufferSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadataInternal);
}

// Cached mode test
TEST_F(MoeEpDispatchKernelTest, CachedMode)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t allWorkspaceSize = sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MoeEpDispatchTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MoeEpDispatchTilingData *tiling_data = reinterpret_cast<MoeEpDispatchTilingData*>(tiling);
    tiling_data->moeEpDispatchInfo.cfg.epWorldSize = 4;
    tiling_data->moeEpDispatchInfo.cfg.epRankId = 0;
    tiling_data->moeEpDispatchInfo.cfg.numExperts = 32;
    tiling_data->moeEpDispatchInfo.cfg.numLocalExperts = 8;
    tiling_data->moeEpDispatchInfo.cfg.numTokens = 512;
    tiling_data->moeEpDispatchInfo.cfg.hidden = 4096;
    tiling_data->moeEpDispatchInfo.cfg.topK = 4;
    tiling_data->moeEpDispatchInfo.cfg.numMaxTokensPerRank = 512;
    tiling_data->moeEpDispatchInfo.cfg.perSlotBytes = 8288;
    tiling_data->moeEpDispatchInfo.cfg.expertAlignment = 1;
    tiling_data->moeEpDispatchInfo.doCpuSync = 0;
    tiling_data->moeEpDispatchInfo.isCached = 1;
    tiling_data->moeEpDispatchInfo.networkMode = 0;
    tiling_data->moeEpDispatchInfo.aivNum = 48;
    tiling_data->moeEpDispatchInfo.totalUbSize = 196352;

    uint8_t *context = (uint8_t *)AscendC::GmAlloc(1024);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *topkIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *topkWeights = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(float));
    uint8_t *cachedSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerRankHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *numRecvPerExpertHp = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint64_t));
    uint8_t *dstBufferSlotIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));
    uint8_t *recvSrcMetadataInternal = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint32_t));

    auto moeEpDispatchWrapper = [] (GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            moe_ep_dispatch<0, 1, 0>(context, x, topkIdx, topkWeights,
                cachedSlotIdx, numRecvPerRankHp, numRecvPerExpertHp,
                dstBufferSlotIdx, recvSrcMetadataInternal, workspaceGM, tilingGM);
        };
    ICPU_RUN_KF(moeEpDispatchWrapper, 20, context, x, topkIdx, topkWeights, cachedSlotIdx,
                numRecvPerRankHp, numRecvPerExpertHp, dstBufferSlotIdx, recvSrcMetadataInternal,
                workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)topkIdx);
    AscendC::GmFree((void*)topkWeights);
    AscendC::GmFree((void*)cachedSlotIdx);
    AscendC::GmFree((void*)numRecvPerRankHp);
    AscendC::GmFree((void*)numRecvPerExpertHp);
    AscendC::GmFree((void*)dstBufferSlotIdx);
    AscendC::GmFree((void*)recvSrcMetadataInternal);
}
