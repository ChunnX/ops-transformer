/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
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
#include "mega_moe_tiling_def.h"

extern "C" __global__ __aicore__ void mega_moe(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR weight1, GM_ADDR weightScales1, GM_ADDR weight2, GM_ADDR weightScales2,
    GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM);

class MegaMoeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MegaMoeTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MegaMoeTest TearDown\n" << std::endl;
    }
};

TEST_F(MegaMoeTest, MegaMoeTestBasic)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 100 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MegaMoeTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MegaMoeTilingData *tilingData = reinterpret_cast<MegaMoeTilingData*>(tiling);
    memset(tilingData, 0, tilingSize);
    tilingData->m = 128;
    tilingData->k = 512;
    tilingData->n = 256;
    tilingData->m0 = 128;
    tilingData->k0 = 512;
    tilingData->n0 = 256;
    tilingData->topK = 2;
    tilingData->expertPerRank = 4;
    tilingData->EP = 1;
    tilingData->rank = 0;
    tilingData->maxOutputSize = 128 * 2 * 1 * 3;
    tilingData->groupListType = 1;
    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;

    uint8_t *mc2Context = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *scales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *xActiveMask = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *expertScales = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *weight1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *weightScales1 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *weight2 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *weightScales2 = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint8_t));

    memset(mc2Context, 0, 1024);
    memset(x, 0, 1024);
    memset(expertIdx, 0, 1024);
    memset(y, 0, 1024);

    ICPU_RUN_KF(mega_moe, 1, mc2Context, x, expertIdx, scales, xActiveMask, expertScales,
                weight1, weightScales1, weight2, weightScales2, y, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)mc2Context);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertIdx);
    AscendC::GmFree((void*)scales);
    AscendC::GmFree((void*)xActiveMask);
    AscendC::GmFree((void*)expertScales);
    AscendC::GmFree((void*)weight1);
    AscendC::GmFree((void*)weightScales1);
    AscendC::GmFree((void*)weight2);
    AscendC::GmFree((void*)weightScales2);
    AscendC::GmFree((void*)y);
}