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
#include <iostream>
#include <string>
#include <cstdint>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "grouped_mat_mul_allto_allv_tiling_def.h"
#include "../../../op_kernel/grouped_mat_mul_allto_allv.cpp"

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class GroupedMatMulAlltoAllvTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvTest TearDown\n" << std::endl;
    }
};

TEST_F(GroupedMatMulAlltoAllvTest, GroupedMatMulAlltoAllvTest0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 10 * 1024 * 1024;
    size_t usrWorkspaceSize = 10 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(GmmA2avTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    memset(tiling, 0, tilingSize);

    GmmA2avTilingData *tilingData = reinterpret_cast<GmmA2avTilingData *>(tiling);

    uint64_t A = 4096;
    uint64_t H = 7168;
    uint64_t N1 = 4096;
    uint64_t BsK = 4096;
    uint64_t BS = 2048;
    uint64_t epWorldSize = 1;
    uint64_t e = 1;

    tilingData->taskTilingInfo.BSK = BsK;
    tilingData->taskTilingInfo.BS = BS;
    tilingData->taskTilingInfo.H1 = H;
    tilingData->taskTilingInfo.H2 = H;
    tilingData->taskTilingInfo.A = A;
    tilingData->taskTilingInfo.N1 = N1;
    tilingData->taskTilingInfo.N2 = N1;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize;
    tilingData->taskTilingInfo.e = e;
    tilingData->taskTilingInfo.ubSize = 65536;
    tilingData->taskTilingInfo.mainLoopExpertNum = 1;
    tilingData->taskTilingInfo.tailLoopExpertNum = 0;
    tilingData->taskTilingInfo.totalLoopCount = 1;

    tilingData->workspaceInfo.wsGmmOutputSize = A * N1 * sizeof(uint16_t);
    tilingData->workspaceInfo.wsGmmComputeWorkspaceSize = 1024;
    tilingData->workspaceInfo.wsSharedGmmComputeWorkspaceSize = 1024;

    tilingData->gmmBaseTiling.gmmQuantParams.groupNum = 1;
    tilingData->gmmBaseTiling.gmmArray.mList[0] = A;
    tilingData->gmmBaseTiling.gmmArray.kList[0] = H;
    tilingData->gmmBaseTiling.gmmArray.nList[0] = N1;

    tilingData->sharedGmmTiling.gmmQuantParams.groupNum = 1;
    tilingData->sharedGmmTiling.gmmArray.mList[0] = BS;
    tilingData->sharedGmmTiling.gmmArray.kList[0] = H;
    tilingData->sharedGmmTiling.gmmArray.nList[0] = N1;

    uint8_t *gmmxGM = (uint8_t *)AscendC::GmAlloc(A * H * sizeof(uint16_t));
    uint8_t *gmmweightGM = (uint8_t *)AscendC::GmAlloc(e * H * N1 * sizeof(uint16_t));
    uint8_t *sendCountsTensorOptionalGM = nullptr;
    uint8_t *recvCountsTensorOptionalGM = nullptr;
    uint8_t *mmxOptionalGM = nullptr;
    uint8_t *mmweightOptionalGM = nullptr;
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(BsK * N1 * sizeof(uint16_t));
    uint8_t *mmyOptionalGM = nullptr;

    ICPU_SET_TILING_KEY(0);
    auto groupedMatMulAlltoAllvWrapper = [](GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM,
                                            GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
                                            GM_ADDR mmweightOptionalGM, GM_ADDR yGM, GM_ADDR mmyOptionalGM,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        grouped_mat_mul_allto_allv<false, false, false>(gmmxGM, gmmweightGM, sendCountsTensorOptionalGM,
                                                        recvCountsTensorOptionalGM, mmxOptionalGM, mmweightOptionalGM,
                                                        yGM, mmyOptionalGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(groupedMatMulAlltoAllvWrapper, 20, gmmxGM, gmmweightGM, sendCountsTensorOptionalGM,
                recvCountsTensorOptionalGM, mmxOptionalGM, mmweightOptionalGM, yGM, mmyOptionalGM, workspace, tiling);

    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)gmmxGM);
    AscendC::GmFree((void *)gmmweightGM);
    AscendC::GmFree((void *)yGM);
}