/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include <cstdint>
#include "tikicpulib.h"
#include "data_utils.h"
#endif

extern "C" __global__ __aicore__ void mega_moe(
    GM_ADDR context, GM_ADDR x, GM_ADDR topk_ids, GM_ADDR topk_weights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weight_scales1, GM_ADDR weight_scales2,
    GM_ADDR x_active_mask, GM_ADDR scales,
    GM_ADDR y, GM_ADDR expert_token_nums, GM_ADDR workspace, GM_ADDR tiling);

class mega_moe_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "mega_moe_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "mega_moe_test TearDown\n" << std::endl;
    }
};

#ifdef __CCE_KT_TEST__
TEST_F(mega_moe_test, ascend950_case0_h4096_bs128)
{
    system("mkdir -p ./mega_moe_data");
    
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    
    uint32_t BS = 128;
    uint32_t H = 4096;
    uint32_t N = 1024;
    uint32_t K = 6;
    uint32_t expertPerRank = 4;
    uint32_t epWorldSize = 4;
    uint32_t blockDim = 64;
    
    size_t contextByteSize = 1 * sizeof(int32_t);
    size_t xByteSize = BS * H * sizeof(bfloat16);
    size_t topkIdsByteSize = BS * K * sizeof(int32_t);
    size_t topkWeightsByteSize = BS * K * sizeof(float);
    size_t weight1ByteSize = expertPerRank * N * H * sizeof(uint8_t);
    size_t weight2ByteSize = expertPerRank * H * (N / 2) * sizeof(uint8_t);
    size_t weightScales1ByteSize = expertPerRank * N * ((H + 63) / 64) * 2 * sizeof(uint8_t);
    size_t weightScales2ByteSize = expertPerRank * H * ((N / 2 + 63) / 64) * 2 * sizeof(uint8_t);
    size_t yByteSize = BS * H * sizeof(bfloat16);
    size_t expertTokenNumsByteSize = expertPerRank * sizeof(int32_t);
    
    uint8_t* context_buf = (uint8_t*)AscendC::GmAlloc(contextByteSize);
    uint8_t* x_buf = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* topk_ids_buf = (uint8_t*)AscendC::GmAlloc(topkIdsByteSize);
    uint8_t* topk_weights_buf = (uint8_t*)AscendC::GmAlloc(topkWeightsByteSize);
    uint8_t* weight1_buf = (uint8_t*)AscendC::GmAlloc(weight1ByteSize);
    uint8_t* weight2_buf = (uint8_t*)AscendC::GmAlloc(weight2ByteSize);
    uint8_t* weight_scales1_buf = (uint8_t*)AscendC::GmAlloc(weightScales1ByteSize);
    uint8_t* weight_scales2_buf = (uint8_t*)AscendC::GmAlloc(weightScales2ByteSize);
    uint8_t* y_buf = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* expert_token_nums_buf = (uint8_t*)AscendC::GmAlloc(expertTokenNumsByteSize);
    
    uint8_t* tiling_buf = (uint8_t*)AscendC::GmAlloc(sizeof(MegaMoeTilingData));
    
    std::string curPath = "./mega_moe_data";
    
    ICPU_SET_TILING_KEY(3510);
    ICPU_RUN_KF(
        mega_moe, blockDim,
        context_buf, x_buf, topk_ids_buf, topk_weights_buf,
        weight1_buf, weight2_buf, weight_scales1_buf, weight_scales2_buf,
        nullptr, nullptr,
        y_buf, expert_token_nums_buf, workspace, tiling_buf
    );
    
    AscendC::GmFree(context_buf);
    AscendC::GmFree(x_buf);
    AscendC::GmFree(topk_ids_buf);
    AscendC::GmFree(topk_weights_buf);
    AscendC::GmFree(weight1_buf);
    AscendC::GmFree(weight2_buf);
    AscendC::GmFree(weight_scales1_buf);
    AscendC::GmFree(weight_scales2_buf);
    AscendC::GmFree(y_buf);
    AscendC::GmFree(expert_token_nums_buf);
    AscendC::GmFree(tiling_buf);
    AscendC::GmFree(workspace);
}
#endif

TEST_F(mega_moe_test, placeholder_test)
{
    std::cout << "mega_moe placeholder test - needs real hardware environment" << std::endl;
}