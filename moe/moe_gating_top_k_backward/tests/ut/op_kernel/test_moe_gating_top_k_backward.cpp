/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k_backward.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "moe_gating_top_k_backward_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void moe_gating_top_k_backward(
    uint8_t* x_norm, uint8_t* grad_y, uint8_t* expert_idx, uint8_t* grad_x, uint8_t* workspace, uint8_t* tiling);

class moe_gating_top_k_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_gating_top_k_backward_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_gating_top_k_backward_test TearDown\n" << endl;
    }
};

TEST_F(moe_gating_top_k_backward_test, test_case_0)
{
    size_t m = 1;
    size_t n = 8;
    size_t k = 4;
    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;

    size_t x_norm_FileSize = m * n * sizeof(float);
    size_t grad_y_FileSize = m * k * sizeof(float);
    size_t expert_idx_FileSize = m * k * sizeof(int32_t);
    size_t grad_x_FileSize = m * n * sizeof(float);
    size_t workspace_FileSize = 16781184;
    size_t tiling_FileSize = sizeof(MoeGatingTopKBackwardTilingData);

    uint8_t* x_norm = (uint8_t*)AscendC::GmAlloc(x_norm_FileSize);
    uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(grad_y_FileSize);
    uint8_t* expert_idx = (uint8_t*)AscendC::GmAlloc(expert_idx_FileSize);
    uint8_t* grad_x = (uint8_t*)AscendC::GmAlloc(grad_x_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);
    MoeGatingTopKBackwardTilingData* tiling_data = reinterpret_cast<MoeGatingTopKBackwardTilingData*>(tiling);

    system(
    "cp -r "
    "../../../../../moe/moe_gating_top_k_backward/tests/ut/op_kernel/"
    "moe_gating_top_k_backward_data ./ && chmod -R 755 ./moe_gating_top_k_backward_data/");
    system("cd ./moe_gating_top_k_backward_data/ && rm -rf ./*bin");
    system("cd ./moe_gating_top_k_backward_data/ && python3 gen_data.py test_moe_gating_top_k_backward_case_00");
    system("cd ./moe_gating_top_k_backward_data/ && python3 gen_tiling.py test_moe_gating_top_k_backward_case_00");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_gating_top_k_backward_data/x_norm.bin", x_norm_FileSize, x_norm, x_norm_FileSize);
    ReadFile(path + "/moe_gating_top_k_backward_data/grad_y.bin", grad_y_FileSize, grad_y, grad_y_FileSize);
    ReadFile(path + "/moe_gating_top_k_backward_data/expert_idx.bin", expert_idx_FileSize, expert_idx, expert_idx_FileSize);
    ReadFile(path + "/moe_gating_top_k_backward_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_gating_top_k_backward, blockDim, x_norm, grad_y, expert_idx, grad_x, workspace, (uint8_t*)tiling_data);

    WriteFile(path + "/moe_gating_top_k_backward_data/grad_x.bin", grad_x, grad_x_FileSize);

    AscendC::GmFree((void*)x_norm);
    AscendC::GmFree((void*)grad_y);
    AscendC::GmFree((void*)expert_idx);
    AscendC::GmFree((void*)grad_x);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
