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
#include "gtest/gtest.h"
#include "moe_init_routing_v3_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_init_routing_v3(uint8_t *x, uint8_t *expertIdx, uint8_t *scale,
                                                          uint8_t *offset, uint8_t *expandedX, uint8_t *expandedRowIdx,
                                                          uint8_t *expertTokensCountOrCumsum, uint8_t *expandedScale,
                                                          uint8_t *workspace, uint8_t *tiling);

class moe_init_routing_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "moe_init_routing_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_init_routing_v3_test TearDown\n" << endl;
    }
};

// UNQUANTIZED_FULLLOAD
TEST_F(moe_init_routing_v3_test, test_case_8)
{
    size_t num_rows = 78;
    size_t cols = 2880;
    size_t k = 4;
    size_t expert_num = 946;
    uint64_t tilingKey = 2100000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 78 2880 4 946 float32 -1 141 0 946 0 1 1 [873,889] 1"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case7");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, nullptr, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);
            
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py float32 1 0");
    free(path_);
}

TEST_F(moe_init_routing_v3_test, test_case_9)
{
    size_t num_rows = 97;
    size_t cols = 2880;
    size_t k = 4;
    size_t expert_num = 734;
    uint64_t tilingKey = 2100000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 97 2880 4 734 float32 -1 734 0 313 0 1 1 [0,734] 1"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case8");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);
            
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py float32 1 1");
    free(path_);
}

TEST_F(moe_init_routing_v3_test, test_case_10)
{
    size_t num_rows = 94;
    size_t cols = 2880;
    size_t k = 4;
    size_t expert_num = 958;
    uint64_t tilingKey = 2100000;
    uint32_t blockDim = 40;

    size_t x_FileSize = num_rows * cols * sizeof(float);
    size_t expertIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t scale_FileSize = num_rows * sizeof(float);
    size_t expandedX_FileSize = num_rows * k * cols * sizeof(float);
    size_t expandedRowIdx_FileSize = num_rows * k * sizeof(int32_t);
    size_t expertTokensCumsum_FileSize = expert_num * sizeof(int64_t);
    size_t expandedScale_FileSize = num_rows * k * sizeof(float);
    size_t workspace_FileSize = static_cast<size_t>(num_rows * k * 24 + blockDim * 32 * 2 + num_rows * k * 4 +
                                                    expert_num * 4 + 32 + blockDim * cols * 4 + 16 * 1024 * 1024);
    size_t tiling_FileSize = sizeof(MoeInitRoutingV3TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *expertIdx = (uint8_t *)AscendC::GmAlloc(expertIdx_FileSize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scale_FileSize);
    uint8_t *expandedX = (uint8_t *)AscendC::GmAlloc(expandedX_FileSize);
    uint8_t *expandedRowIdx = (uint8_t *)AscendC::GmAlloc(expandedRowIdx_FileSize);
    uint8_t *expertTokensCountOrCumsum = (uint8_t *)AscendC::GmAlloc(expertTokensCumsum_FileSize);
    uint8_t *expandedScale = (uint8_t *)AscendC::GmAlloc(expandedScale_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system(
        "cp -r "
        "../../../../../moe/moe_init_routing_v3/tests/ut/op_kernel/"
        "moe_init_routing_v3_data ./ && chmod -R 755 ./moe_init_routing_v3_data/");
    system("ls -lh ./moe_init_routing_v3_data/");
    system("cd ./moe_init_routing_v3_data/ && rm -rf ./*bin");
    system("cd ./moe_init_routing_v3_data/ && python3 gen_data.py 94 2880 4 958 float32 -1 199 0 958 0 0 1 [554,650] 0"); // float16
    system("cd ./moe_init_routing_v3_data/ && python3 gen_tiling.py case9");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/moe_init_routing_v3_data/input_x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/input_expertIdx.bin", expertIdx_FileSize, expertIdx, expertIdx_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    ReadFile(path + "/moe_init_routing_v3_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(moe_init_routing_v3, blockDim, x, expertIdx, scale, nullptr, expandedX, expandedRowIdx,
                expertTokensCountOrCumsum, expandedScale, workspace, tiling);
            
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_x.bin", expandedX, expandedX_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_row_idx.bin", expandedRowIdx, expandedRowIdx_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expert_tokens_count.bin", expertTokensCountOrCumsum, expertTokensCumsum_FileSize);
    WriteFile(path + "/moe_init_routing_v3_data/output_expanded_scale.bin", expandedScale, expandedScale_FileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)expertIdx);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)expandedX);
    AscendC::GmFree((void *)expandedRowIdx);
    AscendC::GmFree((void *)expertTokensCountOrCumsum);
    AscendC::GmFree((void *)expandedScale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./moe_init_routing_v3_data/ && python3 compare.py float32 0 1");
    free(path_);
}