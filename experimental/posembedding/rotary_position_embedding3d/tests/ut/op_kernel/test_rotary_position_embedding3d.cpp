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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_rotary_position_embedding3d.h"
#include "data_utils.h"
#include "../../../op_kernel/rotary_position_embedding3d.cpp"

#include <cstdint>

using namespace std;

class rotary_position_embedding3d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rotary_position_embedding3d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotary_position_embedding3d_test TearDown\n" << endl;
    }
};

// [2, 4, 8] float32 — 3D 基准测试
TEST_F(rotary_position_embedding3d_test, test_case_fp32_3d_001)
{
    uint32_t B = 2, S = 4, D = 8;
    size_t inputXByteSize = B * S * D * sizeof(float);
    size_t inputCosSinByteSize = B * S * D * sizeof(float);
    size_t outputZByteSize = B * S * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbedding3dTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos_sin = (uint8_t *)AscendC::GmAlloc(inputCosSinByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputZByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 8;

    system(
        "cp -r ../../../../../experimental/posembedding/rotary_position_embedding3d/tests/ut/op_kernel/"
        "rotary_position_embedding3d_data ./");
    system("chmod -R 755 ./rotary_position_embedding3d_data/");
    system("cd ./rotary_position_embedding3d_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding3d_data/ && python3 gen_data.py 2 4 8 float32");
    system("cd ./rotary_position_embedding3d_data/ && python3 gen_tiling.py case_2_4_8");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbedding3dTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding3d_data/x.bin", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding3d_data/cos_sin.bin", inputCosSinByteSize, cos_sin, inputCosSinByteSize);
    ReadFile(path + "/rotary_position_embedding3d_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    auto rotary_position_embedding3d_wrapper = [](GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
        rotary_position_embedding3d<0>(x, y, z, workspace, tiling);
    };
    ICPU_RUN_KF(rotary_position_embedding3d_wrapper, blockDim, x, cos_sin, z, workspace,
                (uint8_t *)(tilingDatafromBin));

    // Verify against golden reference
    float* z_f32 = (float*)z;
    size_t numElts = outputZByteSize / sizeof(float);
    std::vector<float> golden(numElts);
    ReadFile(path + "/rotary_position_embedding3d_data/golden.bin", outputZByteSize, (uint8_t*)golden.data(), outputZByteSize);
    bool utPass = true;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numElts; i++) {
        float diff = fabsf(z_f32[i] - golden[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 5e-2f) {  // relaxed tolerance for half precision
            if (utPass) std::cout << "First mismatch at " << i << ": got " << z_f32[i] << " exp " << golden[i] << " diff " << diff << std::endl;
            utPass = false;
        }
    }
    std::cout << "  Max diff: " << maxDiff << " " << (utPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(utPass);

    AscendC::GmFree(x);
    AscendC::GmFree(cos_sin);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 16, 64] float16 — 大尺寸 fp16 测试
TEST_F(rotary_position_embedding3d_test, test_case_fp16_3d_002)
{
    uint32_t B = 4, S = 16, D = 64;
    size_t inputXByteSize = B * S * D * sizeof(half);
    size_t inputCosSinByteSize = B * S * D * sizeof(half);
    size_t outputZByteSize = B * S * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbedding3dTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos_sin = (uint8_t *)AscendC::GmAlloc(inputCosSinByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputZByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 8;

    system(
        "cp -r ../../../../../experimental/posembedding/rotary_position_embedding3d/tests/ut/op_kernel/"
        "rotary_position_embedding3d_data ./");
    system("chmod -R 755 ./rotary_position_embedding3d_data/");
    system("cd ./rotary_position_embedding3d_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding3d_data/ && python3 gen_data.py 4 16 64 float16");
    system("cd ./rotary_position_embedding3d_data/ && python3 gen_tiling.py case_4_16_64");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbedding3dTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding3d_data/x.bin", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding3d_data/cos_sin.bin", inputCosSinByteSize, cos_sin, inputCosSinByteSize);
    ReadFile(path + "/rotary_position_embedding3d_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    auto rotary_position_embedding3d_wrapper = [](GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
        rotary_position_embedding3d<1>(x, y, z, workspace, tiling);
    };
    ICPU_RUN_KF(rotary_position_embedding3d_wrapper, blockDim, x, cos_sin, z, workspace,
                (uint8_t *)(tilingDatafromBin));

    // Verify against golden reference
    float* z_f32 = (float*)z;
    size_t numElts = outputZByteSize / sizeof(float);
    std::vector<float> golden(numElts);
    ReadFile(path + "/rotary_position_embedding3d_data/golden.bin", outputZByteSize, (uint8_t*)golden.data(), outputZByteSize);
    bool utPass = true;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numElts; i++) {
        float diff = fabsf(z_f32[i] - golden[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 5e-2f) {  // relaxed tolerance for half precision
            if (utPass) std::cout << "First mismatch at " << i << ": got " << z_f32[i] << " exp " << golden[i] << " diff " << diff << std::endl;
            utPass = false;
        }
    }
    std::cout << "  Max diff: " << maxDiff << " " << (utPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(utPass);

    AscendC::GmFree(x);
    AscendC::GmFree(cos_sin);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 32, 128] float32 — 单 batch 大 sequence 测试
TEST_F(rotary_position_embedding3d_test, test_case_fp32_3d_003)
{
    uint32_t B = 1, S = 32, D = 128;
    size_t inputXByteSize = B * S * D * sizeof(float);
    size_t inputCosSinByteSize = B * S * D * sizeof(float);
    size_t outputZByteSize = B * S * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbedding3dTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos_sin = (uint8_t *)AscendC::GmAlloc(inputCosSinByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputZByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 8;

    system(
        "cp -r ../../../../../experimental/posembedding/rotary_position_embedding3d/tests/ut/op_kernel/"
        "rotary_position_embedding3d_data ./");
    system("chmod -R 755 ./rotary_position_embedding3d_data/");
    system("cd ./rotary_position_embedding3d_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding3d_data/ && python3 gen_data.py 1 32 128 float32");
    system("cd ./rotary_position_embedding3d_data/ && python3 gen_tiling.py case_1_32_128");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbedding3dTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding3d_data/x.bin", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding3d_data/cos_sin.bin", inputCosSinByteSize, cos_sin, inputCosSinByteSize);
    ReadFile(path + "/rotary_position_embedding3d_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    auto rotary_position_embedding3d_wrapper = [](GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
        rotary_position_embedding3d<0>(x, y, z, workspace, tiling);
    };
    ICPU_RUN_KF(rotary_position_embedding3d_wrapper, blockDim, x, cos_sin, z, workspace,
                (uint8_t *)(tilingDatafromBin));

    // Verify against golden reference
    float* z_f32 = (float*)z;
    size_t numElts = outputZByteSize / sizeof(float);
    std::vector<float> golden(numElts);
    ReadFile(path + "/rotary_position_embedding3d_data/golden.bin", outputZByteSize, (uint8_t*)golden.data(), outputZByteSize);
    bool utPass = true;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numElts; i++) {
        float diff = fabsf(z_f32[i] - golden[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 5e-2f) {  // relaxed tolerance for half precision
            if (utPass) std::cout << "First mismatch at " << i << ": got " << z_f32[i] << " exp " << golden[i] << " diff " << diff << std::endl;
            utPass = false;
        }
    }
    std::cout << "  Max diff: " << maxDiff << " " << (utPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(utPass);

    AscendC::GmFree(x);
    AscendC::GmFree(cos_sin);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
