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
#include "test_inplace_partial_rotary_mul_grad.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(
    GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
    GM_ADDR dx, GM_ADDR dcos, GM_ADDR dsin,
    GM_ADDR workspace, GM_ADDR tiling);

class inplace_partial_rotary_mul_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "inplace_partial_rotary_mul_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "inplace_partial_rotary_mul_grad_test TearDown\n" << endl;
    }
};

TEST_F(inplace_partial_rotary_mul_grad_test, test_case_mode1_interleave_fp32)
{
    uint32_t B = 1, S = 1, N = 1, D = 128;
    size_t inputDyByteSize = B * S * N * D * sizeof(float);
    size_t inputCosByteSize = B * S * N * D * sizeof(float);
    size_t tilingDataSize = sizeof(InplacePartialRotaryMulGradUTTilingData);

    // inplace: dy and dx share the same buffer
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputDyByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* x = nullptr;      // x is nullptr (unsupported)
    uint8_t* dx = dy;       // inplace: same as dy
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    system(
        "cp -r ../../../../../posembedding/inplace_partial_rotary_mul_grad/tests/ut/op_kernel/"
        "inplace_partial_rotary_mul_grad_data ./");
    system("chmod -R 755 ./inplace_partial_rotary_mul_grad_data/");
    system("cd ./inplace_partial_rotary_mul_grad_data/ && rm -rf ./*bin");
    system("cd ./inplace_partial_rotary_mul_grad_data/ && python3 gen_data.py 1 1 1 128 1 1 1 128 float32 float32");
    system("cd ./inplace_partial_rotary_mul_grad_data/ && python3 gen_tiling.py case0");
    char* path_ = get_current_dir_name();
    string path(path_);
    InplacePartialRotaryMulGradUTTilingData* tilingDatafromBin =
        reinterpret_cast<InplacePartialRotaryMulGradUTTilingData*>(tiling);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/dy.bin", inputDyByteSize, dy, inputDyByteSize);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/cos.bin", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/sin.bin", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(205);  // TILING_KEY_A: NO_BROADCAST → AAndB
    ICPU_RUN_KF(
        inplace_partial_rotary_mul_grad, blockDim, dy, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(inplace_partial_rotary_mul_grad_test, test_case_mode0_half_fp16)
{
    uint32_t B = 2, S = 4, N = 8, D = 64;
    size_t inputDyByteSize = B * S * N * D * sizeof(half);
    size_t inputCosByteSize = 1 * S * 1 * D * sizeof(half); // cos = (1, S, 1, D)
    size_t tilingDataSize = sizeof(InplacePartialRotaryMulGradUTTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputDyByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* x = nullptr;
    uint8_t* dx = dy;
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 32;

    system("cd ./inplace_partial_rotary_mul_grad_data/ && rm -rf ./*bin");
    system("cd ./inplace_partial_rotary_mul_grad_data/ && python3 gen_data.py 2 4 8 64 1 4 1 64 float16 float16");
    system("cd ./inplace_partial_rotary_mul_grad_data/ && python3 gen_tiling.py case1");
    char* path_ = get_current_dir_name();
    string path(path_);
    InplacePartialRotaryMulGradUTTilingData* tilingDatafromBin =
        reinterpret_cast<InplacePartialRotaryMulGradUTTilingData*>(tiling);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/dy.bin", inputDyByteSize, dy, inputDyByteSize);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/cos.bin", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/sin.bin", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/inplace_partial_rotary_mul_grad_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(204);  // TILING_KEY_AB: SBND → AB
    ICPU_RUN_KF(
        inplace_partial_rotary_mul_grad, blockDim, dy, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
