/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_rotary_position_embedding_grad.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rotary_position_embedding_grad(
    GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
    GM_ADDR workspace, GM_ADDR tiling);

class rotary_position_embedding_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "rotary_position_embedding_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotary_position_embedding_grad_test TearDown\n" << endl;
    }
};

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp32_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp32_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1101);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp32_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp32_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp32_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(111);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp32_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1111);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp32_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp32_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp32_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(121);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp32_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1121);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp32_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(21);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp32_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(float);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1021);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp16_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp16_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1100);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp16_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_fp16_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp16_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(110);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp16_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1110);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp16_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_fp16_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1010);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp16_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(120);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp16_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1120);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp16_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_fp16_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(half);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1020);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_bf16_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_bf16_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1102);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_bf16_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BSND_bf16_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_bf16_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(112);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_bf16_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1112);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_bf16_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_BNSD_bf16_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1012);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_bf16_pad_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(122);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_bf16_pad_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 10;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1122);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_bf16_align_needbackward)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(22);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rotary_position_embedding_grad_test, test_case_mode_0_SBND_bf16_align_single)
{
    uint32_t B = 2, S = 64, N = 4, D = 64;
    size_t inputXByteSize = B * S * N * D * sizeof(DT_BF16);
    size_t inputcosByteSize = 1 * S * 1 * D * sizeof(DT_BF16);
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputXByteSize + 32);
    uint8_t* cosGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* sinGrad = (uint8_t*)AscendC::GmAlloc(inputcosByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1022);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(cosGrad);
    AscendC::GmFree(sinGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 62] "BSND" float16 small noNeedBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp16_small_noNeedBackward)
{
    size_t inputGradByteSize = 1 * 64 * 2 * 62 * sizeof(half);
    size_t inputCosByteSize = 1 * 64 * 1 * 62 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(20000);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 62] "BSND" float16 small needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp16_small_needBackward)
{
    size_t inputGradByteSize = 1 * 64 * 2 * 62 * sizeof(half);
    size_t inputCosByteSize = 1 * 64 * 1 * 62 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(21000);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 62] "BSND" bfloat16 small noNeedBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_bf16_small_noNeedBackward)
{
    size_t inputGradByteSize = 1 * 64 * 2 * 62 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 64 * 1 * 62 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(20010);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 62] "BSND" bfloat16 small needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_bf16_small_needBackward)
{
    size_t inputGradByteSize = 1 * 64 * 2 * 62 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 64 * 1 * 62 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(21010);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 62] "BSND" float32 small no needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp32_small_noNeedBackward)
{
    size_t inputGradByteSize = 1 * 64 * 2 * 62 * sizeof(float);
    size_t inputCosByteSize = 1 * 64 * 1 * 62 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(20020);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 62] "BSND" float32 small needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp32_small_needBackward)
{
    size_t inputGradByteSize = 1 * 64 * 2 * 62 * sizeof(float);
    size_t inputCosByteSize = 1 * 64 * 1 * 62 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(21020);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 1024, 20, 254] "BSND" float16 large noNeedBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp16_large_noNeedBackward)
{
    size_t inputGradByteSize = 1 * 1024 * 20 * 254 * sizeof(half);
    size_t inputCosByteSize = 1 * 1024 * 1 * 254 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(20100);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 1024, 20, 254] "BSND" float16 large needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp16_large_needBackward)
{
    size_t inputGradByteSize = 1 * 1024 * 20 * 254 * sizeof(half);
    size_t inputCosByteSize = 1 * 1024 * 1 * 62 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(21100);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 1024, 20, 254] "BSND" bfloat16 large noNeedBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_bf16_large_noNeedBackward)
{
    size_t inputGradByteSize = 1 * 1024 * 20 * 254 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 1024 * 1 * 254 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(20110);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 1024, 20, 254] "BSND" bfloat16 large needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_bf16_large_needBackward)
{
    size_t inputGradByteSize = 1 * 1024 * 20 * 254 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 1024 * 1 * 62 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(21110);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 1024, 20, 254] "BSND" float32 large no needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp32_large_noNeedBackward)
{
    size_t inputGradByteSize = 1 * 1024 * 20 * 254 * sizeof(float);
    size_t inputCosByteSize = 1 * 1024 * 1 * 62 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(20120);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 1024, 20, 254] "BSND" float32 large needBackward
TEST_F(rotary_position_embedding_grad_test, test_case_mode_1_fp32_large_needBackward)
{
    size_t inputGradByteSize = 1 * 1024 * 20 * 254 * sizeof(float);
    size_t inputCosByteSize = 1 * 1024 * 1 * 254 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t inputXByteSize = inputGradByteSize;
    size_t outputDxByteSize = inputXByteSize;
    size_t outputDcosByteSize = inputCosByteSize;
    size_t outputDsinByteSize = inputSinByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputGradByteSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(inputCosByteSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(inputSinByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputDxByteSize);
    uint8_t* dcos = (uint8_t*)AscendC::GmAlloc(outputDcosByteSize);
    uint8_t* dsin = (uint8_t*)AscendC::GmAlloc(outputDsinByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingGradTilingData* tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(21120);
    ICPU_RUN_KF(
        rotary_position_embedding_grad, blockDim, grad, cos, sin, x, dx, dcos, dsin, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dcos);
    AscendC::GmFree(dsin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
