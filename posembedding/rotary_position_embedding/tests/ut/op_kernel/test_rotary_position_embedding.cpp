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
#include "test_rotary_position_embedding.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rotary_position_embedding(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                GM_ADDR workspace, GM_ADDR tiling);

class rotary_position_embedding_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rotary_position_embedding_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotary_position_embedding_test TearDown\n" << endl;
    }
};

// [1, 64, 2, 22] "BSND" float16 pad
TEST_F(rotary_position_embedding_test, test_case_mode_1_pad_fp16_001)
{
    size_t inputXByteSize = 1 * 64 * 2 * 22 * sizeof(half);
    size_t inputCosByteSize = 1 * 64 * 1 * 22 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2001);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 22] "BSND" bfloat16 pad
TEST_F(rotary_position_embedding_test, test_case_mode_1_pad_bf16_001)
{
    size_t inputXByteSize = 1 * 64 * 2 * 22 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 64 * 1 * 22 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2011);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 22] "BSND" float pad
TEST_F(rotary_position_embedding_test, test_case_mode_1_pad_fp32_001)
{
    size_t inputXByteSize = 1 * 64 * 2 * 22 * sizeof(float);
    size_t inputCosByteSize = 1 * 64 * 1 * 22 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2021);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 64] "BSND" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_fp16_001)
{
    size_t inputXByteSize = 1 * 64 * 2 * 64 * sizeof(half);
    size_t inputCosByteSize = 1 * 64 * 1 * 64 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 8192, 10, 256] "BSND" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_fp16_002)
{
    size_t inputXByteSize = 1 * 8192 * 10 * 256 * sizeof(half);
    size_t inputCosByteSize = 1 * 8192 * 1 * 256 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 64] "BSND" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_bf16_001)
{
    size_t inputXByteSize = 1 * 64 * 2 * 64 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 64 * 1 * 64 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2010);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 8192, 10, 256] "BSND" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_bf16_002)
{
    size_t inputXByteSize = 1 * 8192 * 10 * 256 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 8192 * 1 * 256 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2010);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 64, 2, 64] "BSND" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_fp32_001)
{
    size_t inputXByteSize = 1 * 64 * 2 * 64 * sizeof(float);
    size_t inputCosByteSize = 1 * 64 * 1 * 64 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2020);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 8192, 10, 256] "BSND" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_fp32_002)
{
    size_t inputXByteSize = 1 * 8192 * 10 * 256 * sizeof(float);
    size_t inputCosByteSize = 1 * 8192 * 1 * 256 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2020);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 2, 64, 64] "BNSD" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_bnsd_fp16_001)
{
    size_t inputXByteSize = 1 * 2 * 64 * 64 * sizeof(half);
    size_t inputCosByteSize = 1 * 1 * 64 * 64 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 10, 8192, 256] "BNSD" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_bnsd_fp16_002)
{
    size_t inputXByteSize = 1 * 10 * 8192 * 256 * sizeof(half);
    size_t inputCosByteSize = 1 * 1 * 8192 * 256 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 2, 64, 64] "BNSD" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_bnsd_bf16_001)
{
    size_t inputXByteSize = 1 * 2 * 64 * 64 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 1 * 64 * 64 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2010);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 10, 8192, 256] "BNSD" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_bnsd_bf16_002)
{
    size_t inputXByteSize = 1 * 10 * 8192 * 256 * sizeof(DT_BF16);
    size_t inputCosByteSize = 1 * 1 * 8192 * 256 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2010);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 2, 64, 64] "BNSD" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_bnsd_fp32_001)
{
    size_t inputXByteSize = 1 * 2 * 64 * 64 * sizeof(float);
    size_t inputCosByteSize = 1 * 1 * 64 * 64 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2020);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 10, 8192, 256] "BNSD" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_bnsd_fp32_002)
{
    size_t inputXByteSize = 1 * 10 * 8192 * 256 * sizeof(float);
    size_t inputCosByteSize = 1 * 1 * 8192 * 256 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2020);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [64, 1, 2, 64] "SBND" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_sbnd_fp16_001)
{
    size_t inputXByteSize = 64 * 1 * 2 * 64 * sizeof(half);
    size_t inputCosByteSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 1, 10, 256] "SBND" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_sbnd_fp16_002)
{
    size_t inputXByteSize = 8192 * 1 * 10 * 256 * sizeof(half);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [64, 1, 2, 64] "SBND" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_sbnd_bf16_001)
{
    size_t inputXByteSize = 64 * 1 * 2 * 64 * sizeof(DT_BF16);
    size_t inputCosByteSize = 64 * 1 * 1 * 64 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2010);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 1, 10, 256] "SBND" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_sbnd_bf16_002)
{
    size_t inputXByteSize = 8192 * 1 * 10 * 256 * sizeof(DT_BF16);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2010);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [64, 1, 2, 64] "SBND" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_sbnd_fp32_001)
{
    size_t inputXByteSize = 64 * 1 * 2 * 64 * sizeof(float);
    size_t inputCosByteSize = 64 * 1 * 1 * 64 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2020);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 1, 10, 256] "SBND" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_sbnd_fp32_002)
{
    size_t inputXByteSize = 8192 * 1 * 10 * 256 * sizeof(float);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2020);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 10, 256] "SBND" float16 splitbs
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bs_sbnd_fp16_001)
{
    size_t inputXByteSize = 8192 * 20 * 10 * 256 * sizeof(half);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2100);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 10, 256] "SBND" bfloat16 splitbs
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bs_sbnd_bf16_001)
{
    size_t inputXByteSize = 8192 * 20 * 10 * 256 * sizeof(DT_BF16);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2110);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 10, 256] "SBND" float32 splitbs
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bs_sbnd_fp32_001)
{
    size_t inputXByteSize = 8192 * 20 * 10 * 256 * sizeof(float);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2120);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 144, 256] "SBND" float16 splitbsn
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bsn_sbnd_fp16_001)
{
    size_t inputXByteSize = 8192 * 20 * 144 * 256 * sizeof(half);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2200);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 144, 256] "SBND" bfloat16 splitbsn
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bsn_sbnd_bf16_001)
{
    size_t inputXByteSize = 8192 * 20 * 144 * 256 * sizeof(DT_BF16);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2210);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 144, 256] "SBND" float32 splitbsn
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bsn_sbnd_fp32_001)
{
    size_t inputXByteSize = 8192 * 20 * 144 * 256 * sizeof(float);
    size_t inputCosByteSize = 8192 * 1 * 1 * 256 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2220);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 10, 255] "SBND" float16 splitbs pad
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bs_pad_sbnd_fp16_001)
{
    size_t inputXByteSize = 8192 * 20 * 10 * 255 * sizeof(half);
    size_t inputCosByteSize = 8192 * 1 * 1 * 255 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2101);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 10, 255] "SBND" bfloat16 splitbs
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bs_pad_sbnd_bf16_001)
{
    size_t inputXByteSize = 8192 * 20 * 10 * 255 * sizeof(DT_BF16);
    size_t inputCosByteSize = 8192 * 1 * 1 * 255 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2111);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 10, 255] "SBND" float32 splitbs
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bs_pad_sbnd_fp32_001)
{
    size_t inputXByteSize = 8192 * 20 * 10 * 255 * sizeof(float);
    size_t inputCosByteSize = 8192 * 1 * 1 * 255 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2121);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 144, 255] "SBND" float16 splitbsn
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bsn_pad_sbnd_fp16_001)
{
    size_t inputXByteSize = 8192 * 20 * 144 * 255 * sizeof(half);
    size_t inputCosByteSize = 8192 * 1 * 1 * 255 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2201);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 144, 255] "SBND" bfloat16 splitbsn
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bsn_pad_sbnd_bf16_001)
{
    size_t inputXByteSize = 8192 * 20 * 144 * 255 * sizeof(DT_BF16);
    size_t inputCosByteSize = 8192 * 1 * 1 * 255 * sizeof(DT_BF16);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2211);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [8192, 20, 144, 255] "SBND" float32 splitbsn
TEST_F(rotary_position_embedding_test, test_case_mode_1_split_bsn_pad_sbnd_fp32_001)
{
    size_t inputXByteSize = 8192 * 20 * 144 * 255 * sizeof(float);
    size_t inputCosByteSize = 8192 * 1 * 1 * 255 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(2221);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

/*************** test cases for mode 0: rotate_half ***************/
/****************************** BNSD ******************************/
// [2, 5, 8192, 128] "BNSD" float32 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnsd_fp32_aligned_001)
{
    size_t inputXByteSize = 2 * 5 * 8192 * 128 * sizeof(float);
    size_t inputCosByteSize = 8192 * 128 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [2, 5, 8192, 128] "BNSD" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnsd_bf16_aligned_001)
{
    size_t inputXByteSize = 2 * 5 * 8192 * 128 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 8192 * 128 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1013);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [2, 5, 8192, 124] "BNSD" float32 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnsd_fp32_unaligned_001)
{
    size_t inputXByteSize = 2 * 5 * 8192 * 124 * sizeof(float);
    size_t inputCosByteSize = 8192 * 124 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [2, 5, 8192, 120] "BNSD" bf16 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnsd_bf16_unaligned_001)
{
    size_t inputXByteSize = 2 * 5 * 8192 * 120 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 8192 * 120 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1013);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

/****************************** BSND ******************************/
// [4, 4096, 4, 128] "BSND" float32 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bsnd_fp32_aligned_001)
{
    size_t inputXByteSize = 4 * 4096 * 4 * 128 * sizeof(float);
    size_t inputCosByteSize = 4096 * 128 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1022);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 4096, 4, 128] "BSND" float16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bsnd_fp16_aligned_001)
{
    size_t inputXByteSize = 4 * 4096 * 4 * 128 * sizeof(half);
    size_t inputCosByteSize = 4096 * 128 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1022);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 4096, 4, 128] "BSND" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bsnd_bf16_aligned_001)
{
    size_t inputXByteSize = 4 * 4096 * 4 * 128 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4096 * 128 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1023);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 4096, 4, 60] "BSND" float32 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bsnd_fp32_unaligned_001)
{
    size_t inputXByteSize = 4 * 4096 * 4 * 60 * sizeof(float);
    size_t inputCosByteSize = 4096 * 60 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1022);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 4096, 4, 60] "BSND" float16 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bsnd_fp16_unaligned_001)
{
    size_t inputXByteSize = 4 * 4096 * 4 * 60 * sizeof(half);
    size_t inputCosByteSize = 4096 * 60 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1022);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 4096, 4, 120] "BSND" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bsnd_bf16_unaligned_001)
{
    size_t inputXByteSize = 4 * 4096 * 4 * 120 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4096 * 120 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1023);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

/****************************** SBND ******************************/
// [4096, 4, 32, 128] "SBND" float32 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_sbnd_fp32_aligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 32 * 128 * sizeof(float);
    size_t inputCosByteSize = 4096 * 128 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1032);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 32, 128] "SBND" float16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_sbnd_fp16_aligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 32 * 128 * sizeof(half);
    size_t inputCosByteSize = 4096 * 128 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1032);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 32, 128] "SBND" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_sbnd_bf16_aligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 32 * 128 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4096 * 128 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1033);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 6, 60] "SBND" float32 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_sbnd_fp32_unaligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 6 * 60 * sizeof(float);
    size_t inputCosByteSize = 4096 * 60 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1032);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 6, 60] "SBND" float16 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_sbnd_fp16_unaligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 6 * 60 * sizeof(half);
    size_t inputCosByteSize = 4096 * 60 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1032);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 6, 120]  "SBND" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_sbnd_bf16_unaligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 6 * 120 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4096 * 120 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1033);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

/****************************** NO_BROADCAST ******************************/
// [4096, 4, 2, 128] "NO_BROADCAST" float32 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_no_broadcast_fp32_aligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 2 * 128 * sizeof(float);
    size_t inputCosByteSize = 4096 * 4 * 2 * 128 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1042);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 2, 128] "NO_BROADCAST" float16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_no_broadcast_fp16_aligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 2 * 128 * sizeof(half);
    size_t inputCosByteSize = 4096 * 4 * 2 * 128 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1042);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 2, 128] "NO_BROADCAST" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_no_broadcast_bf16_aligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 2 * 128 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4096 * 4 * 2 * 128 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1043);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 6, 60] "NO_BROADCAST" float32 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_no_broadcast_fp32_unaligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 6 * 60 * sizeof(float);
    size_t inputCosByteSize = 4096 * 4 * 6 * 60 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1042);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 6, 60] "NO_BROADCAST" float16 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_no_broadcast_fp16_unaligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 6 * 60 * sizeof(half);
    size_t inputCosByteSize = 4096 * 4 * 6 * 60 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1042);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4096, 4, 6, 120]  "NO_BROADCAST" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_no_broadcast_bf16_unaligned_001)
{
    size_t inputXByteSize = 4096 * 4 * 6 * 120 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4096 * 4 * 6 * 120 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1043);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

/****************************** BND ******************************/
// [1, 4, 32, 128] "BND" float32 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnd_fp32_aligned_001)
{
    size_t inputXByteSize = 1 * 4 * 32 * 128 * sizeof(float);
    size_t inputCosByteSize = 1 * 128 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1052);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 4, 32, 128] "BND" float16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnd_fp16_aligned_001)
{
    size_t inputXByteSize = 1 * 4 * 32 * 128 * sizeof(half);
    size_t inputCosByteSize = 1 * 128 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1052);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 4, 32, 128] "BND" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnd_bf16_aligned_001)
{
    size_t inputXByteSize = 1 * 4 * 32 * 128 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 1 * 128 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1053);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 4, 6, 60] "BND" float32 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnd_fp32_unaligned_001)
{
    size_t inputXByteSize = 1 * 4 * 6 * 60 * sizeof(float);
    size_t inputCosByteSize = 1 * 60 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1052);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 4, 6, 60] "BND" float16 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnd_fp16_unaligned_001)
{
    size_t inputXByteSize = 1 * 4 * 6 * 60 * sizeof(half);
    size_t inputCosByteSize = 1 * 60 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1052);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 4, 6, 120]  "BND" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_bnd_bf16_unaligned_001)
{
    size_t inputXByteSize = 1 * 4 * 6 * 120 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 1 * 120 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1053);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

/****************************** R_B1SD ******************************/
// [4, 8, 2048, 128] "R_B1SD" float32 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_r_b1sd_fp32_aligned_001)
{
    size_t inputXByteSize = 4 * 8 * 2048 * 128 * sizeof(float);
    size_t inputCosByteSize = 4 * 1 * 2048 * 128 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1062);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 8, 2048, 128] "R_B1SD" float16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_r_b1sd_fp16_aligned_001)
{
    size_t inputXByteSize = 4 * 8 * 2048 * 128 * sizeof(half);
    size_t inputCosByteSize = 4 * 1 * 2048 * 128 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1062);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [4, 8, 2048, 128] "R_B1SD" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_r_b1sd_bf16_aligned_001)
{
    size_t inputXByteSize = 4 * 8 * 2048 * 128 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 4 * 1 * 2048 * 128 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1063);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [2, 4, 4096, 60] "R_B1SD" float32 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_r_b1sd_fp32_unaligned_001)
{
    size_t inputXByteSize = 2 * 4 * 4096 * 60 * sizeof(float);
    size_t inputCosByteSize = 2 * 1 * 4096 * 60 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1062);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [2, 4, 4096, 60] "R_B1SD" float16 unaligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_r_b1sd_fp16_unaligned_001)
{
    size_t inputXByteSize = 2 * 4 * 4096 * 60 * sizeof(half);
    size_t inputCosByteSize = 2 * 1 * 4096 * 60 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1062);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [1, 4, 4096, 120]  "R_B1SD" bf16 aligned
TEST_F(rotary_position_embedding_test, test_case_mode_0_r_b1sd_bf16_unaligned_001)
{
    size_t inputXByteSize = 1 * 4 * 4096 * 120 * sizeof(bfloat16_t);
    size_t inputCosByteSize = 1 * 1 * 4096 * 120 * sizeof(bfloat16_t);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char *path_ = get_current_dir_name();
    string path(path_);

    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ICPU_SET_TILING_KEY(1063);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}