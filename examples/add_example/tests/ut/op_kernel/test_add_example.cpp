/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_example.cpp
 * \brief
 */

#include "../../../op_kernel/add_example.cpp"  
#include "add_example_tiling.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>  
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

class add_example_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_example_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_example_test TearDown\n" << endl;
    }
};

TEST_F(add_example_test, test_case_0)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t zByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(AddExampleTilingData);
    uint32_t blockDim = 8;

    // 分配全局内存（AscendC::GmAlloc 是 Ascend 高层 API，正确）
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddExampleTilingData* tilingDatafromBin = reinterpret_cast<AddExampleTilingData*>(tiling);
    tilingDatafromBin->totalLength = 32 * 4 * 4 * 4;
    tilingDatafromBin->tileNum = 8;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(add_example<0>,
        blockDim,
        x,
        y,
        z,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}