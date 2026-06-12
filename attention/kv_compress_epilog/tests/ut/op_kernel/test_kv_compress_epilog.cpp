/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "test_kv_compress_epilog_tiling_def.h"
#include "opdev/op_test.h"
#include "opdev/op_tiling_test.h"

using namespace AscendC;

constexpr size_t BS = 1024;
constexpr size_t D = 256;
constexpr float SCALES_ATTR = 1.0f;

class KvCompressEpilogKernelTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvCompressEpilogKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvCompressEpilogKernelTest TearDown" << std::endl;
    }

    void SetUp() override
    {
        tilingData.set_bs(BS);
        tilingData.set_d(D);
        tilingData.set_kvCacheCol(D);
        tilingData.set_kvCacheRowStride(D);
        tilingData.set_kvCacheBlockSize(1);
        tilingData.set_kvCacheBlockStride(D);
        tilingData.set_scaleCol(1);
        tilingData.set_concatCol(D);
        tilingData.set_padCol(0);
        tilingData.set_quantMode(1);
        tilingData.set_roundScale(1);
        tilingData.set_perGroupSize(64);
        tilingData.set_rowOfFormerBlock(BS);
        tilingData.set_rowOfTailBlock(BS);
        tilingData.set_rowLoopOfFormerBlock(BS);
        tilingData.set_rowLoopOfTailBlock(BS);
        tilingData.set_rowFactor(1);
        tilingData.set_tailRowFactorOfFormerBlock(1);
        tilingData.set_tailRowFactorOfTailBlock(1);
        tilingData.set_scalesAttr(SCALES_ATTR);
    }

    optiling::KvCompressEpilogTilingData tilingData;
};

TEST_F(KvCompressEpilogKernelTest, tiling_data_basic)
{
    EXPECT_EQ(tilingData.bs, BS);
    EXPECT_EQ(tilingData.d, D);
    EXPECT_EQ(tilingData.quantMode, 1);
    EXPECT_EQ(tilingData.roundScale, 1);
    EXPECT_FLOAT_EQ(tilingData.scalesAttr, SCALES_ATTR);
    EXPECT_GT(tilingData.GetDataSize(), 0);
}
