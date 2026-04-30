/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/BatchToSpace_tiling_data.h"
#include "../../../op_host/BatchToSpace_tiling.h"

using namespace std;
using namespace optiling;

// 辅助函数：打印和验证TilingData
void PrintAndVerifyTilingData(const optiling::BatchtospaceTilingData& tilingData,
                               const std::vector<int64_t>& expectInShape,
                               const std::vector<int64_t>& expectOutShape,
                               int64_t expectBlockSize,
                               const std::vector<int64_t>& expectCrops)
{
    std::cout << "\n========== TilingData Details ==========" << std::endl;
    
    std::cout << "inShape: [" << tilingData.inShape[0] << ", " << tilingData.inShape[1] << ", "
              << tilingData.inShape[2] << ", " << tilingData.inShape[3] << "]" << std::endl;
    std::cout << "expect:  [" << expectInShape[0] << ", " << expectInShape[1] << ", "
              << expectInShape[2] << ", " << expectInShape[3] << "]" << std::endl;
    
    std::cout << "outShape: [" << tilingData.outShape[0] << ", " << tilingData.outShape[1] << ", "
              << tilingData.outShape[2] << ", " << tilingData.outShape[3] << "]" << std::endl;
    std::cout << "expect:   [" << expectOutShape[0] << ", " << expectOutShape[1] << ", "
              << expectOutShape[2] << ", " << expectOutShape[3] << "]" << std::endl;
    
    std::cout << "blockSize: " << tilingData.blockSize << " (expect: " << expectBlockSize << ")" << std::endl;
    
    std::cout << "crops: [[" << tilingData.crops[0][0] << ", " << tilingData.crops[0][1] << "], "
              << "[" << tilingData.crops[1][0] << ", " << tilingData.crops[1][1] << "]]" << std::endl;
    std::cout << "expect: [[" << expectCrops[0] << ", " << expectCrops[1] << "], "
              << "[" << expectCrops[2] << ", " << expectCrops[3] << "]]" << std::endl;
    
    std::cout << "totalCount: " << tilingData.totalCount << std::endl;
    std::cout << "perCoreCount: " << tilingData.perCoreCount << std::endl;
    std::cout << "ubAxis: " << tilingData.ubAxis << std::endl;
    std::cout << "ubFactor: " << tilingData.ubFactor << std::endl;
    std::cout << "bufferSize: " << tilingData.bufferSize << std::endl;
    
    std::cout << "=========================================\n" << std::endl;
    
    // 验证关键字段
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(tilingData.inShape[i], expectInShape[i]) << "inShape[" << i << "] mismatch";
        EXPECT_EQ(tilingData.outShape[i], expectOutShape[i]) << "outShape[" << i << "] mismatch";
    }
    EXPECT_EQ(tilingData.blockSize, expectBlockSize);
    EXPECT_EQ(tilingData.crops[0][0], expectCrops[0]);
    EXPECT_EQ(tilingData.crops[0][1], expectCrops[1]);
    EXPECT_EQ(tilingData.crops[1][0], expectCrops[2]);
    EXPECT_EQ(tilingData.crops[1][1], expectCrops[3]);
    
    // 验证ubFactor合理性：ubFactor不应该超过outShape[ubAxis]
    if (tilingData.ubAxis < 4) {
        EXPECT_LE(tilingData.ubFactor, tilingData.outShape[tilingData.ubAxis]) 
            << "ubFactor should not exceed outShape[ubAxis], ubFactor=" << tilingData.ubFactor 
            << ", outShape[ubAxis]=" << tilingData.outShape[tilingData.ubAxis];
        EXPECT_GT(tilingData.ubFactor, 0) << "ubFactor should be positive";
    }
}

class BatchToSpaceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchToSpaceTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchToSpaceTilingTest TearDown" << std::endl;
    }
};

// Large C场景1详细验证: C=256, 验证完整的TilingData和ubFactor合理性
// 分析ubFactor计算过程（遵循文档逻辑）：
// outShape=[1,3,4,256], bufferSize=32KB=32768 bytes, dtypeSize=4 (FLOAT)
// tmpBufferSize = 32768/4 = 8192 elements
// 从C轴(axis=3)开始，tmpBufferSize每次更新为factor：
//   axisSize = CeilAlign(256,32) = 256
//   factor = 8192/256 = 32 > 0 → tmpBufferSize = 32 (表示能容纳32个C轴元素)
// W轴(axis=2):
//   axisSize = 4
//   factor = 32/4 = 8 > 0 → tmpBufferSize = 8 (表示能容纳8个(W,C)元素组)
// H轴(axis=1):
//   axisSize = 3
//   factor = 8/3 = 2 > 0 → tmpBufferSize = 2 (表示能容纳2个(H,W,C)元素组)
// N轴(axis=0):
//   axisSize = 1
//   factor = 2/1 = 2 > 0 → tmpBufferSize = 2
// 所有轴都能容纳，ubAxis=0(N轴), ubFactor=outShape[0]=1
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_256_verify_tilingdata)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {1, 1, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 256}, {4, 2, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 3, 4, 256}, {1, 3, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::vector<int64_t> expectInShape = {4, 2, 2, 256};
    std::vector<int64_t> expectOutShape = {1, 3, 4, 256};
    std::vector<int64_t> expectCrops = {1, 1, 0, 0};
    
    PrintAndVerifyTilingData(*tilingDataPtr, expectInShape, expectOutShape, 2, expectCrops);
    
    // 验证IsLargeC判断：C=256 > CacheLineSize/2=128
    uint64_t tilingKey = context->GetTilingKey();
    EXPECT_EQ(tilingKey, 0) << "C=256 should be Large C scenario";
    
    // 验证bufferSize计算：min(ubSize/2, 64KB) = min(65536/2, 65536) = 32768 bytes
    EXPECT_EQ(tilingDataPtr->bufferSize, 32768);
    
    // 验证ubAxis=0(N轴), ubFactor=1(所有轴都能容纳时，ubFactor=outShape[0])
    EXPECT_EQ(tilingDataPtr->ubAxis, 0) << "ubAxis should be 0 (N axis)";
    EXPECT_EQ(tilingDataPtr->ubFactor, 1) << "ubFactor should be outShape[0]=1";
    
    // 验证totalCount：不需要切分，totalCount = outShape[0] = 1
    EXPECT_EQ(tilingDataPtr->totalCount, 1);
    
    // 验证核数设置
    uint32_t blockDim = context->GetBlockDim();
    EXPECT_GT(blockDim, 0);
    EXPECT_LE(blockDim, compileInfo.coreNum);
    
    DestroyTilingContext(context);
}

// Large C场景：超大shape导致ubFactor需要切分 - W轴切分
// outShape=[2, 100, 100, 256], bufferSize=32KB
// 分析（tmpBufferSize每次更新为factor）：
// tmpBufferSize = 8192 elements
// C轴(axis=3): axisSize=256, factor=8192/256=32 → tmpBufferSize=32
// W轴(axis=2): axisSize=100, factor=32/100=0 < 1 → 触发切分！
//   ubAxis=2(W轴)
//   ubFactor=tmpBufferSize=32 (表示能容纳32个(W,C)元素组，即32个C轴元素)
//   验证：ubFactor(32) <= outShape[2](100) ✓
// totalCount = 2 * ceil(100/32) = 2 * 4 = 8
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_w_axis_split)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{8, 50, 50, 256}, {8, 50, 50, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 100, 100, 256}, {2, 100, 100, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::cout << "\n=== W轴切分场景 ===" << std::endl;
    std::cout << "outShape: [2, 100, 100, 256]" << std::endl;
    std::cout << "bufferSize: " << tilingDataPtr->bufferSize << " bytes" << std::endl;
    std::cout << "ubAxis: " << tilingDataPtr->ubAxis << " (expect: 2 for W axis)" << std::endl;
    std::cout << "ubFactor: " << tilingDataPtr->ubFactor << " (expect: 32)" << std::endl;
    std::cout << "totalCount: " << tilingDataPtr->totalCount << " (expect: 8)" << std::endl;
    
    // 验证ubAxis=2(W轴)
    EXPECT_EQ(tilingDataPtr->ubAxis, 2);
    
    // 验证ubFactor=32，来自C轴的factor值
    EXPECT_EQ(tilingDataPtr->ubFactor, 32);
    EXPECT_LE(tilingDataPtr->ubFactor, 100) << "ubFactor should not exceed outShape[2]";
    
    // 验证totalCount = 2 * ceil(100/32) = 8
    EXPECT_EQ(tilingDataPtr->totalCount, 8);
    
    uint64_t tilingKey = context->GetTilingKey();
    EXPECT_EQ(tilingKey, 0) << "C=256 should be Large C";
    
    DestroyTilingContext(context);
}

// Large C场景：H轴需要切分
// outShape=[1, 100, 4, 256], bufferSize=32KB
// 分析（tmpBufferSize每次更新为factor）：
// tmpBufferSize = 8192
// C轴(axis=3): axisSize=256, factor=32 → tmpBufferSize=32
// W轴(axis=2): axisSize=4, factor=8 → tmpBufferSize=8
// H轴(axis=1): axisSize=100, factor=8/100=0 < 1 → 触发切分！
//   ubAxis=1(H轴)
//   ubFactor=tmpBufferSize=8 (表示能容纳8个(H,W,C)元素组，即8个(W,C)元素组)
//   验证：ubFactor(8) <= outShape[1](100) ✓
// totalCount = 1 * ceil(100/8) = 13
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_h_axis_split)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 50, 2, 256}, {4, 50, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 100, 4, 256}, {1, 100, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::cout << "\n=== H轴切分场景 ===" << std::endl;
    std::cout << "outShape: [1, 100, 4, 256]" << std::endl;
    std::cout << "bufferSize: " << tilingDataPtr->bufferSize << " bytes" << std::endl;
    std::cout << "ubAxis: " << tilingDataPtr->ubAxis << " (expect: 1 for H axis)" << std::endl;
    std::cout << "ubFactor: " << tilingDataPtr->ubFactor << " (expect: 8)" << std::endl;
    std::cout << "totalCount: " << tilingDataPtr->totalCount << " (expect: 13)" << std::endl;
    
    // 验证ubAxis=1(H轴)
    EXPECT_EQ(tilingDataPtr->ubAxis, 1);
    
    // 验证ubFactor=8，来自W轴的factor值
    EXPECT_EQ(tilingDataPtr->ubFactor, 8);
    EXPECT_LE(tilingDataPtr->ubFactor, 100) << "ubFactor should not exceed outShape[1]";
    
    // 验证totalCount = 1 * ceil(100/8) = 13
    EXPECT_EQ(tilingDataPtr->totalCount, 13);
    
    uint64_t tilingKey = context->GetTilingKey();
    EXPECT_EQ(tilingKey, 0);
    
    DestroyTilingContext(context);
}

// Large C场景3详细验证: C=129 (边界值)
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_129_boundary_verify)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 129}, {4, 2, 2, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 129}, {1, 4, 4, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::vector<int64_t> expectInShape = {4, 2, 2, 129};
    std::vector<int64_t> expectOutShape = {1, 4, 4, 129};
    std::vector<int64_t> expectCrops = {0, 0, 0, 0};
    
    PrintAndVerifyTilingData(*tilingDataPtr, expectInShape, expectOutShape, 2, expectCrops);
    
    // 验证IsLargeC判断：C=129 > 128
    uint64_t tilingKey = context->GetTilingKey();
    EXPECT_EQ(tilingKey, 0) << "C=129 should be Large C (129 > 128)";
    
    DestroyTilingContext(context);
}

// Large C场景4详细验证: 大batch场景 [16, 8, 8, 256]
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_big_batch_verify)
{
    struct BatchtospaceCompileInfo compileInfo = {32, 65536};
    
    std::vector<int64_t> cropsData = {2, 2, 2, 2};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{16, 8, 8, 256}, {16, 8, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 14, 14, 256}, {4, 14, 14, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::vector<int64_t> expectInShape = {16, 8, 8, 256};
    std::vector<int64_t> expectOutShape = {4, 14, 14, 256};
    std::vector<int64_t> expectCrops = {2, 2, 2, 2};
    
    PrintAndVerifyTilingData(*tilingDataPtr, expectInShape, expectOutShape, 2, expectCrops);
    
    // 验证batch变换：16 -> 4 (16/(2*2)=4)
    EXPECT_EQ(tilingDataPtr->inShape[0], 16);
    EXPECT_EQ(tilingDataPtr->outShape[0], 4);
    
    // 验证H/W变换：H_out=8*2-2-2=14, W_out=8*2-2-2=14
    EXPECT_EQ(tilingDataPtr->outShape[1], 14);
    EXPECT_EQ(tilingDataPtr->outShape[2], 14);
    
    // 验证多核分配
    uint32_t blockDim = context->GetBlockDim();
    EXPECT_GT(blockDim, 0);
    EXPECT_LE(blockDim, 32);  // 最多32核
    
    // 验证totalCount和perCoreCount合理
    EXPECT_GT(tilingDataPtr->totalCount, 0);
    EXPECT_GT(tilingDataPtr->perCoreCount, 0);
    
    DestroyTilingContext(context);
}

// Large C场景5详细验证: FLOAT16 dtype
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_float16_verify)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 256}, {4, 2, 2, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 256}, {1, 4, 4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::vector<int64_t> expectInShape = {4, 2, 2, 256};
    std::vector<int64_t> expectOutShape = {1, 4, 4, 256};
    std::vector<int64_t> expectCrops = {0, 0, 0, 0};
    
    PrintAndVerifyTilingData(*tilingDataPtr, expectInShape, expectOutShape, 2, expectCrops);
    
    // 验证dtype对bufferSize的影响
    // FLOAT16 (2 bytes), bufferSize应该比FLOAT (4 bytes)更大
    // bufferSize = min(ubSize/2, 64KB) / 2 = 32768/2 = 16384 elements
    EXPECT_GT(tilingDataPtr->bufferSize, 8192);  // FLOAT16的bufferSize应该更大
    
    DestroyTilingContext(context);
}

// Large C场景6详细验证: INT64 dtype
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_int64_verify)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 256}, {4, 2, 2, 256}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 256}, {1, 4, 4, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    auto context = CreateTilingContext(tilingContextPara);
    ASSERT_NE(context, nullptr);
    
    auto status = BatchtospaceTilingFunc(context);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    
    auto* tilingDataPtr = context->GetTilingData<optiling::BatchtospaceTilingData>();
    ASSERT_NE(tilingDataPtr, nullptr);
    
    std::vector<int64_t> expectInShape = {4, 2, 2, 256};
    std::vector<int64_t> expectOutShape = {1, 4, 4, 256};
    std::vector<int64_t> expectCrops = {0, 0, 0, 0};
    
    PrintAndVerifyTilingData(*tilingDataPtr, expectInShape, expectOutShape, 2, expectCrops);
    
    // 验证dtype对bufferSize的影响
    // INT64 (8 bytes), bufferSize应该最小
    // bufferSize = min(ubSize/2, 64KB) / 8 = 32768/8 = 4096 elements
    EXPECT_LT(tilingDataPtr->bufferSize, 8192);  // INT64的bufferSize应该最小
    
    DestroyTilingContext(context);
}

// Large C场景5: block_size=4的大block场景 [16, 4, 4, 384]
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_blocksize_4)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {3, 3, 2, 2};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{16, 4, 4, 384}, {16, 4, 4, 384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 13, 14, 384}, {1, 13, 14, 384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(4));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // 验证:
    // inShape = [16, 4, 4, 384]
    // outShape = [1, 13, 14, 384]
    // N_out = 16/(4*4) = 1
    // H_out = 4*4-3-3 = 13
    // W_out = 4*4-2-2 = 14
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景6: FLOAT16 dtype, C=256
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_float16)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 256}, {4, 2, 2, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 256}, {1, 4, 4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // FLOAT16的dtypeSize=2, bufferSize可以更大
    // bufferSize = min(ubSize/2, 64KB) / sizeof(FLOAT16)
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景7: BF16 dtype, C=512
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_bf16)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {1, 1, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 4, 4, 512}, {4, 4, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 7, 8, 512}, {1, 7, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景8: INT32 dtype, C=256 (虽然通常不会用这么大的INT32，但测试dtype支持)
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_int32)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 256}, {4, 2, 2, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 256}, {1, 4, 4, 256}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景9: INT64 dtype, C=256
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_int64)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 256}, {4, 2, 2, 256}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 256}, {1, 4, 4, 256}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // INT64的dtypeSize=8, bufferSize会相应减小
    // bufferSize = min(ubSize/2, 64KB) / 8 ≈ 8K elements
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景10: 大H/W场景 [4, 16, 16, 256]
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_big_hw)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {4, 4, 4, 4};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 16, 16, 256}, {4, 16, 16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 28, 28, 256}, {1, 28, 28, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // inShape = [4, 16, 16, 256]
    // outShape = [1, 28, 28, 256]
    // totalCount = 28 * 28 = 784 (假设ubAxis切分在H或W轴)
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景11: 无裁剪场景
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_no_crop)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{8, 8, 8, 256}, {8, 8, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 16, 16, 256}, {2, 16, 16, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景12: 只有H方向裁剪
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_crop_h_only)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {2, 2, 0, 0};  // 只裁剪H
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 4, 4, 256}, {4, 4, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 6, 8, 256}, {1, 6, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // H_out = 4*2-2-2 = 6
    // W_out = 4*2-0-0 = 8
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景13: 只有W方向裁剪
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_crop_w_only)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 1, 1};  // 只裁剪W
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 4, 4, 256}, {4, 4, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 8, 6, 256}, {1, 8, 6, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // H_out = 4*2-0-0 = 8
    // W_out = 4*2-1-1 = 6
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Large C场景14: 不对称裁剪
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_large_c_asymmetric_crop)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {3, 1, 2, 0};  // 不对称裁剪
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 4, 4, 256}, {4, 4, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 6, 256}, {1, 4, 6, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 0;
    
    // H_out = 4*2-3-1 = 4
    // W_out = 4*2-2-0 = 6
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Small C场景: C=64 (Small C场景对比)
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_small_c_64)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 64}, {4, 2, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},  // C=64 < 128
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 64}, {1, 4, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 1;  // TILING_KEY_IS_SMALL_C (C=64 < 128)
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}

// Small C边界场景: C=128 (刚好等于边界，应该走Small C)
TEST_F(BatchToSpaceTilingTest, BatchToSpace_tiling_small_c_128_boundary)
{
    struct BatchtospaceCompileInfo compileInfo = {8, 65536};
    
    std::vector<int64_t> cropsData = {0, 0, 0, 0};
    
    gert::TilingContextPara tilingContextPara("Batchtospace",
        {
            {{{4, 2, 2, 128}, {4, 2, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},  // C=128 = CacheLine/2
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 4, 128}, {1, 4, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    tilingContextPara.attrs.emplace_back("block_size", static_cast<int64_t>(2));
    tilingContextPara.SetTensorData(1, cropsData.data(), cropsData.size() * sizeof(int64_t));
    
    uint64_t expectTilingKey = 1;  // TILING_KEY_IS_SMALL_C (C=128 <= 128)
    
    std::vector<size_t> expectWorkspaces = {16777216};
    
    auto result = ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
    EXPECT_EQ(result, true);
}