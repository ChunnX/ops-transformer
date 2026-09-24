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
#include <cstring>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/gather_pa_kv_cache_tiling.h"

using namespace std;
using namespace ge;

namespace {
struct GatherPaKvCacheTilingFields {
    int32_t blockSize;
    int32_t numTokens;
    int32_t numblkTabCol;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    int32_t typeByte;
    int32_t hasSeqStarts;
    int32_t isSeqLensCumsum;
    int64_t kCacheBlockStride;
    int64_t vCacheBlockStride;
};

GatherPaKvCacheTilingFields ReadTilingFields(const TilingInfo &tilingInfo)
{
    GatherPaKvCacheTilingFields fields{};
    if (tilingInfo.tilingDataSize >= sizeof(fields)) {
        std::memcpy(&fields, tilingInfo.tilingData.get(), sizeof(fields));
    }
    return fields;
}
} // namespace

class GatherPaKvCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "GatherPaKvCacheTiling SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "GatherPaKvCacheTiling TearDown" << std::endl;
    }
};

TEST_F(GatherPaKvCacheTiling, GatherPaKvCacheTiling_test_case0) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("GatherPaKvCache",
        {
            {{{30, 101, 128, 32}, {30, 101, 128, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // keyCache
            {{{30, 127, 128, 32}, {30, 127, 128, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // valueCache
            {{{36, 8}, {36, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // blocktables
            {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND}, // seqLens
            {{{18933, 3232}, {18933, 3232}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // key
            {{{18933, 4064}, {18933, 4064}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // value
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, // seqOffset
            
        },
        {
            {{{18933, 3232}, {18933, 3232}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // key
            {{{18933, 4064}, {18933, 4064}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 577;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GatherPaKvCacheTiling, GatherPaKvCacheTiling_test_case1) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("GatherPaKvCache",
        {
            {{{128, 128, 16, 144}, {128, 128, 16, 144}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // keyCache
            {{{128, 128, 16, 128}, {128, 128, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // valueCache
            {{{16, 12}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND}, // blocktables
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND}, // seqLens
            {{{8931, 16, 144}, {8931, 16, 144}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{8931, 16, 128}, {8931, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND}, // seqOffset
            
        },
        {
            {{{8931, 16, 144}, {8931, 16, 144}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{8931, 16, 128}, {8931, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 619;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GatherPaKvCacheTiling, GatherPaKvCacheTiling_test_case2) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("GatherPaKvCache",
        {
            {{{128, 128, 16, 144}, {128, 128, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // keyCache
            {{{128, 128, 16, 128}, {128, 128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // valueCache
            {{{16, 12}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND}, // blocktables
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND}, // seqLens
            {{{9547, 16, 144}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{9547, 16, 128}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // value
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND}, // seqOffset
            
        },
        {
            {{{9547, 16, 144}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{9547, 16, 128}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 618;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GatherPaKvCacheTiling, flattenedStorageShapeUsesLogicalViewShape) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("GatherPaKvCache",
        {
            {{{1093, 128, 1, 512}, {80584704}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // keyCache view
            {{{1093, 128, 1, 64}, {80584704}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // valueCache view
            {{{8, 4}, {32}}, ge::DT_INT32, ge::FORMAT_ND}, // blockTables view
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // seqLens
            {{{512, 1, 512}, {262144}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key view
            {{{512, 1, 64}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value view
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // seqOffset
        },
        {
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{512, 1, 64}, {512, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo
    );

    auto &keyCacheDesc = tilingContextPara.inputTensorDesc_[0];
    keyCacheDesc.stride_.SetDimNum(4);
    keyCacheDesc.stride_.SetStride(0, 73728);
    keyCacheDesc.stride_.SetStride(1, 512);
    keyCacheDesc.stride_.SetStride(2, 512);
    keyCacheDesc.stride_.SetStride(3, 1);
    keyCacheDesc.hasStride_ = true;

    auto &valueCacheDesc = tilingContextPara.inputTensorDesc_[1];
    valueCacheDesc.stride_.SetDimNum(4);
    valueCacheDesc.stride_.SetStride(0, 73728);
    valueCacheDesc.stride_.SetStride(1, 64);
    valueCacheDesc.stride_.SetStride(2, 64);
    valueCacheDesc.stride_.SetStride(3, 1);
    valueCacheDesc.hasStride_ = true;

    auto &blockTablesDesc = tilingContextPara.inputTensorDesc_[2];
    blockTablesDesc.stride_.SetDimNum(2);
    blockTablesDesc.stride_.SetStride(0, 4);
    blockTablesDesc.stride_.SetStride(1, 1);
    blockTablesDesc.hasStride_ = true;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(GatherPaKvCacheTilingFields));
    auto tilingData = ReadTilingFields(tilingInfo);
    EXPECT_EQ(tilingData.blockSize, 128);
    EXPECT_EQ(tilingData.tokenSizeK, 512);
    EXPECT_EQ(tilingData.tokenSizeV, 64);
    EXPECT_EQ(tilingData.numTokens, 8);
    EXPECT_EQ(tilingData.numblkTabCol, 4);
    EXPECT_EQ(tilingData.kCacheBlockStride, 73728);
}

TEST_F(GatherPaKvCacheTiling, paNzFlattenedStorageShapeUsesLogicalDimensions) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("GatherPaKvCache",
        {
            {{{30, 101, 128, 32}, {12410880}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // keyCache view
            {{{30, 127, 128, 32}, {15605760}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // valueCache view
            {{{36, 8}, {288}}, ge::DT_INT32, ge::FORMAT_ND}, // blockTables view
            {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND}, // seqLens
            {{{18933, 3232}, {61191456}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // key view
            {{{18933, 4064}, {76943712}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // value view
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, // seqOffset
        },
        {
            {{{18933, 3232}, {61191456}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // key view
            {{{18933, 4064}, {76943712}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // value view
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo
    );

    auto &keyCacheDesc = tilingContextPara.inputTensorDesc_[0];
    keyCacheDesc.stride_.SetDimNum(4);
    keyCacheDesc.stride_.SetStride(0, 101 * 128 * 32);
    keyCacheDesc.stride_.SetStride(1, 128 * 32);
    keyCacheDesc.stride_.SetStride(2, 32);
    keyCacheDesc.stride_.SetStride(3, 1);
    keyCacheDesc.hasStride_ = true;

    auto &valueCacheDesc = tilingContextPara.inputTensorDesc_[1];
    valueCacheDesc.stride_.SetDimNum(4);
    valueCacheDesc.stride_.SetStride(0, 127 * 128 * 32);
    valueCacheDesc.stride_.SetStride(1, 128 * 32);
    valueCacheDesc.stride_.SetStride(2, 32);
    valueCacheDesc.stride_.SetStride(3, 1);
    valueCacheDesc.hasStride_ = true;

    auto &blockTablesDesc = tilingContextPara.inputTensorDesc_[2];
    blockTablesDesc.stride_.SetDimNum(2);
    blockTablesDesc.stride_.SetStride(0, 8);
    blockTablesDesc.stride_.SetStride(1, 1);
    blockTablesDesc.hasStride_ = true;

    auto &keyRefDesc = tilingContextPara.inputTensorDesc_[4];
    keyRefDesc.stride_.SetDimNum(2);
    keyRefDesc.stride_.SetStride(0, 3232);
    keyRefDesc.stride_.SetStride(1, 1);
    keyRefDesc.hasStride_ = true;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(GatherPaKvCacheTilingFields));
    auto tilingData = ReadTilingFields(tilingInfo);
    EXPECT_EQ(tilingData.blockSize, 128);
    EXPECT_EQ(tilingData.tokenSizeK, 3232);
    EXPECT_EQ(tilingData.tokenSizeV, 4064);
    EXPECT_EQ(tilingData.numTokens, 36);
    EXPECT_EQ(tilingData.numblkTabCol, 8);
}
