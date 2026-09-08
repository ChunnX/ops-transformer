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
#include <gtest/gtest.h>
#include "../../../op_host/dequant_rope_quant_kvcache_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

struct DequantRopeQuantKvcacheCompileInfo {};

class DequantRopeQuantKvcacheTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantRopeQuantKvcacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantRopeQuantKvcacheTiling TearDown" << std::endl;
    }
};

namespace {
constexpr int64_t TEST_BATCH = 2;
constexpr int64_t TEST_SEQ_LEN = 3;
constexpr int64_t TEST_Q_HEAD_NUM = 8;
constexpr int64_t TEST_KV_HEAD_NUM = 2;
constexpr int64_t TEST_HEAD_DIM = 128;
constexpr int64_t TEST_HIDDEN_SIZE =
    (TEST_Q_HEAD_NUM + TEST_KV_HEAD_NUM + TEST_KV_HEAD_NUM) * TEST_HEAD_DIM;
constexpr int64_t TEST_QUANT_SIZE = TEST_KV_HEAD_NUM * TEST_HEAD_DIM;
constexpr int64_t TEST_CACHE_SEQ_LEN = 1024;
constexpr size_t TEST_SCALE_K_INPUT_INDEX = 6;
constexpr size_t TEST_SCALE_V_INPUT_INDEX = 7;
constexpr size_t TEST_OFFSET_K_INPUT_INDEX = 8;
constexpr size_t TEST_OFFSET_V_INPUT_INDEX = 9;

gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

gert::TilingContextPara BuildOptionalInputShapeContext(const bool is3D,
                                                       const std::vector<int64_t> &weightScaleShape,
                                                       const std::vector<int64_t> &activationScaleShape,
                                                       const std::vector<int64_t> &biasShape)
{
    const std::vector<int64_t> xShape =
        is3D ? std::vector<int64_t>{TEST_BATCH, TEST_SEQ_LEN, TEST_HIDDEN_SIZE} :
               std::vector<int64_t>{TEST_BATCH, TEST_HIDDEN_SIZE};
    const std::vector<int64_t> sinCosShape =
        is3D ? std::vector<int64_t>{TEST_BATCH, TEST_SEQ_LEN, 1, TEST_HEAD_DIM} :
               std::vector<int64_t>{TEST_BATCH, TEST_HEAD_DIM};
    const std::vector<int64_t> qShape =
        is3D ? std::vector<int64_t>{TEST_BATCH, TEST_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM} :
               std::vector<int64_t>{TEST_BATCH, TEST_Q_HEAD_NUM, TEST_HEAD_DIM};
    const std::vector<int64_t> kvShape =
        is3D ? std::vector<int64_t>{TEST_BATCH, TEST_SEQ_LEN, TEST_KV_HEAD_NUM, TEST_HEAD_DIM} :
               std::vector<int64_t>{TEST_BATCH, TEST_KV_HEAD_NUM, TEST_HEAD_DIM};
    const std::vector<int64_t> cacheShape = {TEST_BATCH, TEST_CACHE_SEQ_LEN, TEST_KV_HEAD_NUM, TEST_HEAD_DIM};
    const std::vector<int64_t> quantShape = {TEST_KV_HEAD_NUM, TEST_HEAD_DIM};
    static DequantRopeQuantKvcacheCompileInfo compileInfo = {};

    return gert::TilingContextPara(
        "DequantRopeQuantKvcache",
        {
            {MakeStorageShape(xShape), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape(sinCosShape), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(sinCosShape), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(cacheShape), ge::DT_INT8, ge::FORMAT_ND},
            {MakeStorageShape(cacheShape), ge::DT_INT8, ge::FORMAT_ND},
            {MakeStorageShape({TEST_BATCH}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape(quantShape), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(quantShape), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(quantShape), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(quantShape), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(weightScaleShape), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(activationScaleShape), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(biasShape), ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape(qShape), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(kvShape), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(kvShape), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(cacheShape), ge::DT_INT8, ge::FORMAT_ND},
            {MakeStorageShape(cacheShape), ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"size_splits",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(
                 {TEST_Q_HEAD_NUM * TEST_HEAD_DIM, TEST_KV_HEAD_NUM * TEST_HEAD_DIM,
                  TEST_KV_HEAD_NUM * TEST_HEAD_DIM})},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
}
} // namespace


TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_01) {
  DequantRopeQuantKvcacheCompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 0;
  string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 0 0 0 0 1 ";
  std::vector<size_t> expectWorkspaces = {16777216};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_02) {
  DequantRopeQuantKvcacheCompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 0;
  string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 1 0 0 0 1 ";
  std::vector<size_t> expectWorkspaces = {16777216};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_03) {
  DequantRopeQuantKvcacheCompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 0;
  string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 0 1 0 0 0 1 ";
  std::vector<size_t> expectWorkspaces = {16777216};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_04) {
  DequantRopeQuantKvcacheCompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 0;
  string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 0 0 0 0 1 ";
  std::vector<size_t> expectWorkspaces = {16777216};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_05) {
  DequantRopeQuantKvcacheCompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 0;
  string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 0 0 0 0 0 1 ";
  std::vector<size_t> expectWorkspaces = {16777216};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DequantRopeQuantKvcacheTiling, OptionalInputShapes3DValid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(DequantRopeQuantKvcacheTiling, OptionalInputShapes2DValid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(false, {TEST_HIDDEN_SIZE}, {TEST_BATCH}, {TEST_HIDDEN_SIZE});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(DequantRopeQuantKvcacheTiling, WeightScaleShapeInvalid)
{
    auto tilingContextPara = BuildOptionalInputShapeContext(
        true, {1, TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, WeightScaleMissing)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, ActivationScaleShapeInvalid)
{
    auto tilingContextPara = BuildOptionalInputShapeContext(
        true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN - 1}, {TEST_HIDDEN_SIZE});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, BiasShapeInvalid)
{
    auto tilingContextPara = BuildOptionalInputShapeContext(
        true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE - 1});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantScaleKShapeInvalid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_SCALE_K_INPUT_INDEX].shape_ =
        MakeStorageShape({1, TEST_KV_HEAD_NUM, TEST_HEAD_DIM});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantScaleVShapeInvalid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_SCALE_V_INPUT_INDEX].shape_ =
        MakeStorageShape({1, TEST_KV_HEAD_NUM, TEST_HEAD_DIM});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantScaleAndOffsetFlattenedShapesValid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    for (size_t index = TEST_SCALE_K_INPUT_INDEX; index <= TEST_OFFSET_V_INPUT_INDEX; ++index) {
        tilingContextPara.inputTensorDesc_[index].shape_ = MakeStorageShape({TEST_QUANT_SIZE});
    }
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantScaleMatrixShapeInvalid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_SCALE_K_INPUT_INDEX].shape_ = MakeStorageShape({1, TEST_QUANT_SIZE});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantOffsetKShapeInvalid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_OFFSET_K_INPUT_INDEX].shape_ =
        MakeStorageShape({1, TEST_KV_HEAD_NUM, TEST_HEAD_DIM});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantOffsetVShapeInvalid)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_OFFSET_V_INPUT_INDEX].shape_ =
        MakeStorageShape({1, TEST_KV_HEAD_NUM, TEST_HEAD_DIM});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantOffsetKMissing)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_OFFSET_K_INPUT_INDEX].shape_ = MakeStorageShape({});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DequantRopeQuantKvcacheTiling, QuantOffsetVMissing)
{
    auto tilingContextPara =
        BuildOptionalInputShapeContext(true, {TEST_HIDDEN_SIZE}, {TEST_BATCH * TEST_SEQ_LEN}, {TEST_HIDDEN_SIZE});
    tilingContextPara.inputTensorDesc_[TEST_OFFSET_V_INPUT_INDEX].shape_ = MakeStorageShape({});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
