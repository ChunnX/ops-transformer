/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MsaIndexScoreTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreTilingArch35 TearDown" << std::endl;
    }
};

namespace {
constexpr uint64_t MSA_KEY_BF16 = 0;
constexpr uint64_t MSA_KEY_FP16 = 1;
constexpr uint64_t MSA_KEY_FP16_INT8 = 3;
constexpr uint64_t MSA_KEY_HIFLOAT8 = 4;
constexpr uint64_t MSA_KEY_FP8_E5M2 = 5;
constexpr uint64_t MSA_KEY_FP8_E4M3FN = 6;

gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    switch (dims.size()) {
        case 1:
            return gert::StorageShape({dims[0]}, {dims[0]});
        case 2:
            return gert::StorageShape({dims[0], dims[1]}, {dims[0], dims[1]});
        case 3:
            return gert::StorageShape({dims[0], dims[1], dims[2]}, {dims[0], dims[1], dims[2]});
        case 4:
            return gert::StorageShape({dims[0], dims[1], dims[2], dims[3]}, {dims[0], dims[1], dims[2], dims[3]});
        default:
            return gert::StorageShape({}, {});
    }
}

gert::TilingContextPara::TensorDescription Present(const std::vector<int64_t> &shape, ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(MakeStorageShape(shape), dtype, ge::FORMAT_ND);
}

gert::TilingContextPara::TensorDescription Absent(ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(gert::StorageShape({}, {}), dtype, ge::FORMAT_ND);
}

struct MsaTilingSpec {
    std::vector<int64_t> queryShape{32, 8, 128};
    std::vector<int64_t> keyShape{2, 128, 1, 128};
    bool hasBlockTable{true};
    std::vector<int64_t> blockTableShape{1, 2};
    bool hasScale{false};
    std::vector<int64_t> scaleShape{};
    bool hasMask{false};
    std::vector<int64_t> qlenShape{2};
    std::vector<int64_t> klenShape{1};
    std::vector<int64_t> startLocShape{1};
    std::vector<int64_t> scoreShape{8, 32, 16};
    ge::DataType queryDtype{ge::DT_FLOAT16};
    ge::DataType keyDtype{ge::DT_FLOAT16};
    std::string layout{"BBND"};
    int64_t sparseMode{0};
    int64_t initBlocks{0};
    int64_t localBlocks{1};
};

gert::TilingContextPara MakeTilingPara(const MsaTilingSpec &spec, void *compileInfo)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back(Present(spec.queryShape, spec.queryDtype));
    inputs.push_back(Present(spec.keyShape, spec.keyDtype));
    inputs.push_back(spec.hasBlockTable ? Present(spec.blockTableShape, ge::DT_INT32) : Absent(ge::DT_INT32));
    inputs.push_back(spec.hasScale ? Present(spec.scaleShape, ge::DT_FLOAT) : Absent(ge::DT_FLOAT));
    inputs.push_back(spec.hasMask ? Present({2048, 2048}, ge::DT_INT8) : Absent(ge::DT_INT8));
    inputs.push_back(spec.qlenShape.empty() ? Absent(ge::DT_INT32) : Present(spec.qlenShape, ge::DT_INT32));
    inputs.push_back(spec.klenShape.empty() ? Absent(ge::DT_INT32) : Present(spec.klenShape, ge::DT_INT32));
    inputs.push_back(spec.startLocShape.empty() ? Absent(ge::DT_INT32) : Present(spec.startLocShape, ge::DT_INT32));

    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.push_back(Present(spec.scoreShape, ge::DT_FLOAT));

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>(spec.layout)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.sparseMode)},
        {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.initBlocks)},
        {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(spec.localBlocks)}};

    return gert::TilingContextPara("MsaIndexScore", inputs, outputs, attrs, compileInfo, "Ascend950", 56, 262144,
                                   16384);
}
} // namespace

TEST_F(MsaIndexScoreTilingArch35, bbnd_fp16_nonquant_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_F(MsaIndexScoreTilingArch35, bbnd_bf16_causal_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_BF16;
    spec.keyDtype = ge::DT_BF16;
    spec.sparseMode = 3;
    spec.hasMask = true;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_BF16);
}

TEST_F(MsaIndexScoreTilingArch35, bbnd_fp16_int8_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP16_INT8);
}

TEST_F(MsaIndexScoreTilingArch35, bnbd_fp16_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.layout = "BNBD";
    spec.keyShape = {2, 1, 128, 128};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_F(MsaIndexScoreTilingArch35, tnd_fp16_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.layout = "TND";
    spec.keyShape = {256, 1, 128};
    spec.hasBlockTable = false;
    spec.klenShape = {2};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_F(MsaIndexScoreTilingArch35, bbnd_fp8_e4m3fn_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_FLOAT8_E4M3FN;
    spec.keyDtype = ge::DT_FLOAT8_E4M3FN;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP8_E4M3FN);
}

TEST_F(MsaIndexScoreTilingArch35, bbnd_fp8_e5m2_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_FLOAT8_E5M2;
    spec.keyDtype = ge::DT_FLOAT8_E5M2;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP8_E5M2);
}

TEST_F(MsaIndexScoreTilingArch35, bbnd_hifloat8_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.queryDtype = ge::DT_HIFLOAT8;
    spec.keyDtype = ge::DT_HIFLOAT8;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_HIFLOAT8);
}

TEST_F(MsaIndexScoreTilingArch35, empty_query_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.queryShape = {0, 8, 128};
    spec.scoreShape = {8, 0, 16};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_SUCCESS, MSA_KEY_FP16);
}

TEST_F(MsaIndexScoreTilingArch35, layout_invalid_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.layout = "BSND";
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, causal_without_mask_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.sparseMode = 3;
    spec.hasMask = false;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, default_mask_with_atten_mask_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.sparseMode = 0;
    spec.hasMask = true;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, head_dim_not_aligned_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.queryShape = {32, 8, 8};
    spec.keyShape = {2, 128, 1, 8};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, block_size_invalid_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.keyShape = {2, 64, 1, 128};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, non_quant_with_scale_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.hasScale = true;
    spec.scaleShape = {2, 1, 128};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, tnd_with_block_table_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.layout = "TND";
    spec.keyShape = {256, 1, 128};
    spec.hasBlockTable = true;
    spec.klenShape = {2};
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, init_blocks_too_large_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.initBlocks = 8; // maxBlocksPerBatch = 2
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, int8_without_scale_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.keyDtype = ge::DT_INT8;
    spec.hasScale = false;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}

TEST_F(MsaIndexScoreTilingArch35, pa_missing_block_table_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    MsaTilingSpec spec;
    spec.hasBlockTable = false;
    ExecuteTestCase(MakeTilingPara(spec, &compileInfo), ge::GRAPH_FAILED);
}
