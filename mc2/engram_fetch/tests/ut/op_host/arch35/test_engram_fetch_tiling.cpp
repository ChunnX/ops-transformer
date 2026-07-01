/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fetch_tiling.cpp
 * \brief engram_fetch 算子 host 侧 tiling UT
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

namespace EngramFetchUT {

static const std::string OP_NAME = "EngramFetch";

struct EngramFetchTestParam {
    std::string caseName;
    std::initializer_list<int64_t> commContextShape;
    ge::DataType commContextDtype;
    ge::Format commContextFormat;
    std::initializer_list<int64_t> indicesShape;
    ge::DataType indicesDtype;
    ge::Format indicesFormat;
    std::initializer_list<int64_t> fetchedShape;
    ge::DataType fetchedDtype;
    ge::Format fetchedFormat;
    int64_t hiddenSize;
    int64_t numEntriesPerRank;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
};

inline std::ostream& operator<<(std::ostream& os, const EngramFetchTestParam& param)
{
    return os << param.caseName;
}

static EngramFetchTestParam g_testCases[] = {
    {"success_bf16_basic",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_fp16_basic",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_FLOAT16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_fp32_basic",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_FLOAT, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_large_tokens",
     {100 * 256}, ge::DT_INT32, ge::FORMAT_ND,
     {128}, ge::DT_INT32, ge::FORMAT_ND,
     {128, 256}, ge::DT_BF16, ge::FORMAT_ND,
     256, 100, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_large_hidden",
     {10 * 4096}, ge::DT_INT32, ge::FORMAT_ND,
     {32}, ge::DT_INT32, ge::FORMAT_ND,
     {32, 4096}, ge::DT_BF16, ge::FORMAT_ND,
     4096, 10, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_single_token",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {1}, ge::DT_INT32, ge::FORMAT_ND,
     {1, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_empty_indices",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {0}, ge::DT_INT32, ge::FORMAT_ND,
     {0, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"success_zero_entries_per_rank",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 0, "3510",
     ge::GRAPH_SUCCESS, 0UL, "", {16777216}},

    {"fail_commContext_dtype_float16",
     {4 * 512}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_commContext_dtype_bf16",
     {4 * 512}, ge::DT_BF16, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_indices_dtype_float16",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_indices_dtype_float32",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_FLOAT, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_dtype_int32",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_INT32, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_dtype_int8",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_INT8, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_indices_2d",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {2, 4}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_indices_3d",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_1d",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {4096}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_3d",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {2, 4, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_dim0_mismatch",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {16, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_dim1_mismatch",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 256}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_hidden_size_zero",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     0, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_hidden_size_negative",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     -1, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_num_entries_negative",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, -1, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_hidden_size_not_aligned",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 100}, ge::DT_BF16, ge::FORMAT_ND,
     100, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_commContext_format_nz",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_indices_format_nz",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_fetched_format_nz",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_num_entries_exceed_int32max",
     {4 * 512}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 2147483648L, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},

    {"fail_commContext_empty",
     {0}, ge::DT_INT32, ge::FORMAT_ND,
     {8}, ge::DT_INT32, ge::FORMAT_ND,
     {8, 512}, ge::DT_BF16, ge::FORMAT_ND,
     512, 4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}},
};

class EngramFetchArch35TilingTest : public testing::TestWithParam<EngramFetchTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramFetchArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramFetchArch35TilingTest TearDown." << std::endl;
    }
};

static struct EngramFetchCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const EngramFetchTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape commContextShape = {param.commContextShape, param.commContextShape};
    gert::StorageShape indicesShape = {param.indicesShape, param.indicesShape};
    gert::StorageShape fetchedShape = {param.fetchedShape, param.fetchedShape};

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{commContextShape, param.commContextDtype, param.commContextFormat},
         {indicesShape, param.indicesDtype, param.indicesFormat}});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{fetchedShape, param.fetchedDtype, param.fetchedFormat}});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hiddenSize)},
         {"num_entries_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numEntriesPerRank)}});

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

TEST_P(EngramFetchArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

INSTANTIATE_TEST_CASE_P(EngramFetchTilingUT, EngramFetchArch35TilingTest, testing::ValuesIn(g_testCases));

} // namespace EngramFetchUT
