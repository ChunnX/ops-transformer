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
 * \file test_quant_grouped_matmul_inplace_add_arch35_tiling.cpp
 * \brief CSV-driven unit tests for QuantGroupedMatmulInplaceAdd arch35 tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/quant_grouped_matmul_inplace_add_tiling.h"
#include "../../../op_kernel/arch35/quant_grouped_matmul_inplace_add_tiling_data.h"
#include "csv_test_utils.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {

using csv_test_utils::GetExeDirPath;
using csv_test_utils::ParseBool;
using csv_test_utils::ParseDims;
using csv_test_utils::SplitStr2Vec;
using csv_test_utils::ToLower;
using csv_test_utils::Trim;

DataType ParseDtype(const string &dtype)
{
    static const map<string, DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT64", ge::DT_INT64},
        {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
        {"HIFLOAT8", ge::DT_HIFLOAT8},
    };
    auto it = dtypeMap.find(Trim(dtype));
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

gert::Shape BuildShape(const vector<int64_t> &dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

gert::StorageShape MakeShape(const vector<int64_t> &dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildShape(dims);
    shape.MutableStorageShape() = BuildShape(dims);
    return shape;
}

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

struct QGmmInplaceAddArch35Param {
    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    int64_t groupNum = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    DataType xDtype = ge::DT_UNDEFINED;
    DataType wDtype = ge::DT_UNDEFINED;
    DataType scale2Dtype = ge::DT_UNDEFINED;
    DataType scale1Dtype = ge::DT_UNDEFINED;
    DataType yDtype = ge::DT_UNDEFINED;
    int64_t groupListType = 0;
    bool expectSuccess = false;
    bool checkTilingKey = false;
    uint64_t expectTilingKey = 0;
    vector<int64_t> scale2Shape;
    vector<int64_t> scale1Shape;

    void Test() const
    {
        optiling::GMMCompileInfo compileInfo = {
            32, 64, 262144, 524288, 134217728, 262144, 65536, 65536, platform_ascendc::SocVersion::ASCEND950,
            NpuArch::DAV_3510,
        };

        gert::StorageShape xShape = MakeShape({k, m});
        gert::StorageShape weightShape = MakeShape({k, n});
        gert::StorageShape groupListShape = MakeShape({groupNum});
        gert::StorageShape yShape = MakeShape({groupNum, m, n});

        gert::TilingContextPara tilingContextPara(
            "QuantGroupedMatmulInplaceAdd",
            {
                {xShape, xDtype, ge::FORMAT_ND},                         // x1
                {weightShape, wDtype, ge::FORMAT_ND},                    // x2
                {MakeShape(scale2Shape), scale2Dtype, ge::FORMAT_ND},    // scale2
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},           // group_list
                {yShape, yDtype, ge::FORMAT_ND},                          // y (inplace input)
                {MakeShape(scale1Shape), scale1Dtype, ge::FORMAT_ND},     // scale1
            },
            {{yShape, yDtype, ge::FORMAT_ND}},
            {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
             {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
            &compileInfo,
            "3510", compileInfo.aicNum, compileInfo.ubSize);

        TilingInfo tilingInfo;
        bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_EQ(tilingResult, expectSuccess) << "case=" << caseName;
        if (!expectSuccess) {
            return;
        }

        ASSERT_GT(tilingInfo.blockNum, 0U) << "case=" << caseName;
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams))
            << "case=" << caseName;
        if (checkTilingKey) {
            ASSERT_EQ(tilingInfo.tilingKey, expectTilingKey) << "case=" << caseName;
        }
    }
};

vector<QGmmInplaceAddArch35Param> GetParams(const string &socVersion)
{
    vector<QGmmInplaceAddArch35Param> params;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(
        rootPath + "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host/test_quant_grouped_matmul_inplace_add_arch35_tiling.csv");
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
        return params;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < 18U) {
            continue;
        }

        size_t idx = 0;
        QGmmInplaceAddArch35Param p;
        p.socVersion = Trim(items[idx++]);
        if (p.socVersion != socVersion) {
            continue;
        }
        p.caseName = Trim(items[idx++]);
        p.enable = ParseBool(items[idx++]);
        if (!p.enable) {
            continue;
        }
        p.prefix = Trim(items[idx++]);
        p.groupNum = stoll(Trim(items[idx++]));
        p.m = stoll(Trim(items[idx++]));
        p.k = stoll(Trim(items[idx++]));
        p.n = stoll(Trim(items[idx++]));
        p.xDtype = ParseDtype(items[idx++]);
        p.wDtype = ParseDtype(items[idx++]);
        p.scale2Dtype = ParseDtype(items[idx++]);
        p.scale1Dtype = ParseDtype(items[idx++]);
        p.yDtype = ParseDtype(items[idx++]);
        p.groupListType = stoll(Trim(items[idx++]));
        p.expectSuccess = ParseBool(items[idx++]);
        string keyField = Trim(items[idx++]);
        p.checkTilingKey = !keyField.empty() && keyField != "NONE";
        p.expectTilingKey = p.checkTilingKey ? static_cast<uint64_t>(stoull(keyField)) : 0;
        p.scale2Shape = ParseDims(items[idx++]);
        p.scale1Shape = ParseDims(items[idx++]);
        params.push_back(p);
    }
    return params;
}

string MakeParamName(const testing::TestParamInfo<QGmmInplaceAddArch35Param> &info)
{
    string name = info.param.prefix;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}
} // namespace

namespace QuantGroupedMatmulInplaceAddArch35TilingUT {
const vector<QGmmInplaceAddArch35Param> &GetAscend950Params()
{
    static const vector<QGmmInplaceAddArch35Param> params = GetParams("Ascend950");
    return params;
}

class TestQuantGroupedMatmulInplaceAddArch35Tiling : public testing::TestWithParam<QGmmInplaceAddArch35Param> {
};

TEST_P(TestQuantGroupedMatmulInplaceAddArch35Tiling, csvDrivenCase)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(QUANT_GROUPED_MATMUL_INPLACE_ADD_950, TestQuantGroupedMatmulInplaceAddArch35Tiling,
                         testing::ValuesIn(GetAscend950Params()), MakeParamName);

TEST(TestQuantGroupedMatmulInplaceAddArch35TilingExtra, post_tiling_memcpy_fail_when_tiling_buffer_too_small)
{
    optiling::GMMCompileInfo compileInfo = {
        32, 64, 262144, 524288, 134217728, 262144, 65536, 65536, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510,
    };

    gert::StorageShape xShape = MakeShape({512, 96});
    gert::StorageShape weightShape = MakeShape({512, 128});
    gert::StorageShape groupListShape = MakeShape({4});
    gert::StorageShape yShape = MakeShape({4, 96, 128});

    gert::TilingContextPara para(
        "QuantGroupedMatmulInplaceAdd",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {MakeShape({12, 128, 2}), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeShape({12, 96, 2}), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize,
        8); // intentionally too small raw tiling buffer to force memcpy_s failure in PostTiling

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST(TestQuantGroupedMatmulInplaceAddArch35TilingExtra, print_quant_params_debug_branch_smoke)
{
#if defined(_WIN32)
    _putenv_s("ASCEND_GLOBAL_LOG_LEVEL", "0");
#else
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
#endif

    optiling::GMMCompileInfo compileInfo = {
        32, 64, 262144, 524288, 134217728, 262144, 65536, 65536, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510,
    };
    gert::StorageShape xShape = MakeShape({512, 96});
    gert::StorageShape weightShape = MakeShape({512, 128});
    gert::StorageShape groupListShape = MakeShape({4});
    gert::StorageShape yShape = MakeShape({4, 96, 128});
    gert::TilingContextPara para(
        "QuantGroupedMatmulInplaceAdd",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {MakeShape({12, 128, 2}), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeShape({12, 96, 2}), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(para, info));
}
} // namespace QuantGroupedMatmulInplaceAddArch35TilingUT
