/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_swiglu_quant_v2_tiling.cpp
 * \brief CSV-driven unit tests for grouped_matmul_swiglu_quant_v2 tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_base_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_fusion_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;
using namespace optiling;

__attribute__((weak)) int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 0;
}

namespace {

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    auto len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    string exePath(path, len);
    auto pos = exePath.find_last_of("\\/");
    return pos == string::npos ? string(".\\") : exePath.substr(0, pos + 1);
#else
    char path[4096] = {0};
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return "./";
    }
    path[len] = '\0';
    string exePath(path);
    auto pos = exePath.find_last_of('/');
    return pos == string::npos ? string("./") : exePath.substr(0, pos + 1);
#endif
}

void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    const auto delimiterLen = delimiter.size();
    string::size_type currPos = 0;
    string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

string Trim(string value)
{
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

bool ParseBool(const string &value)
{
    string lower = value;
    transform(lower.begin(), lower.end(), lower.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes";
}

ge::DataType ParseDtype(const string &dtype)
{
    static const std::map<string, ge::DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"INT64", ge::DT_INT64},
        {"FLOAT8_E4M3FN", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT4_E2M1", ge::DT_FLOAT4_E2M1},
        {"HIFLOAT8", ge::DT_HIFLOAT8},
        {"UNDEFINED", ge::DT_UNDEFINED},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

ge::Format ParseFormat(const string &format)
{
    static const std::map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> dimTokens;
    SplitStr2Vec(trimmed, ":", dimTokens);
    vector<int64_t> dims;
    for (const auto &token : dimTokens) {
        dims.emplace_back(stoll(Trim(token)));
    }
    return dims;
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

gert::StorageShape MakeStorageShape(const vector<int64_t> &originDims, const vector<int64_t> &storageDims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildShape(originDims);
    shape.MutableStorageShape() = BuildShape(storageDims);
    return shape;
}

struct TensorDescParam {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
};

constexpr size_t CSV_COL_COUNT = 58U;

bool RunTemplateResetCase(int32_t tilingPriority)
{
    if (tilingPriority != 2) {
        return false;
    }

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    if (tilingData == nullptr || workspaceHolder == nullptr) {
        return false;
    }
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    auto contextHolder = gert::TilingContextFaker()
                             .SetOpType("GroupedMatmulSwigluQuantV2")
                             .NodeIoNum(0, 0)
                             .TilingData(tilingData.get())
                             .Workspace(workspace)
                             .Build();
    auto context = contextHolder.GetContext();
    if (context == nullptr) {
        return false;
    }

    GroupedMatmulSwigluQuantV2Tiling950 tiling(context);
    tiling.Reset(context);
    return true;
}

struct GroupedMatmulSwigluQuantV2TilingCase {
    void Run() const
    {
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulSwigluQuantV2", BuildTensorDescs(inputs), BuildTensorDescs(outputs),
            {
                {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dequantMode)},
                {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(dequantDtype)},
                {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantDtype)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            },
            &compileInfo);

        if (caseType == "reset_only") {
            EXPECT_TRUE(RunTemplateResetCase(tilingPriority));
            return;
        }
        if (expectSuccess) {
            TilingInfo tilingInfo;
            ExecuteTiling(tilingContextPara, tilingInfo);
            EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
            return;
        }

        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    int64_t expectTilingKey = 0;
    mutable optiling::GMMSwigluV2CompileInfo compileInfo;
    vector<TensorDescParam> inputs;
    vector<TensorDescParam> outputs;
    int64_t dequantMode = 0;
    float dequantDtype = 0.0F;
    int64_t quantMode = 0;
    int64_t quantDtype = 0;
    bool transposeWeight = false;
    int64_t groupListType = 0;
    string caseType = "tiling";
    int32_t tilingPriority = 2;

private:
    static vector<gert::TilingContextPara::TensorDescription> BuildTensorDescs(const vector<TensorDescParam> &tensors)
    {
        vector<gert::TilingContextPara::TensorDescription> descs;
        descs.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            descs.emplace_back(MakeStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype, tensor.format);
        }
        return descs;
    }
};

vector<GroupedMatmulSwigluQuantV2TilingCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulSwigluQuantV2TilingCase> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath +
                   "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host/"
                   "test_grouped_matmul_swiglu_quant_v2_tiling.csv");
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
        return cases;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < CSV_COL_COUNT) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulSwigluQuantV2TilingCase tc;
        tc.socVersion = Trim(items[idx++]);
        if (tc.socVersion != socVersion) {
            continue;
        }
        tc.caseName = Trim(items[idx++]);
        tc.enable = ParseBool(Trim(items[idx++]));
        if (!tc.enable) {
            continue;
        }
        tc.prefix = Trim(items[idx++]);
        tc.expectSuccess = ParseBool(Trim(items[idx++]));
        tc.expectTilingKey = stoll(Trim(items[idx++]));

        tc.compileInfo = {
            static_cast<uint64_t>(stoull(Trim(items[idx++]))),
            static_cast<uint32_t>(stoul(Trim(items[idx++]))),
            static_cast<uint32_t>(stoul(Trim(items[idx++]))),
            static_cast<uint32_t>(stoul(Trim(items[idx++]))),
            static_cast<uint32_t>(stoul(Trim(items[idx++]))),
            ParseBool(Trim(items[idx++])),
        };

        for (int tensorIdx = 0; tensorIdx < 8; ++tensorIdx) {
            tc.inputs.push_back({ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                                 ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))});
        }
        for (int tensorIdx = 0; tensorIdx < 2; ++tensorIdx) {
            tc.outputs.push_back({ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                                  ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))});
        }

        tc.dequantMode = stoll(Trim(items[idx++]));
        tc.dequantDtype = stof(Trim(items[idx++]));
        tc.quantMode = stoll(Trim(items[idx++]));
        tc.quantDtype = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        if (items.size() > idx) {
            string parsedCaseType = Trim(items[idx++]);
            if (!parsedCaseType.empty()) {
                tc.caseType = parsedCaseType;
            }
        }
        if (items.size() > idx) {
            string parsedTilingPriority = Trim(items[idx++]);
            if (!parsedTilingPriority.empty()) {
                tc.tilingPriority = static_cast<int32_t>(stol(parsedTilingPriority));
            }
        }
        cases.push_back(tc);
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulSwigluQuantV2TilingCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

} // namespace

namespace GroupedMatmulSwigluQuantV2UT {

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestGroupedMatmulSwigluQuantV2Tiling950);


const vector<GroupedMatmulSwigluQuantV2TilingCase> &GetAscend950Cases()
{
    static const vector<GroupedMatmulSwigluQuantV2TilingCase> cases = LoadCases("Ascend950");
    return cases;
}

const vector<GroupedMatmulSwigluQuantV2TilingCase> &GetAscend910BCases()
{
    static const vector<GroupedMatmulSwigluQuantV2TilingCase> cases = LoadCases("Ascend910B");
    return cases;
}

class TestGroupedMatmulSwigluQuantV2Tiling950 : public testing::TestWithParam<GroupedMatmulSwigluQuantV2TilingCase> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling950 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling950 TearDown" << std::endl;
    }
};

TEST_P(TestGroupedMatmulSwigluQuantV2Tiling950, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V2_TILING_950, TestGroupedMatmulSwigluQuantV2Tiling950,
                         testing::ValuesIn(GetAscend950Cases()), MakeParamName);

class TestGroupedMatmulSwigluQuantV2Tiling910B : public testing::TestWithParam<GroupedMatmulSwigluQuantV2TilingCase> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling910B SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantV2Tiling910B TearDown" << std::endl;
    }
};

TEST_P(TestGroupedMatmulSwigluQuantV2Tiling910B, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V2_TILING_910B, TestGroupedMatmulSwigluQuantV2Tiling910B,
                         testing::ValuesIn(GetAscend910BCases()), MakeParamName);

} // namespace GroupedMatmulSwigluQuantV2UT
