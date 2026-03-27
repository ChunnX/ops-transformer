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
 * \file test_grouped_matmul_swiglu_quant_v2_infershape.cpp
 * \brief CSV-driven unit tests for grouped_matmul_swiglu_quant_v2 infershape.
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

#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#define private public
#include "platform/platform_info.h"

using namespace std;

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
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes";
}

ge::DataType ParseDtype(const string &dtype)
{
    static const std::map<string, ge::DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT32", ge::DT_INT32},
        {"INT64", ge::DT_INT64},
        {"UINT64", ge::DT_UINT64},
        {"FLOAT8_E4M3FN", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT4_E2M1", ge::DT_FLOAT4_E2M1},
        {"INT4", ge::DT_INT4},
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
        {"NDC1HWC0", ge::FORMAT_NDC1HWC0},
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed == "NONE" || trimmed == "ZERO" || trimmed.empty()) {
        return {0};
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

gert::StorageShape MakeStorageShape(const vector<int64_t> &dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildShape(dims);
    shape.MutableStorageShape() = BuildShape(dims);
    return shape;
}

void SetupPlatformForCase(const string &socVersion)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = socVersion;
    platformInfo.str_info.short_soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

struct GroupedMatmulSwigluQuantV2InfershapeCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmulSwigluQuantV2",
            {
                {MakeStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {MakeStorageShape(ParseDims(xScaleShape)), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat)},
                {MakeStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ParseFormat(groupListFormat)},
                {MakeStorageShape(ParseDims(weightShape)), ParseDtype(weightDtype), ParseFormat(weightFormat)},
                {MakeStorageShape(ParseDims(weightScaleShape)), ParseDtype(weightScaleDtype), ParseFormat(weightScaleFormat)},
                {MakeStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ParseFormat(biasFormat)},
                {MakeStorageShape(ParseDims(offsetShape)), ParseDtype(offsetDtype), ParseFormat(offsetFormat)},
                {MakeStorageShape(ParseDims(antiquantOffsetShape)), ParseDtype(antiquantOffsetDtype),
                 ParseFormat(antiquantOffsetFormat)},
            },
            {
                {MakeStorageShape(ParseDims("NONE")), ParseDtype(yDtype), ParseFormat(yFormat)},
                {MakeStorageShape(ParseDims("NONE")), ParseDtype(yScaleDtype), ParseFormat(yScaleFormat)},
            },
            {
                {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dequantMode)},
                {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(dequantDtype)},
                {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantDtype)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            });

        vector<vector<int64_t>> expectShape;
        expectShape.push_back(ParseDims(expectYShape));
        expectShape.push_back(ParseDims(expectYScaleShape));
        ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    string caseType;
    bool expectSuccess = false;
    string expectYShape;
    string expectYScaleShape;

    string xShape;
    string xScaleShape;
    string groupListShape;
    string weightShape;
    string weightScaleShape;
    string biasShape;
    string offsetShape;
    string antiquantOffsetShape;

    string xDtype;
    string xScaleDtype;
    string groupListDtype;
    string weightDtype;
    string weightScaleDtype;
    string biasDtype;
    string offsetDtype;
    string antiquantOffsetDtype;
    string yDtype;
    string yScaleDtype;

    string xFormat;
    string xScaleFormat;
    string groupListFormat;
    string weightFormat;
    string weightScaleFormat;
    string biasFormat;
    string offsetFormat;
    string antiquantOffsetFormat;
    string yFormat;
    string yScaleFormat;

    int64_t dequantMode = 0;
    float dequantDtype = 0.0F;
    int64_t quantMode = 0;
    int64_t quantDtype = 0;
    bool transposeWeight = false;
    int64_t groupListType = 0;
};

vector<GroupedMatmulSwigluQuantV2InfershapeCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulSwigluQuantV2InfershapeCase> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath +
                   "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host/"
                   "test_grouped_matmul_swiglu_quant_v2_infershape.csv");
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 42U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulSwigluQuantV2InfershapeCase tc;
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
        tc.caseType = Trim(items[idx++]);
        tc.expectSuccess = ParseBool(Trim(items[idx++]));
        tc.expectYShape = Trim(items[idx++]);
        tc.expectYScaleShape = Trim(items[idx++]);

        tc.xShape = Trim(items[idx++]);
        tc.xScaleShape = Trim(items[idx++]);
        tc.groupListShape = Trim(items[idx++]);
        tc.weightShape = Trim(items[idx++]);
        tc.weightScaleShape = Trim(items[idx++]);
        tc.biasShape = Trim(items[idx++]);
        tc.offsetShape = Trim(items[idx++]);
        tc.antiquantOffsetShape = Trim(items[idx++]);

        tc.xDtype = Trim(items[idx++]);
        tc.xScaleDtype = Trim(items[idx++]);
        tc.groupListDtype = Trim(items[idx++]);
        tc.weightDtype = Trim(items[idx++]);
        tc.weightScaleDtype = Trim(items[idx++]);
        tc.biasDtype = Trim(items[idx++]);
        tc.offsetDtype = Trim(items[idx++]);
        tc.antiquantOffsetDtype = Trim(items[idx++]);
        tc.yDtype = Trim(items[idx++]);
        tc.yScaleDtype = Trim(items[idx++]);

        tc.xFormat = Trim(items[idx++]);
        tc.xScaleFormat = Trim(items[idx++]);
        tc.groupListFormat = Trim(items[idx++]);
        tc.weightFormat = Trim(items[idx++]);
        tc.weightScaleFormat = Trim(items[idx++]);
        tc.biasFormat = Trim(items[idx++]);
        tc.offsetFormat = Trim(items[idx++]);
        tc.antiquantOffsetFormat = Trim(items[idx++]);
        tc.yFormat = Trim(items[idx++]);
        tc.yScaleFormat = Trim(items[idx++]);

        tc.dequantMode = stoll(Trim(items[idx++]));
        tc.dequantDtype = stof(Trim(items[idx++]));
        tc.quantMode = stoll(Trim(items[idx++]));
        tc.quantDtype = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        cases.push_back(tc);
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulSwigluQuantV2InfershapeCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}
} // namespace

namespace GroupedMatmulSwigluQuantV2InfershapeUT {
const vector<GroupedMatmulSwigluQuantV2InfershapeCase> &GetAscend950Cases()
{
    static const vector<GroupedMatmulSwigluQuantV2InfershapeCase> cases = LoadCases("Ascend950");
    return cases;
}

class TestGroupedMatmulSwigluQuantV2Infershape
    : public testing::TestWithParam<GroupedMatmulSwigluQuantV2InfershapeCase> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedMatmulSwigluQuantV2Infershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V2_INFERSHAPE_950, TestGroupedMatmulSwigluQuantV2Infershape,
                         testing::ValuesIn(GetAscend950Cases()), MakeParamName);
} // namespace GroupedMatmulSwigluQuantV2InfershapeUT