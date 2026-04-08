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
 * \file test_grouped_matmul_infershape.cpp
 * \brief CSV-driven unit tests for grouped_matmul infershape.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

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
        {"UNDEFINED", ge::DT_UNDEFINED},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

ge::Format ParseFormat(const string &format)
{
    static const std::map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"NCL", ge::FORMAT_NCL},
        {"NCHW", ge::FORMAT_NCHW},
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed == "NONE") {
        // Keep compatibility with existing CSVs: NONE means empty optional tensor ([0]).
        return {0};
    }
    if (trimmed == "ZERO") {
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

struct GroupedMatmulInfershapeCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmul",
            {
                {MakeStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {MakeStorageShape(ParseDims(weightShape)), ParseDtype(weightDtype), ParseFormat(weightFormat)},
                {MakeStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ParseFormat(biasFormat)},
                {MakeStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype), ParseFormat(scaleFormat)},
                {MakeStorageShape(ParseDims(offsetShape)), ParseDtype(offsetDtype), ParseFormat(offsetFormat)},
                {MakeStorageShape(ParseDims(antiquantScaleShape)), ParseDtype(antiquantScaleDtype),
                 ParseFormat(antiquantScaleFormat)},
                {MakeStorageShape(ParseDims(antiquantOffsetShape)), ParseDtype(antiquantOffsetDtype),
                 ParseFormat(antiquantOffsetFormat)},
                {MakeStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ParseFormat(groupListFormat)},
                {MakeStorageShape(ParseDims(perTokenScaleShape)), ParseDtype(perTokenScaleDtype),
                 ParseFormat(perTokenScaleFormat)},
            },
            {
                {MakeStorageShape(ParseDims("")), ParseDtype(outDtype), ParseFormat(outFormat)},
            },
            {
                {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputDtypeAttr)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
                {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            });

        vector<vector<int64_t>> expectShape;
        expectShape.push_back(ParseDims(expectOutputShape));
        ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    string caseType;
    bool expectSuccess = false;
    string expectOutputShape;

    string xShape;
    string weightShape;
    string biasShape;
    string scaleShape;
    string offsetShape;
    string antiquantScaleShape;
    string antiquantOffsetShape;
    string groupListShape;
    string perTokenScaleShape;

    string xDtype;
    string weightDtype;
    string biasDtype;
    string scaleDtype;
    string offsetDtype;
    string antiquantScaleDtype;
    string antiquantOffsetDtype;
    string groupListDtype;
    string perTokenScaleDtype;
    string outDtype;

    string xFormat;
    string weightFormat;
    string biasFormat;
    string scaleFormat;
    string offsetFormat;
    string antiquantScaleFormat;
    string antiquantOffsetFormat;
    string groupListFormat;
    string perTokenScaleFormat;
    string outFormat;

    int64_t splitItem = 3;
    int64_t outputDtypeAttr = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
};

vector<GroupedMatmulInfershapeCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulInfershapeCase> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath + "gmm/grouped_matmul/tests/ut/op_host/test_grouped_matmul_infershape.csv");
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 43U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulInfershapeCase tc;
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
        tc.expectOutputShape = Trim(items[idx++]);

        tc.xShape = Trim(items[idx++]);
        tc.weightShape = Trim(items[idx++]);
        tc.biasShape = Trim(items[idx++]);
        tc.scaleShape = Trim(items[idx++]);
        tc.offsetShape = Trim(items[idx++]);
        tc.antiquantScaleShape = Trim(items[idx++]);
        tc.antiquantOffsetShape = Trim(items[idx++]);
        tc.groupListShape = Trim(items[idx++]);
        tc.perTokenScaleShape = Trim(items[idx++]);

        tc.xDtype = Trim(items[idx++]);
        tc.weightDtype = Trim(items[idx++]);
        tc.biasDtype = Trim(items[idx++]);
        tc.scaleDtype = Trim(items[idx++]);
        tc.offsetDtype = Trim(items[idx++]);
        tc.antiquantScaleDtype = Trim(items[idx++]);
        tc.antiquantOffsetDtype = Trim(items[idx++]);
        tc.groupListDtype = Trim(items[idx++]);
        tc.perTokenScaleDtype = Trim(items[idx++]);
        tc.outDtype = Trim(items[idx++]);

        tc.xFormat = Trim(items[idx++]);
        tc.weightFormat = Trim(items[idx++]);
        tc.biasFormat = Trim(items[idx++]);
        tc.scaleFormat = Trim(items[idx++]);
        tc.offsetFormat = Trim(items[idx++]);
        tc.antiquantScaleFormat = Trim(items[idx++]);
        tc.antiquantOffsetFormat = Trim(items[idx++]);
        tc.groupListFormat = Trim(items[idx++]);
        tc.perTokenScaleFormat = Trim(items[idx++]);
        tc.outFormat = Trim(items[idx++]);

        tc.splitItem = stoll(Trim(items[idx++]));
        tc.outputDtypeAttr = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.transposeX = ParseBool(Trim(items[idx++]));
        tc.groupType = stoll(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        tc.actType = stoll(Trim(items[idx++]));
        cases.push_back(tc);
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulInfershapeCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

} // namespace

namespace GroupedMatmulInfershapeUT {

const vector<GroupedMatmulInfershapeCase> &GetGroupedMatmulInfershapeCsvCases()
{
    static const vector<GroupedMatmulInfershapeCase> cases = [] {
        vector<GroupedMatmulInfershapeCase> merged = LoadCases("Ascend950");
        vector<GroupedMatmulInfershapeCase> ascend910b = LoadCases("Ascend910B");
        merged.insert(merged.end(), ascend910b.begin(), ascend910b.end());
        return merged;
    }();
    return cases;
}

class TestGroupedMatmulInfershape : public testing::TestWithParam<GroupedMatmulInfershapeCase> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedMatmulInfershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_MATMUL_INFERSHAPE_CSV, TestGroupedMatmulInfershape,
                         testing::ValuesIn(GetGroupedMatmulInfershapeCsvCases()), MakeParamName);

} // namespace GroupedMatmulInfershapeUT
