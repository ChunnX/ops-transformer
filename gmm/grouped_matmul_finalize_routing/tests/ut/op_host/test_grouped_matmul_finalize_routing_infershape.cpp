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
 * \file test_grouped_matmul_finalize_routing_infershape.cpp
 * \brief CSV-driven unit tests for grouped_matmul_finalize_routing infershape.
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
        {"INT4", ge::DT_INT4},
        {"INT8", ge::DT_INT8},
        {"INT32", ge::DT_INT32},
        {"INT64", ge::DT_INT64},
        {"UINT64", ge::DT_UINT64},
        {"FLOAT8_E4M3FN", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
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
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed == "NONE" || trimmed == "ZERO") {
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

struct GroupedMatmulFinalizeRoutingInfershapeCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {MakeStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {MakeStorageShape(ParseDims(weightShape)), ParseDtype(weightDtype), ParseFormat(weightFormat)},
                {MakeStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype), ParseFormat(scaleFormat)},
                {MakeStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ParseFormat(biasFormat)},
                {MakeStorageShape(ParseDims(pertokenScaleShape)), ParseDtype(pertokenScaleDtype),
                 ParseFormat(pertokenScaleFormat)},
                {MakeStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ParseFormat(groupListFormat)},
                {MakeStorageShape(ParseDims(sharedInputShape)), ParseDtype(sharedInputDtype), ParseFormat(sharedInputFormat)},
                {MakeStorageShape(ParseDims(logitShape)), ParseDtype(logitDtype), ParseFormat(logitFormat)},
                {MakeStorageShape(ParseDims(rowIndexShape)), ParseDtype(rowIndexDtype), ParseFormat(rowIndexFormat)},
                {MakeStorageShape(ParseDims(offsetShape)), ParseDtype(offsetDtype), ParseFormat(offsetFormat)},
            },
            {
                {MakeStorageShape(ParseDims("")), ParseDtype(yDtype), ParseFormat(yFormat)},
            },
            {
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtype)},
                {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(sharedInputWeight)},
                {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedInputOffset)},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
                {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeW)},
                {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputBs)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            });

        vector<vector<int64_t>> expectShape;
        expectShape.push_back(ParseDims(expectYShape));
        ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    string caseType;
    bool expectSuccess = false;
    string expectYShape;

    string xShape;
    string weightShape;
    string scaleShape;
    string biasShape;
    string pertokenScaleShape;
    string groupListShape;
    string sharedInputShape;
    string logitShape;
    string rowIndexShape;
    string offsetShape;

    string xDtype;
    string weightDtype;
    string scaleDtype;
    string biasDtype;
    string pertokenScaleDtype;
    string groupListDtype;
    string sharedInputDtype;
    string logitDtype;
    string rowIndexDtype;
    string offsetDtype;
    string yDtype;

    string xFormat;
    string weightFormat;
    string scaleFormat;
    string biasFormat;
    string pertokenScaleFormat;
    string groupListFormat;
    string sharedInputFormat;
    string logitFormat;
    string rowIndexFormat;
    string offsetFormat;
    string yFormat;

    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t outputBs = 0;
    int64_t groupListType = 0;
};

vector<GroupedMatmulFinalizeRoutingInfershapeCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulFinalizeRoutingInfershapeCase> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath +
                   "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/"
                   "test_grouped_matmul_finalize_routing_infershape.csv");
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 47U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulFinalizeRoutingInfershapeCase tc;
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

        tc.xShape = Trim(items[idx++]);
        tc.weightShape = Trim(items[idx++]);
        tc.scaleShape = Trim(items[idx++]);
        tc.biasShape = Trim(items[idx++]);
        tc.pertokenScaleShape = Trim(items[idx++]);
        tc.groupListShape = Trim(items[idx++]);
        tc.sharedInputShape = Trim(items[idx++]);
        tc.logitShape = Trim(items[idx++]);
        tc.rowIndexShape = Trim(items[idx++]);
        tc.offsetShape = Trim(items[idx++]);

        tc.xDtype = Trim(items[idx++]);
        tc.weightDtype = Trim(items[idx++]);
        tc.scaleDtype = Trim(items[idx++]);
        tc.biasDtype = Trim(items[idx++]);
        tc.pertokenScaleDtype = Trim(items[idx++]);
        tc.groupListDtype = Trim(items[idx++]);
        tc.sharedInputDtype = Trim(items[idx++]);
        tc.logitDtype = Trim(items[idx++]);
        tc.rowIndexDtype = Trim(items[idx++]);
        tc.offsetDtype = Trim(items[idx++]);
        tc.yDtype = Trim(items[idx++]);

        tc.xFormat = Trim(items[idx++]);
        tc.weightFormat = Trim(items[idx++]);
        tc.scaleFormat = Trim(items[idx++]);
        tc.biasFormat = Trim(items[idx++]);
        tc.pertokenScaleFormat = Trim(items[idx++]);
        tc.groupListFormat = Trim(items[idx++]);
        tc.sharedInputFormat = Trim(items[idx++]);
        tc.logitFormat = Trim(items[idx++]);
        tc.rowIndexFormat = Trim(items[idx++]);
        tc.offsetFormat = Trim(items[idx++]);
        tc.yFormat = Trim(items[idx++]);

        tc.dtype = stoll(Trim(items[idx++]));
        tc.sharedInputWeight = stof(Trim(items[idx++]));
        tc.sharedInputOffset = stoll(Trim(items[idx++]));
        tc.transposeX = ParseBool(Trim(items[idx++]));
        tc.transposeW = ParseBool(Trim(items[idx++]));
        tc.outputBs = stoll(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        cases.push_back(tc);
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingInfershapeCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}
} // namespace

namespace GroupedMatmulFinalizeRoutingInfershapeUT {
const vector<GroupedMatmulFinalizeRoutingInfershapeCase> &GetAscend950Cases()
{
    static const vector<GroupedMatmulFinalizeRoutingInfershapeCase> cases = LoadCases("Ascend950");
    return cases;
}

class TestGroupedMatmulFinalizeRoutingInfershape
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingInfershapeCase> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedMatmulFinalizeRoutingInfershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_FINALIZE_ROUTING_INFERSHAPE_950, TestGroupedMatmulFinalizeRoutingInfershape,
                         testing::ValuesIn(GetAscend950Cases()), MakeParamName);
} // namespace GroupedMatmulFinalizeRoutingInfershapeUT
