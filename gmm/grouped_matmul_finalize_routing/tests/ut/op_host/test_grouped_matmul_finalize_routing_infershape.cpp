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
#include <cstdlib>
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

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

using namespace std;

namespace {
constexpr size_t kPathBufferSize = 4096;

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    auto len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    string exePath(path, len);
    auto pos = exePath.find_last_of("\\/");
    return pos == string::npos ? string(".\\") : exePath.substr(0, pos + 1);
#else
    char path[kPathBufferSize] = {0};
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

string GetCurrentFileDir()
{
    string filePath = __FILE__;
    auto pos = filePath.find_last_of("\\/");
    return pos == string::npos ? string(".") : filePath.substr(0, pos);
}

string GetCwd()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetCurrentDirectoryA(MAX_PATH, path);
    return len == 0 ? string(".") : string(path, len);
#else
    char path[kPathBufferSize] = {0};
    if (getcwd(path, sizeof(path)) == nullptr) {
        return ".";
    }
    return string(path);
#endif
}

string JoinPath(const string &dir, const string &file)
{
    if (dir.empty()) {
        return file;
    }
    char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

bool FileExists(const string &path)
{
    ifstream in(path);
    return in.is_open();
}

string ResolveCsvPath(const string &csvName)
{
    const string cwd = GetCwd();
    const string codePath = []() {
        const char *env = std::getenv("CODE_PATH");
        return env == nullptr ? string() : string(env);
    }();

    vector<string> candidates;
    candidates.emplace_back(JoinPath(GetExeDirPath(), csvName));
    candidates.emplace_back(JoinPath(GetCurrentFileDir(), csvName));
    candidates.emplace_back(JoinPath(cwd, csvName));
    candidates.emplace_back(
        JoinPath(cwd, "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_host/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/" + csvName));
    }

    for (const auto &path : candidates) {
        if (FileExists(path)) {
            return path;
        }
    }
    return candidates.front();
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
    static const map<string, ge::DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"INT64", ge::DT_INT64},
        {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
        {"UNDEFINED", ge::DT_UNDEFINED},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

ge::Format ParseFormat(const string &format)
{
    static const map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
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

struct GroupedMatmulFinalizeRoutingInfershapeCase {
    void Run() const
    {
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {MakeStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {MakeStorageShape(ParseDims(wShape)), ParseDtype(wDtype), ParseFormat(wFormat)},
                {MakeStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype), ParseFormat(scaleFormat)},
                {MakeStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ParseFormat(biasFormat)},
                {MakeStorageShape(ParseDims(pertokenScaleShape)), ParseDtype(pertokenScaleDtype),
                 ParseFormat(pertokenScaleFormat)},
                {MakeStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ParseFormat(groupListFormat)},
                {MakeStorageShape(ParseDims(sharedInputShape)), ParseDtype(sharedInputDtype),
                 ParseFormat(sharedInputFormat)},
                {MakeStorageShape(ParseDims(logitShape)), ParseDtype(logitDtype), ParseFormat(logitFormat)},
                {MakeStorageShape(ParseDims(rowIndexShape)), ParseDtype(rowIndexDtype), ParseFormat(rowIndexFormat)},
            },
            {
                {MakeStorageShape(ParseDims("NONE")), ParseDtype(yDtype), ParseFormat(yFormat)},
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

        vector<vector<int64_t>> expectShape = {ParseDims(expectYShape)};
        ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
    }

    string caseName;
    bool enable = true;
    string prefix;
    string caseType;
    bool expectSuccess = false;
    string expectYShape;

    string xShape;
    string wShape;
    string scaleShape;
    string biasShape;
    string pertokenScaleShape;
    string groupListShape;
    string sharedInputShape;
    string logitShape;
    string rowIndexShape;

    string xDtype;
    string wDtype;
    string scaleDtype;
    string biasDtype;
    string pertokenScaleDtype;
    string groupListDtype;
    string sharedInputDtype;
    string logitDtype;
    string rowIndexDtype;
    string yDtype;

    string xFormat;
    string wFormat;
    string scaleFormat;
    string biasFormat;
    string pertokenScaleFormat;
    string groupListFormat;
    string sharedInputFormat;
    string logitFormat;
    string rowIndexFormat;
    string yFormat;

    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t outputBs = 0;
    int64_t groupListType = 1;
};

vector<GroupedMatmulFinalizeRoutingInfershapeCase> LoadCases()
{
    vector<GroupedMatmulFinalizeRoutingInfershapeCase> cases;
    const string csvPath = ResolveCsvPath("test_grouped_matmul_finalize_routing_infershape.csv");
    ifstream csvData(csvPath, ios::in);
    EXPECT_TRUE(csvData.is_open()) << "cannot open case file " << csvPath;
    if (!csvData.is_open()) {
        return cases;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "caseName" || items.size() < 42U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulFinalizeRoutingInfershapeCase tc;
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
        tc.wShape = Trim(items[idx++]);
        tc.scaleShape = Trim(items[idx++]);
        tc.biasShape = Trim(items[idx++]);
        tc.pertokenScaleShape = Trim(items[idx++]);
        tc.groupListShape = Trim(items[idx++]);
        tc.sharedInputShape = Trim(items[idx++]);
        tc.logitShape = Trim(items[idx++]);
        tc.rowIndexShape = Trim(items[idx++]);

        tc.xDtype = Trim(items[idx++]);
        tc.wDtype = Trim(items[idx++]);
        tc.scaleDtype = Trim(items[idx++]);
        tc.biasDtype = Trim(items[idx++]);
        tc.pertokenScaleDtype = Trim(items[idx++]);
        tc.groupListDtype = Trim(items[idx++]);
        tc.sharedInputDtype = Trim(items[idx++]);
        tc.logitDtype = Trim(items[idx++]);
        tc.rowIndexDtype = Trim(items[idx++]);
        tc.yDtype = Trim(items[idx++]);

        tc.xFormat = Trim(items[idx++]);
        tc.wFormat = Trim(items[idx++]);
        tc.scaleFormat = Trim(items[idx++]);
        tc.biasFormat = Trim(items[idx++]);
        tc.pertokenScaleFormat = Trim(items[idx++]);
        tc.groupListFormat = Trim(items[idx++]);
        tc.sharedInputFormat = Trim(items[idx++]);
        tc.logitFormat = Trim(items[idx++]);
        tc.rowIndexFormat = Trim(items[idx++]);
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
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingInfershapeCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

gert::InfershapeContextPara BuildOffsetInfershapeContext(const string &offsetShape)
{
    return gert::InfershapeContextPara(
        "GroupedMatmulFinalizeRouting",
        {
            {MakeStorageShape(ParseDims("1024:2048")), ge::DT_INT8, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("16:2048:7168")), ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},
            {MakeStorageShape(ParseDims("16:1:7168")), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("16:7168")), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("1024")), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("16")), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("64:7168")), ge::DT_BF16, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("1024")), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims("1024")), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape(ParseDims(offsetShape)), ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape(ParseDims("NONE")), ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
}
} // namespace

namespace GroupedMatmulFinalizeRoutingInfershapeUT {
const vector<GroupedMatmulFinalizeRoutingInfershapeCase> &GetFinalizeRoutingInfershapeCases()
{
    static const vector<GroupedMatmulFinalizeRoutingInfershapeCase> cases = LoadCases();
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

TEST(TestGroupedMatmulFinalizeRoutingInfershapeManual, offsetShapeValid)
{
    auto infershapeContextPara = BuildOffsetInfershapeContext("16:1:7168");
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{64, 7168}});
}

TEST(TestGroupedMatmulFinalizeRoutingInfershapeManual, offsetShapeInvalidDimNum)
{
    auto infershapeContextPara = BuildOffsetInfershapeContext("16:7168");
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

INSTANTIATE_TEST_SUITE_P(GMMFR_INFERSHAPE_CSV, TestGroupedMatmulFinalizeRoutingInfershape,
                         testing::ValuesIn(GetFinalizeRoutingInfershapeCases()), MakeParamName);
} // namespace GroupedMatmulFinalizeRoutingInfershapeUT
