/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_quant_grouped_matmul_inplace_add.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;

namespace {

constexpr size_t kPathBufferSize = 4096;
constexpr size_t kCsvColumnCount = 21;

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
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
        JoinPath(cwd, "../../../../../gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host/op_api/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host/op_api/" + csvName));
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

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> dims;
    for (const auto &token : tokens) {
        dims.emplace_back(stoll(Trim(token)));
    }
    return dims;
}

vector<int64_t> ParseI64List(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, "|", tokens);
    vector<int64_t> values;
    for (const auto &token : tokens) {
        values.emplace_back(stoll(Trim(token)));
    }
    return values;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},
        {"INT64", ACL_INT64},
        {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"HIFLOAT8", ACL_HIFLOAT8},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclnnStatus ParseStatus(const string &value)
{
    static const map<string, aclnnStatus> statusMap = {
        {"ACLNN_SUCCESS", ACLNN_SUCCESS},
        {"ACLNN_ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
    };
    auto it = statusMap.find(value);
    if (it != statusMap.end()) {
        return it->second;
    }
    return static_cast<aclnnStatus>(stoll(value));
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

struct QGmmInplaceAddOpApiCase {
    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        SetupPlatformForCase(socVersion);

        TensorDesc x1Desc(ParseDims(x1Shape), ParseDtype(x1Dtype), ACL_FORMAT_ND, ParseDims(x1Stride));
        TensorDesc x2Desc(ParseDims(x2Shape), ParseDtype(x2Dtype), ACL_FORMAT_ND, ParseDims(x2Stride));
        TensorDesc scale2Desc(ParseDims(scale2Shape), ParseDtype(scale2Dtype), ACL_FORMAT_ND);
        TensorDesc yDesc(ParseDims(yShape), ParseDtype(yDtype), ACL_FORMAT_ND);
        TensorDesc scale1Desc(ParseDims(scale1Shape), ParseDtype(scale1Dtype), ACL_FORMAT_ND);
        TensorDesc groupListDesc(ParseDims(groupListShape), ACL_INT64, ACL_FORMAT_ND);
        const vector<int64_t> groupVals = ParseI64List(groupListValues);
        if (!groupVals.empty()) {
            groupListDesc.Value(groupVals);
        }

        auto ut = OP_API_UT(aclnnQuantGroupedMatmulInplaceAdd,
                            INPUT(x1Desc, x2Desc, scale1Desc, scale2Desc, groupListDesc, yDesc, groupListType,
                                  groupSize),
                            OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
        if (checkRet) {
            EXPECT_EQ(ret, ParseStatus(expectRet)) << "case=" << caseName;
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string x1Shape;
    string x1Dtype;
    string x1Stride;
    string x2Shape;
    string x2Dtype;
    string x2Stride;
    string scale2Shape;
    string scale2Dtype;
    string yShape;
    string yDtype;
    string scale1Shape;
    string scale1Dtype;
    string groupListShape;
    string groupListValues;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
    string expectRet;
    bool checkRet = true;
};

vector<QGmmInplaceAddOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<QGmmInplaceAddOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < kCsvColumnCount) {
            continue;
        }

        QGmmInplaceAddOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.enable = ParseBool(Trim(cols[i++]));
        c.x1Shape = Trim(cols[i++]);
        c.x1Dtype = Trim(cols[i++]);
        c.x1Stride = Trim(cols[i++]);
        c.x2Shape = Trim(cols[i++]);
        c.x2Dtype = Trim(cols[i++]);
        c.x2Stride = Trim(cols[i++]);
        c.scale2Shape = Trim(cols[i++]);
        c.scale2Dtype = Trim(cols[i++]);
        c.yShape = Trim(cols[i++]);
        c.yDtype = Trim(cols[i++]);
        c.scale1Shape = Trim(cols[i++]);
        c.scale1Dtype = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.groupListType = stoll(Trim(cols[i++]));
        c.groupSize = stoll(Trim(cols[i++]));
        c.expectRet = Trim(cols[i++]);
        c.checkRet = ParseBool(Trim(cols[i++]));
        cases.emplace_back(c);
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<QGmmInplaceAddOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class qgmm_inplace_add_opapi_csv_test : public testing::TestWithParam<QGmmInplaceAddOpApiCase> {};

TEST_P(qgmm_inplace_add_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    qgmm_inplace_add_opapi_csv,
    qgmm_inplace_add_opapi_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_quant_grouped_matmul_inplace_add.csv"))),
    BuildCaseName);

} // namespace
