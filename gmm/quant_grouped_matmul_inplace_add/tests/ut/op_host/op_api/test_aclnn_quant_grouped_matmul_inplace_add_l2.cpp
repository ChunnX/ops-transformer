/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_quant_grouped_matmul_inplace_add.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
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
constexpr size_t kCsvColumnCount = 25;

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
    ssize_t rd = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (rd <= 0) {
        return "./";
    }
    path[rd] = '\0';
    string exePath(path);
    auto pos = exePath.find_last_of('/');
    return pos == string::npos ? string("./") : exePath.substr(0, pos + 1);
#endif
}

string GetCurrentFileDir()
{
    string p = __FILE__;
    auto pos = p.find_last_of("\\/");
    return pos == string::npos ? string(".") : p.substr(0, pos);
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
    const string exePath = JoinPath(GetExeDirPath(), csvName);
    if (FileExists(exePath)) {
        return exePath;
    }
    const string srcPath = JoinPath(GetCurrentFileDir(), csvName);
    if (FileExists(srcPath)) {
        return srcPath;
    }
    return exePath;
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

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty()) {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> out;
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

vector<int64_t> ParseGroupListValues(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.rfind("R|", 0) == 0) {
        vector<string> parts;
        SplitStr2Vec(trimmed, "|", parts);
        if (parts.size() >= 3U) {
            const int64_t v = stoll(Trim(parts[1]));
            const size_t n = static_cast<size_t>(stoull(Trim(parts[2])));
            return vector<int64_t>(n, v);
        }
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, "|", tokens);
    vector<int64_t> out;
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> kMap = {
        {"FLOAT", ACL_FLOAT},
        {"FLOAT16", ACL_FLOAT16},
        {"INT64", ACL_INT64},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"HIFLOAT8", ACL_HIFLOAT8},
    };
    auto it = kMap.find(dtype);
    return it == kMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> kMap = {
        {"ND", ACL_FORMAT_ND},
    };
    auto it = kMap.find(format);
    return it == kMap.end() ? ACL_FORMAT_ND : it->second;
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND950 : it->second);
}

TensorDesc MakeNdTensor(const vector<int64_t> &shape, aclDataType dtype)
{
    return TensorDesc(shape, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
}

struct QgmmInplaceAddCsvCase {
    string socVersion;
    string caseName;
    string checkRet;
    uint64_t expectRet;
    string x1Shape;
    string x1Dtype;
    string x1Format;
    string x2Shape;
    string x2Dtype;
    string x2Format;
    string scale1Shape;
    string scale1Dtype;
    string scale1Format;
    string scale2Shape;
    string scale2Dtype;
    string scale2Format;
    string yShape;
    string yDtype;
    string yFormat;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    string groupListValues;
    int64_t groupListType;
    int64_t groupSize;

    void Run() const
    {
        SetupPlatformForCase(socVersion);
        auto x1 = MakeNdTensor(ParseDims(x1Shape), ParseDtype(x1Dtype));
        auto x2 = MakeNdTensor(ParseDims(x2Shape), ParseDtype(x2Dtype));
        auto scale1 = MakeNdTensor(ParseDims(scale1Shape), ParseDtype(scale1Dtype));
        auto scale2 = MakeNdTensor(ParseDims(scale2Shape), ParseDtype(scale2Dtype));
        auto y = TensorDesc(ParseDims(yShape), ParseDtype(yDtype), ParseFormat(yFormat)).ValueRange(-1, 1);
        auto groupList =
            TensorDesc(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat))
                .ValueRange(0, 1024);
        const auto glVals = ParseGroupListValues(groupListValues);
        if (!glVals.empty()) {
            groupList.Value(glVals);
        }
        ScalarDesc glt(groupListType);
        ScalarDesc gs(groupSize);

        uint64_t workspaceSize = 0;
        auto ut = OP_API_UT(aclnnQuantGroupedMatmulInplaceAdd,
                            INPUT(x1, x2, scale1, scale2, groupList, y, glt, gs), OUTPUT());
        aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }

private:
    static bool ParseBool(const string &value)
    {
        string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return lower == "true" || lower == "1" || lower == "yes";
    }
};

vector<QgmmInplaceAddCsvCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<QgmmInplaceAddCsvCase> cases;
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
        if (cols.size() != kCsvColumnCount) {
            continue;
        }
        QgmmInplaceAddCsvCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.x1Shape = Trim(cols[i++]);
        c.x1Dtype = Trim(cols[i++]);
        c.x1Format = Trim(cols[i++]);
        c.x2Shape = Trim(cols[i++]);
        c.x2Dtype = Trim(cols[i++]);
        c.x2Format = Trim(cols[i++]);
        c.scale1Shape = Trim(cols[i++]);
        c.scale1Dtype = Trim(cols[i++]);
        c.scale1Format = Trim(cols[i++]);
        c.scale2Shape = Trim(cols[i++]);
        c.scale2Dtype = Trim(cols[i++]);
        c.scale2Format = Trim(cols[i++]);
        c.yShape = Trim(cols[i++]);
        c.yDtype = Trim(cols[i++]);
        c.yFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.groupListType = stoll(Trim(cols[i++]));
        c.groupSize = stoll(Trim(cols[i++]));
        if (c.socVersion == "Ascend950" || c.socVersion == "Ascend910B") {
            cases.emplace_back(c);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<QgmmInplaceAddCsvCase> &info)
{
    string name = info.param.caseName;
    for (char &ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
            ch = '_';
        }
    }
    return name;
}

class quant_grouped_matmul_inplace_add_opapi_csv_test : public testing::TestWithParam<QgmmInplaceAddCsvCase> {};

TEST_P(quant_grouped_matmul_inplace_add_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(quant_grouped_matmul_inplace_add_opapi_csv, quant_grouped_matmul_inplace_add_opapi_csv_test,
                         testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_quant_grouped_matmul_inplace_add_l2.csv"))),
                         BuildCaseName);

} // namespace
