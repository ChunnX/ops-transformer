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
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz.h"
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
constexpr size_t kCsvColumnCount = 31;

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
    string path = __FILE__;
    auto pos = path.find_last_of("\\/");
    return pos == string::npos ? string(".") : path.substr(0, pos);
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
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
}

vector<int64_t> ParseI64ListPipe(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty()) {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, "|", tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
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
        {"BF16", ACL_BF16},
        {"INT8", ACL_INT8},
        {"INT64", ACL_INT64},
        {"UINT64", ACL_UINT64},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
    };
    auto it = kMap.find(dtype);
    return it == kMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> kMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
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
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t> &storageShape, const vector<int64_t> &stride)
{
    if (format == ACL_FORMAT_ND && storageShape.empty() && stride.empty()) {
        return TensorDesc(shape, dtype, format).ValueRange(-10, 10);
    }
    return TensorDesc(shape, dtype, format, stride, 0, storageShape).ValueRange(-10, 10);
}

struct SwigluQuantV1CsvCase {
    string socVersion;
    string caseName;
    string api;
    string checkRet;
    uint64_t expectRet;
    string xShape;
    string xDtype;
    string xFormat;
    string xStorageShape;
    string xStride;
    string weightShape;
    string weightDtype;
    string weightFormat;
    string weightStorageShape;
    string weightStride;
    string weightScaleShape;
    string weightScaleDtype;
    string weightScaleFormat;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    string groupListValues;
    string out1Shape;
    string out1Dtype;
    string out1Format;
    string out2Shape;
    string out2Dtype;
    string out2Format;

    void Run() const
    {
        SetupPlatformForCase(socVersion);
        auto x = MakeTensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat), ParseDims(xStorageShape),
                               ParseDims(xStride))
                     .ValueRange(-1, 1);
        const auto wShape = ParseDims(weightShape);
        const auto wStorage = ParseDims(weightStorageShape);
        const auto wStride = ParseDims(weightStride);
        TensorDesc weight = MakeTensorDesc(wShape, ParseDtype(weightDtype), ParseFormat(weightFormat), wStorage, wStride)
                                .ValueRange(-1, 1);
        auto weightScale = MakeTensorDesc(ParseDims(weightScaleShape), ParseDtype(weightScaleDtype),
                                        ParseFormat(weightScaleFormat), {}, {})
                               .ValueRange(0, 3);
        auto xScale =
            MakeTensorDesc(ParseDims(xScaleShape), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat), {}, {})
                .ValueRange(0, 3);
        auto groupList =
            MakeTensorDesc(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat), {}, {})
                .ValueRange(0, 64);
        const auto glVals = ParseI64ListPipe(groupListValues);
        if (!glVals.empty()) {
            groupList.Value(glVals);
        }
        auto out1 = MakeTensorDesc(ParseDims(out1Shape), ParseDtype(out1Dtype), ParseFormat(out1Format), {}, {})
                        .ValueRange(-1, 1);
        auto out2 = MakeTensorDesc(ParseDims(out2Shape), ParseDtype(out2Dtype), ParseFormat(out2Format), {}, {})
                        .ValueRange(-1, 1);

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;
        if (api == "WeightNz") {
            auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNZ,
                                INPUT(x, weight, nullptr, nullptr, weightScale, xScale, groupList),
                                OUTPUT(out1, out2, nullptr));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuant,
                                INPUT(x, weight, nullptr, nullptr, weightScale, xScale, groupList),
                                OUTPUT(out1, out2, nullptr));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }
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

vector<SwigluQuantV1CsvCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<SwigluQuantV1CsvCase> cases;
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
        SwigluQuantV1CsvCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.api = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.xShape = Trim(cols[i++]);
        c.xDtype = Trim(cols[i++]);
        c.xFormat = Trim(cols[i++]);
        c.xStorageShape = Trim(cols[i++]);
        c.xStride = Trim(cols[i++]);
        c.weightShape = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.weightStorageShape = Trim(cols[i++]);
        c.weightStride = Trim(cols[i++]);
        c.weightScaleShape = Trim(cols[i++]);
        c.weightScaleDtype = Trim(cols[i++]);
        c.weightScaleFormat = Trim(cols[i++]);
        c.xScaleShape = Trim(cols[i++]);
        c.xScaleDtype = Trim(cols[i++]);
        c.xScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.out1Shape = Trim(cols[i++]);
        c.out1Dtype = Trim(cols[i++]);
        c.out1Format = Trim(cols[i++]);
        c.out2Shape = Trim(cols[i++]);
        c.out2Dtype = Trim(cols[i++]);
        c.out2Format = Trim(cols[i++]);
        if (c.socVersion == "Ascend950" || c.socVersion == "Ascend910B") {
            cases.emplace_back(c);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<SwigluQuantV1CsvCase> &info)
{
    string name = info.param.caseName;
    for (char &ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
            ch = '_';
        }
    }
    return name;
}

class grouped_matmul_swiglu_quant_opapi_csv_test : public testing::TestWithParam<SwigluQuantV1CsvCase> {};

TEST_P(grouped_matmul_swiglu_quant_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_swiglu_quant_opapi_csv, grouped_matmul_swiglu_quant_opapi_csv_test,
                         testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_l2.csv"))),
                         BuildCaseName);

} // namespace
