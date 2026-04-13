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

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_v3.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace op;

namespace {
constexpr size_t kPathBufferSize = 4096;
constexpr size_t kV3CsvColumnCount = 42;

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
        JoinPath(cwd, "../../../../../gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api/" + csvName));
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
    transform(lower.begin(), lower.end(), lower.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
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
    vector<int64_t> vals;
    for (const auto &token : tokens) {
        vals.emplace_back(stoll(Trim(token)));
    }
    return vals;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},
        {"BF16", ACL_BF16},
        {"INT8", ACL_INT8},
        {"INT4", ACL_INT4},
        {"INT32", ACL_INT32},
        {"INT64", ACL_INT64},
        {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"FLOAT4_E2M1", ACL_FLOAT4_E2M1},
        {"HIFLOAT8", ACL_HIFLOAT8},
        {"NONE", ACL_DT_UNDEFINED},
    };
    auto it = dtypeMap.find(Trim(dtype));
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> formatMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
        {"NONE", ACL_FORMAT_ND},
    };
    auto it = formatMap.find(Trim(format));
    return it == formatMap.end() ? ACL_FORMAT_ND : it->second;
}

aclnnStatus ParseStatus(const string &status)
{
    static const map<string, aclnnStatus> statusMap = {
        {"SUCCESS", ACLNN_SUCCESS},
        {"ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    auto it = statusMap.find(Trim(status));
    return it == statusMap.end() ? ACLNN_SUCCESS : it->second;
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, SocVersion> socMap = {
        {"Ascend910B", SocVersion::ASCEND910B},
        {"Ascend950", SocVersion::ASCEND950},
    };
    auto it = socMap.find(Trim(socVersion));
    SetPlatformSocVersion(it == socMap.end() ? SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t> &storageShape)
{
    if (storageShape.empty()) {
        return TensorDesc(shape, dtype, format).ValueRange(-1, 1);
    }
    return TensorDesc(shape, dtype, format, {}, 0, storageShape).ValueRange(-1, 1);
}

struct GroupedMatmulFinalizeRoutingV3Case {
    void Run() const
    {
        SetupPlatformForCase(socVersion);

        TensorDesc x1 = MakeTensorDesc(ParseDims(x1Shape), ParseDtype(x1Dtype), ACL_FORMAT_ND, {});
        TensorDesc x2 =
            MakeTensorDesc(ParseDims(x2Shape), ParseDtype(x2Dtype), ParseFormat(x2Format), ParseDims(x2StorageShape));
        TensorDesc scale =
            MakeTensorDesc(ParseDims(scaleShape), ParseDtype(scaleDtype), ACL_FORMAT_ND, {}).ValueRange(0, 3);
        TensorDesc bias = MakeTensorDesc(ParseDims(biasShape), ParseDtype(biasDtype), ACL_FORMAT_ND, {});
        TensorDesc offset =
            MakeTensorDesc(ParseDims(offsetShape), ParseDtype(offsetDtype), ParseFormat(offsetFormat), {});
        TensorDesc antiQuantScale =
            MakeTensorDesc(ParseDims(antiQuantScaleShape), ParseDtype(antiQuantScaleDtype), ACL_FORMAT_ND, {});
        TensorDesc perTokenScale =
            MakeTensorDesc(ParseDims(perTokenScaleShape), ParseDtype(perTokenScaleDtype), ACL_FORMAT_ND, {})
                .ValueRange(0, 3);
        TensorDesc groupList =
            MakeTensorDesc(ParseDims(groupListShape), ACL_INT64, ACL_FORMAT_ND, {}).ValueRange(-1, 1);
        const vector<int64_t> groupListVals = ParseI64List(groupListValues);
        if (!groupListVals.empty()) {
            groupList.Value(groupListVals);
        }
        TensorDesc sharedInput =
            MakeTensorDesc(ParseDims(sharedInputShape), ParseDtype(sharedInputDtype), ACL_FORMAT_ND, {});
        TensorDesc logits = MakeTensorDesc(ParseDims(logitsShape), ParseDtype(logitsDtype), ACL_FORMAT_ND, {});
        TensorDesc rowIndex =
            MakeTensorDesc(ParseDims(rowIndexShape), ParseDtype(rowIndexDtype), ACL_FORMAT_ND, {});
        TensorDesc out = MakeTensorDesc(ParseDims(outShape), ParseDtype(outDtype), ACL_FORMAT_ND, {});

        const vector<int64_t> tuningVals = ParseI64List(tuningConfig);
        aclIntArray *tuningConfigPtr = nullptr;
        if (ParseBool(hasTuningConfig)) {
            tuningConfigPtr = aclCreateIntArray(const_cast<int64_t *>(tuningVals.data()), tuningVals.size());
        }

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;

        if (!ParseBool(hasOffset)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, nullptr, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasLogits)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, offset, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, nullptr, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasAntiQuantScale)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, offset, antiQuantScale, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, offset, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, ParseStatus(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string x1Shape;
    string x1Dtype;
    string x2Shape;
    string x2Dtype;
    string x2Format;
    string x2StorageShape;
    string scaleShape;
    string scaleDtype;
    string biasShape;
    string biasDtype;
    string offsetShape;
    string offsetDtype;
    string offsetFormat;
    string antiQuantScaleShape;
    string antiQuantScaleDtype;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string groupListShape;
    string groupListValues;
    string sharedInputShape;
    string sharedInputDtype;
    string logitsShape;
    string logitsDtype;
    string rowIndexShape;
    string rowIndexDtype;
    string outShape;
    string outDtype;
    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t groupListType = 1;
    string tuningConfig;
    string hasTuningConfig;
    string hasOffset;
    string hasAntiQuantScale;
    string hasLogits;
    string expectRet;
    string checkRet;
};

vector<GroupedMatmulFinalizeRoutingV3Case> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulFinalizeRoutingV3Case> cases;
    string line;
    bool skippedHeader = false;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!skippedHeader) {
            skippedHeader = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != kV3CsvColumnCount) {
            continue;
        }
        GroupedMatmulFinalizeRoutingV3Case c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.x1Shape = Trim(cols[i++]);
        c.x1Dtype = Trim(cols[i++]);
        c.x2Shape = Trim(cols[i++]);
        c.x2Dtype = Trim(cols[i++]);
        c.x2Format = Trim(cols[i++]);
        c.x2StorageShape = Trim(cols[i++]);
        c.scaleShape = Trim(cols[i++]);
        c.scaleDtype = Trim(cols[i++]);
        c.biasShape = Trim(cols[i++]);
        c.biasDtype = Trim(cols[i++]);
        c.offsetShape = Trim(cols[i++]);
        c.offsetDtype = Trim(cols[i++]);
        c.offsetFormat = Trim(cols[i++]);
        c.antiQuantScaleShape = Trim(cols[i++]);
        c.antiQuantScaleDtype = Trim(cols[i++]);
        c.perTokenScaleShape = Trim(cols[i++]);
        c.perTokenScaleDtype = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.sharedInputShape = Trim(cols[i++]);
        c.sharedInputDtype = Trim(cols[i++]);
        c.logitsShape = Trim(cols[i++]);
        c.logitsDtype = Trim(cols[i++]);
        c.rowIndexShape = Trim(cols[i++]);
        c.rowIndexDtype = Trim(cols[i++]);
        c.outShape = Trim(cols[i++]);
        c.outDtype = Trim(cols[i++]);
        c.dtype = stoll(Trim(cols[i++]));
        c.sharedInputWeight = stof(Trim(cols[i++]));
        c.sharedInputOffset = stoll(Trim(cols[i++]));
        c.transposeX = ParseBool(Trim(cols[i++]));
        c.transposeW = ParseBool(Trim(cols[i++]));
        c.groupListType = stoll(Trim(cols[i++]));
        c.tuningConfig = Trim(cols[i++]);
        c.hasTuningConfig = Trim(cols[i++]);
        c.hasOffset = Trim(cols[i++]);
        c.hasAntiQuantScale = Trim(cols[i++]);
        c.hasLogits = Trim(cols[i++]);
        c.expectRet = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        cases.emplace_back(c);
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingV3Case> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_finalize_routing_v3_csv_test
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingV3Case> {};

TEST_P(grouped_matmul_finalize_routing_v3_csv_test, run_case)
{
    GetParam().Run();
}

TEST(grouped_matmul_finalize_routing_v3_manual_test, shared_input_optional_on_ascend950)
{
    SetupPlatformForCase("Ascend950");

    TensorDesc x1({192, 2048}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2({4, 2048, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc scale({4, 32, 7168, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias({4, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale({192, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList({4}, ACL_INT64, ACL_FORMAT_ND);
    groupList.Value(vector<int64_t>{24, 24, 24, 24});
    TensorDesc logits({192}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc rowIndex({192}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out({24, 7168}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                        INPUT(x1, x2, scale, bias, nullptr, nullptr, nullptr, perTokenScale, groupList, nullptr,
                              logits, rowIndex, 0, 1.0F, 0, false, false, 1, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_finalize_routing_v3_csv,
    grouped_matmul_finalize_routing_v3_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul_finalize_routing_v3_l2.csv"))),
    BuildCaseName);
}  // namespace
