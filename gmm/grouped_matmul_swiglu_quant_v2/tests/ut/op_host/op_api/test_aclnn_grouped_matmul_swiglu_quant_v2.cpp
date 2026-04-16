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
 * \file test_aclnn_grouped_matmul_swiglu_quant_v2.cpp
 * \brief CSV-driven opapi UT for grouped_matmul_swiglu_quant_v2 (aligned with gmm grouped_matmul opapi UT style).
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
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
constexpr size_t kSwigluCsvColumnCount = 45;

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
    string path = __FILE__;
    auto pos = path.find_last_of("\\/");
    return pos == string::npos ? string(".") : path.substr(0, pos);
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
    candidates.emplace_back(JoinPath(cwd, "../../../../../gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host/op_api/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host/op_api/" + csvName));
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

vector<int64_t> ParseI64List(const string &value, const string &sep = "|")
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> tokens;
    SplitStr2Vec(trimmed, sep, tokens);
    vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(stoll(Trim(token)));
    }
    return out;
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

vector<vector<int64_t>> ParseDimsList(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> items;
    SplitStr2Vec(trimmed, ";", items);
    vector<vector<int64_t>> out;
    out.reserve(items.size());
    for (const auto &item : items) {
        out.emplace_back(ParseDims(item));
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
        {"INT4", ACL_INT4},
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

void FallbackSingleNzWeightToNdStorage(const string &api, aclFormat &weightFormat,
                                       vector<vector<int64_t>> &weightShapesVec,
                                       vector<vector<int64_t>> &weightStorageShapesVec)
{
    if (api != "WeightNzV2" || weightFormat != ACL_FORMAT_FRACTAL_NZ) {
        return;
    }
    if (weightShapesVec.size() != 1 || !weightStorageShapesVec.empty()) {
        return;
    }
    const auto &w = weightShapesVec[0];
    if (w.size() != 5 || w[0] <= 0 || w[3] != 16 || w[4] != 32) {
        return;
    }
    // Some environments crash while creating single 5D FRACTAL_NZ desc in TensorList.
    // Fallback to equivalent ND view + NZ storage, same as stable hand-written UT path.
    const int64_t e = w[0];
    const int64_t n = w[1] * w[4];
    const int64_t k = w[2] * w[3];
    weightShapesVec[0] = {e, k, n};
    weightStorageShapesVec = {w};
    weightFormat = ACL_FORMAT_ND;
}

struct SwigluOpApiCase {
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
    string weightShapes;
    string weightDtype;
    string weightFormat;
    string weightStorageShapes;
    string weightStrides;
    string weightScaleShapes;
    string weightScaleDtype;
    string weightScaleFormat;
    string weightScaleStorageShapes;
    string weightScaleStrides;
    string weightAssistShapes;
    string weightAssistDtype;
    string weightAssistFormat;
    string weightAssistStorageShapes;
    string weightAssistStrides;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    string smoothScaleShape;
    string smoothScaleDtype;
    string smoothScaleFormat;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    int64_t dequantMode;
    int64_t dequantDtype;
    int64_t quantMode;
    int64_t quantDtype;
    string tuningConfig;
    string out1Shape;
    string out1Dtype;
    string out1Format;
    string out2Shape;
    string out2Dtype;
    string out2Format;

    void Run() const
    {
        SetupPlatformForCase(socVersion);
        // Same tensor style as grouped_matmul_swiglu_quant_v2_opapi_csv_test hand-written UT:
        // TensorDesc(...).ValueRange(-10, 10) and TensorListDesc(vector<TensorDesc>).
        auto weightShapesVec = ParseDimsList(weightShapes);
        auto weightStorageShapesVec = ParseDimsList(weightStorageShapes);
        auto weightFormatVal = ParseFormat(weightFormat);
        auto weightScaleShapesVec = ParseDimsList(weightScaleShapes);
        FallbackSingleNzWeightToNdStorage(api, weightFormatVal, weightShapesVec, weightStorageShapesVec);

        TensorDesc x_desc =
            TensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat)).ValueRange(-10, 10);
        vector<TensorDesc> weight_tensor_vec;
        weight_tensor_vec.reserve(weightShapesVec.size());
        for (const auto &dims : weightShapesVec) {
            weight_tensor_vec.emplace_back(
                TensorDesc(dims, ParseDtype(weightDtype), weightFormatVal).ValueRange(-10, 10));
        }
        TensorListDesc weight_desc(weight_tensor_vec);

        TensorDesc xScale_desc =
            TensorDesc(ParseDims(xScaleShape), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat)).ValueRange(-10, 10);
        TensorDesc groupList_desc =
            TensorDesc(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat))
                .ValueRange(-10, 10);
        TensorDesc out1_desc =
            TensorDesc(ParseDims(out1Shape), ParseDtype(out1Dtype), ParseFormat(out1Format)).ValueRange(-10, 10);
        TensorDesc out2_desc =
            TensorDesc(ParseDims(out2Shape), ParseDtype(out2Dtype), ParseFormat(out2Format)).ValueRange(-10, 10);

        const auto weightAssistShapesVec = ParseDimsList(weightAssistShapes);
        const auto smoothScaleDims = ParseDims(smoothScaleShape);
        const bool hasWeightScale = !weightScaleShapesVec.empty();
        const bool hasWeightAssist = !weightAssistShapesVec.empty();
        const bool hasSmoothScale = !smoothScaleDims.empty();

        vector<int64_t> tuningValues = ParseI64List(tuningConfig, "|");
        aclIntArray *tuningConfigArr = tuningValues.empty() ? nullptr : aclCreateIntArray(tuningValues.data(), tuningValues.size());

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;

        // Optional inputs must use nullptr; do not build empty TensorListDesc.
        if (!hasWeightScale) {
            if (hasWeightAssist && hasSmoothScale) {
                vector<TensorDesc> weight_assist_tensor_vec;
                weight_assist_tensor_vec.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat)).ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec);
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat)).ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasWeightAssist) {
                vector<TensorDesc> weight_assist_tensor_vec_b;
                weight_assist_tensor_vec_b.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec_b.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat)).ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec_b);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasSmoothScale) {
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat)).ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else {
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            }
        } else {
            vector<TensorDesc> weight_scale_tensor_vec;
            weight_scale_tensor_vec.reserve(weightScaleShapesVec.size());
            for (const auto &dims : weightScaleShapesVec) {
                weight_scale_tensor_vec.emplace_back(
                    TensorDesc(dims, ParseDtype(weightScaleDtype), ParseFormat(weightScaleFormat)).ValueRange(-10, 10));
            }
            TensorListDesc weight_scale_desc(weight_scale_tensor_vec);
            if (hasWeightAssist && hasSmoothScale) {
                vector<TensorDesc> weight_assist_tensor_vec_c;
                weight_assist_tensor_vec_c.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec_c.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat)).ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec_c);
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat)).ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasWeightAssist) {
                vector<TensorDesc> weight_assist_tensor_vec_d;
                weight_assist_tensor_vec_d.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec_d.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat)).ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec_d);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasSmoothScale) {
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat)).ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else {
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr, groupList_desc,
                                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            }
        }

        if (tuningConfigArr != nullptr) {
            aclDestroyIntArray(tuningConfigArr);
        }
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }
};

vector<SwigluOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<SwigluOpApiCase> cases;
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
        if (cols.size() != kSwigluCsvColumnCount) {
            continue;
        }
        SwigluOpApiCase c;
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
        c.weightShapes = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.weightStorageShapes = Trim(cols[i++]);
        c.weightStrides = Trim(cols[i++]);
        c.weightScaleShapes = Trim(cols[i++]);
        c.weightScaleDtype = Trim(cols[i++]);
        c.weightScaleFormat = Trim(cols[i++]);
        c.weightScaleStorageShapes = Trim(cols[i++]);
        c.weightScaleStrides = Trim(cols[i++]);
        c.weightAssistShapes = Trim(cols[i++]);
        c.weightAssistDtype = Trim(cols[i++]);
        c.weightAssistFormat = Trim(cols[i++]);
        c.weightAssistStorageShapes = Trim(cols[i++]);
        c.weightAssistStrides = Trim(cols[i++]);
        c.xScaleShape = Trim(cols[i++]);
        c.xScaleDtype = Trim(cols[i++]);
        c.xScaleFormat = Trim(cols[i++]);
        c.smoothScaleShape = Trim(cols[i++]);
        c.smoothScaleDtype = Trim(cols[i++]);
        c.smoothScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.dequantMode = stoll(Trim(cols[i++]));
        c.dequantDtype = stoll(Trim(cols[i++]));
        c.quantMode = stoll(Trim(cols[i++]));
        c.quantDtype = stoll(Trim(cols[i++]));
        c.tuningConfig = Trim(cols[i++]);
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

string BuildCaseName(const testing::TestParamInfo<SwigluOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_swiglu_quant_v2_opapi_csv_test : public testing::TestWithParam<SwigluOpApiCase> {};

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_normal_case)	 
 {	 
     int64_t m = 192;	 
     int64_t k = 2048;	 
     int64_t n = 2048;	 
     int64_t e = 4;	 
     int64_t quantGroupSize = 256;	 
 
     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);	 
     TensorDesc weight =	 
         TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);	 
     TensorListDesc weight_desc = TensorListDesc({weight});	 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);	 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});	 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);	 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);	 
     vector<int64_t> tuningConfigVal = { 10 };	 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());	 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);	 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);	 
 
 
     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,	 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,	 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),	 
                         OUTPUT(out1_desc, out2_desc));	 
     uint64_t workspace_size = 0;	 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);	 
     EXPECT_EQ(aclRet, 0);	 
}	 


TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_normal_nz_workspace_case)	 
{	 
     int64_t m = 192;	 
     int64_t k = 2048;	 
     int64_t n = 2048;	 
     int64_t e = 4;	 
     int64_t quantGroupSize = 256;	 
 
     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);	 
     TensorDesc weight = TensorDesc({e, n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 
 
 
     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
}	 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_wrong_nd_case)	 
{	 
     int64_t m = 192;	 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 
 
     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 
 
 
     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
 }	 
 
 
 TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_wrong_nd_no_nz_case)	 
{	 
     int64_t m = 192;	 
     int64_t k = 2048;	 
     int64_t n = 2048;	 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);	 
     TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
} 
 
 
 TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a8_without_weight_assist_matrix_case) 
{
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
 } 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_redundant_weight_assist_matrix_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc weight_assist_matrix = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_assist_matrix_desc = TensorListDesc({weight_assist_matrix}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_matrix_desc, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_normal_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_1d_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     TensorDesc smoothScale_desc = TensorDesc({e}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_2d_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     TensorDesc smoothScale_desc = TensorDesc({e, n / 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 
 
     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_invalid_dim_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     TensorDesc smoothScale_desc = TensorDesc({e, n / 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_wrong_shape_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     TensorDesc smoothScale_desc = TensorDesc({e + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_wtrans_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight = 
         TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {e, 1, k}, 0, {e, k / 64, n / 16, 16, 64}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight}); 
     TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 


TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight0 = 
         TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1); 
     TensorDesc weight1 = 
         TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1); 
     TensorDesc weight2 = 
         TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1); 
     TensorDesc weight3 = 
         TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3}); 
     TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 


TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3}); 
     TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_workspace_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3}); 
     TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 0); 
} 
 
 
TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_wrong_nd_no_nz_case) 
{ 
     int64_t m = 192; 
     int64_t k = 2048; 
     int64_t n = 2048; 
     int64_t e = 4; 
     int64_t quantGroupSize = 256; 

     TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight0 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight1 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight2 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc weight3 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3}); 
     TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3}); 
     TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3); 
     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64); 
     vector<int64_t> tuningConfigVal = { 10 }; 
     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size()); 
     TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1); 
     TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1); 

     auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2, 
                         INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, 
                               nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig), 
                         OUTPUT(out1_desc, out2_desc)); 
     uint64_t workspace_size = 0; 
     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); 
     EXPECT_EQ(aclRet, 161002); 
}

TEST_P(grouped_matmul_swiglu_quant_v2_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_swiglu_quant_v2_opapi_csv,
    grouped_matmul_swiglu_quant_v2_opapi_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_v2.csv"))),
    BuildCaseName);

}  // namespace
