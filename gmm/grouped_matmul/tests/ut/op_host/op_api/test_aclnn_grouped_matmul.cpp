/**
聽* Copyright (c) 2025 Huawei Technologies Co., Ltd.
聽* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
聽* CANN Open Software License Agreement Version 2.0 (the "License").
聽* Please refer to the License for details. You may not use this file except in compliance with the License.
聽* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
聽* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
聽* See LICENSE in the root of the software repository for the full text of the License.
聽*/

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul.h"
#include "../../../../op_api/aclnn_grouped_matmul_v2.h"
#include "../../../../op_api/aclnn_grouped_matmul_v3.h"
#include "../../../../op_api/aclnn_grouped_matmul_v4.h"
#include "../../../../op_api/aclnn_grouped_matmul_v5.h"
#include "../../../../op_api/aclnn_grouped_matmul_weight_nz.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "ut_acl_parse_utils.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kPathBufferSize = 4096;
constexpr size_t kGroupedMatmulCsvColumnCount = 31;
constexpr size_t kGroupedMatmulCsvExtendedColumnCount = 35;
constexpr size_t kGroupedMatmulCsvOptionalTensorColumnCount = 44;

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
    candidates.emplace_back(JoinPath(cwd, "../../../../../gmm/grouped_matmul/tests/ut/op_host/op_api/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/grouped_matmul/tests/ut/op_host/op_api/" + csvName));
    }

    for (const auto &path : candidates) {
        if (FileExists(path)) {
            return path;
        }
    }
    return candidates.front();
}

struct TensorSpec {
    vector<int64_t> viewDims;
    vector<int64_t> stride;
    vector<int64_t> storageDims;
};

vector<int64_t> ParseDimList(const string &value)
{
    return ops::ut::ParseDims(value);
}

TensorSpec ParseTensorSpec(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> sections;
    SplitStr2Vec(trimmed, "#", sections);

    TensorSpec spec;
    if (!sections.empty()) {
        spec.viewDims = ParseDimList(sections[0]);
    }
    if (sections.size() > 1) {
        spec.stride = ParseDimList(sections[1]);
    }
    if (sections.size() > 2) {
        spec.storageDims = ParseDimList(sections[2]);
    }
    return spec;
}

vector<int64_t> ParseDims(const string &value)
{
    return ParseTensorSpec(value).viewDims;
}

vector<int64_t> ParseI64List(const string &value)
{
    return ops::ut::ParseI64List(value);
}

aclDataType ParseDtype(const string &dtype)
{
    return ops::ut::ParseAclDtype(dtype);
}

aclFormat ParseFormat(const string &format)
{
    return ops::ut::ParseAclFormat(format);
}

TensorDesc BuildTensorDesc(const string &shapeSpec, const string &dtype, const string &format)
{
    const TensorSpec spec = ParseTensorSpec(shapeSpec);
    return TensorDesc(spec.viewDims, ParseDtype(dtype), ParseFormat(format), spec.stride, 0, spec.storageDims);
}

vector<string> ParseTensorListSpecs(const string &value)
{
    vector<string> specs;
    SplitStr2Vec(Trim(value), ";", specs);
    if (specs.empty()) {
        specs.emplace_back("NONE");
    }
    for (auto &spec : specs) {
        spec = Trim(spec);
        if (spec.empty()) {
            spec = "NONE";
        }
    }
    return specs;
}

TensorListDesc BuildTensorListDesc(const string &shapeSpecList, const string &dtype, const string &format,
                                   bool withValueRange = true)
{
    vector<TensorDesc> tensors;
    for (const auto &shapeSpec : ParseTensorListSpecs(shapeSpecList)) {
        TensorDesc tensor = BuildTensorDesc(shapeSpec, dtype, format);
        if (withValueRange) {
            tensor.ValueRange(-10, 10);
        }
        tensors.emplace_back(tensor);
    }
    return TensorListDesc(tensors);
}
void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend310P", op::SocVersion::ASCEND310P},
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

struct GroupedMatmulOpApiCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        TensorListDesc x = BuildTensorListDesc(xShape, xDtype, xFormat);
        TensorListDesc weight = BuildTensorListDesc(weightShape, weightDtype, weightFormat);
        TensorListDesc out = BuildTensorListDesc(outShape, outDtype, outFormat, false);
        TensorDesc groupList(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat));
        vector<int64_t> groupListVal = ParseI64List(groupListValues);
        IntArrayDesc groupListArray(groupListVal);
        if (!groupListVal.empty()) {
            groupList.Value(groupListVal);
        }
        TensorListDesc scale = BuildTensorListDesc(scaleShape, scaleDtype, scaleFormat);
        TensorListDesc bias = BuildTensorListDesc(biasShape, biasDtype, biasFormat);
        TensorListDesc perTokenScale = BuildTensorListDesc(perTokenScaleShape, perTokenScaleDtype, perTokenScaleFormat);
        TensorListDesc offset = BuildTensorListDesc(offsetShape, offsetDtype, offsetFormat);
        TensorListDesc antiquantScale = BuildTensorListDesc(antiquantScaleShape, antiquantScaleDtype, antiquantScaleFormat);
        TensorListDesc antiquantOffset = BuildTensorListDesc(antiquantOffsetShape, antiquantOffsetDtype,
                                                            antiquantOffsetFormat);
        auto activationInputOptional = nullptr;
        auto activationQuantScaleOptional = nullptr;
        auto activationQuantOffsetOptional = nullptr;
        auto tuningConfigOptional = nullptr;
        auto activationFeatureOutOptional = nullptr;
        auto dynQuantScaleOutOptional = nullptr;
        auto biasOptional = nullptr;
        auto scaleOptional = nullptr;
        auto perTokenScaleOptional = nullptr;

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;
        const bool enableBias = ParseBool(hasBias);
        const bool enableScale = ParseBool(hasScale);
        const bool enablePerToken = ParseBool(hasPerTokenScale);

        if (api == "V1") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, bias, scale, offset, antiquantScale,
                                                               antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                                               antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                                               antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, biasOptional, scaleOptional, offset,
                                                               antiquantScale, antiquantOffset, groupListArray,
                                                               splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V2") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, bias, scale, offset, antiquantScale,
                                                                 antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                                                 antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                                                 antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, biasOptional, scaleOptional, offset,
                                                                 antiquantScale, antiquantOffset, groupListArray,
                                                                 splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V3") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, bias, scale, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, bias, scaleOptional, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, biasOptional, scale, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, biasOptional, scaleOptional, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "WeightNz") {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V4") {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        }
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string xShape;
    string xDtype;
    string xFormat;
    string weightShape;
    string weightDtype;
    string weightFormat;
    string scaleShape;
    string scaleDtype;
    string scaleFormat;
    string biasShape;
    string biasDtype;
    string biasFormat;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string perTokenScaleFormat;
    string groupListShape;
    string groupListValues;
    string groupListDtype;
    string groupListFormat;
    string outShape;
    string outDtype;
    string outFormat;
    int64_t splitItem;
    int64_t groupType;
    int64_t groupListType;
    int64_t actType;
    uint64_t expectRet;
    string hasBias;
    string hasPerTokenScale;
    string api = "V5";
    string checkRet = "true";
    string hasScale = "true";
    int64_t quantGroupSize = 0;
    string offsetShape = "0";
    string offsetDtype = "FLOAT";
    string offsetFormat = "ND";
    string antiquantScaleShape = "0";
    string antiquantScaleDtype = "FLOAT16";
    string antiquantScaleFormat = "ND";
    string antiquantOffsetShape = "0";
    string antiquantOffsetDtype = "FLOAT16";
    string antiquantOffsetFormat = "ND";
};

vector<GroupedMatmulOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulOpApiCase> cases;
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
        if (cols.size() != kGroupedMatmulCsvColumnCount &&
            cols.size() != kGroupedMatmulCsvExtendedColumnCount &&
            cols.size() != kGroupedMatmulCsvOptionalTensorColumnCount) {
            continue;
        }
        GroupedMatmulOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.xShape = Trim(cols[i++]);
        c.xDtype = Trim(cols[i++]);
        c.xFormat = Trim(cols[i++]);
        c.weightShape = Trim(cols[i++]);
        c.weightDtype = Trim(cols[i++]);
        c.weightFormat = Trim(cols[i++]);
        c.scaleShape = Trim(cols[i++]);
        c.scaleDtype = Trim(cols[i++]);
        c.scaleFormat = Trim(cols[i++]);
        c.biasShape = Trim(cols[i++]);
        c.biasDtype = Trim(cols[i++]);
        c.biasFormat = Trim(cols[i++]);
        c.perTokenScaleShape = Trim(cols[i++]);
        c.perTokenScaleDtype = Trim(cols[i++]);
        c.perTokenScaleFormat = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.groupListDtype = Trim(cols[i++]);
        c.groupListFormat = Trim(cols[i++]);
        c.outShape = Trim(cols[i++]);
        c.outDtype = Trim(cols[i++]);
        c.outFormat = Trim(cols[i++]);
        c.splitItem = stoll(Trim(cols[i++]));
        c.groupType = stoll(Trim(cols[i++]));
        c.groupListType = stoll(Trim(cols[i++]));
        c.actType = stoll(Trim(cols[i++]));
        c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
        c.hasBias = Trim(cols[i++]);
        c.hasPerTokenScale = Trim(cols[i++]);
        if (i < cols.size()) {
            c.api = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.checkRet = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.hasScale = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.quantGroupSize = stoll(Trim(cols[i++]));
        }
        if (i < cols.size()) {
            c.offsetShape = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.offsetDtype = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.offsetFormat = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.antiquantScaleShape = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.antiquantScaleDtype = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.antiquantScaleFormat = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.antiquantOffsetShape = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.antiquantOffsetDtype = Trim(cols[i++]);
        }
        if (i < cols.size()) {
            c.antiquantOffsetFormat = Trim(cols[i++]);
        }
        cases.emplace_back(c);
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class grouped_matmul_opapi_csv_test : public testing::TestWithParam<GroupedMatmulOpApiCase> {};

TEST_P(grouped_matmul_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_opapi_csv,
    grouped_matmul_opapi_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_grouped_matmul.csv"))),
    BuildCaseName);

}  // namespace


