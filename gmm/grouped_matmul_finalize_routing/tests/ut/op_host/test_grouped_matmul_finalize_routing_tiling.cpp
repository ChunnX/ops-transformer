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
 * \file test_grouped_matmul_finalize_routing_tiling.cpp
 * \brief CSV-driven unit tests for grouped_matmul_finalize_routing tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "../../../op_host/grouped_matmul_finalize_routing_base_tiling.h"
#include "../../../op_host/grouped_matmul_finalize_routing_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {
constexpr size_t kPathBufferSize = 4096;
constexpr size_t kCsvColumnCount = 48;

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
    string lower = Trim(value);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes";
}

uint64_t ParseU64(const string &value)
{
    const string trimmed = Trim(value);
    return trimmed.empty() ? 0UL : static_cast<uint64_t>(stoull(trimmed));
}

float ParseFloat(const string &value)
{
    const string trimmed = Trim(value);
    return trimmed.empty() ? 0.0F : stof(trimmed);
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

ge::DataType ParseDtype(const string &dtype)
{
    static const map<string, ge::DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"INT64", ge::DT_INT64},
        {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT4_E2M1", ge::DT_FLOAT4_E2M1},
        {"NONE", ge::DT_UNDEFINED},
        {"UNDEFINED", ge::DT_UNDEFINED},
    };
    auto it = dtypeMap.find(Trim(dtype));
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

ge::Format ParseFormat(const string &format)
{
    static const map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
        {"NONE", ge::FORMAT_ND},
    };
    auto it = formatMap.find(Trim(format));
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

ge::graphStatus ParseGraphStatus(const string &status)
{
    return Trim(status) == "SUCCESS" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

template <typename T, typename = void>
struct HasAscend950 : std::false_type {};

template <typename T>
struct HasAscend950<T, std::void_t<decltype(T::ASCEND950)>> : std::true_type {};

template <typename T>
platform_ascendc::SocVersion ParseAscend950Impl(std::true_type)
{
    return T::ASCEND950;
}

template <typename T>
platform_ascendc::SocVersion ParseAscend950Impl(std::false_type)
{
    return platform_ascendc::SocVersion::ASCEND910B;
}

platform_ascendc::SocVersion ParseSocVersion(const string &socVersion)
{
    const string soc = Trim(socVersion);
    if (soc == "Ascend910B") {
        return platform_ascendc::SocVersion::ASCEND910B;
    }
    if (soc == "Ascend950") {
        // Some toolchain variants do not expose ASCEND950 in platform_ascendc::SocVersion.
        // Keep CSV compatibility: prefer ASCEND950 when available, otherwise degrade gracefully.
        return ParseAscend950Impl<platform_ascendc::SocVersion>(HasAscend950<platform_ascendc::SocVersion>{});
    }
    return static_cast<platform_ascendc::SocVersion>(0);
}

NpuArch ParseNpuArch(const string &npuArch)
{
    static const map<string, NpuArch> archMap = {
        {"DAV_3510", NpuArch::DAV_3510},
        {"UNKNOWN", static_cast<NpuArch>(0)},
    };
    auto it = archMap.find(Trim(npuArch));
    return it == archMap.end() ? static_cast<NpuArch>(0) : it->second;
}

struct GroupedMatmulFinalizeRoutingTilingCase {
    void Run() const
    {
        optiling::GroupedMatmulFinalizeRoutingCompileInfo compileInfo;
        compileInfo.aicNum = ParseU64(aicNum);
        compileInfo.aivNum = ParseU64(aivNum);
        compileInfo.ubSize = ParseU64(ubSize);
        compileInfo.l1Size = ParseU64(l1Size);
        compileInfo.l2Size = ParseU64(l2Size);
        compileInfo.l0CSize = ParseU64(l0CSize);
        compileInfo.l0ASize = ParseU64(l0ASize);
        compileInfo.l0BSize = ParseU64(l0BSize);
        compileInfo.btSize = ParseU64(btSize);
        compileInfo.cubeFreq = ParseFloat(cubeFreq);
        compileInfo.socVersion = ParseSocVersion(socVersion);
        compileInfo.supportL0c2out = ParseBool(supportL0c2out);
        compileInfo.supportL12BtBf16 = ParseBool(supportL12BtBf16);
        compileInfo.npuArch = ParseNpuArch(npuArch);

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {MakeStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(wShape)), ParseDtype(wDtype), ParseFormat(wFormat)},
                {MakeStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(pertokenScaleShape)), ParseDtype(pertokenScaleDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(sharedInputShape)), ParseDtype(sharedInputDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(logitShape)), ParseDtype(logitDtype), ge::FORMAT_ND},
                {MakeStorageShape(ParseDims(rowindexShape)), ParseDtype(rowindexDtype), ge::FORMAT_ND},
            },
            {
                {MakeStorageShape(ParseDims(yShape)), ParseDtype(yDtype), ge::FORMAT_ND},
            },
            {
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(dtypeAttr)))},
                {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(ParseFloat(sharedInputWeight))},
                {"shared_input_offset",
                 Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(sharedInputOffset)))},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(ParseBool(transposeX))},
                {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(ParseBool(transposeW))},
                {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(outputBs)))},
                {"group_list_type",
                 Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(groupListType)))},
                {"tuning_config",
                 Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(tuningConfig)))},
            },
            &compileInfo);

        const ge::graphStatus expectStatus = ParseGraphStatus(expectResult);
        if (expectStatus == ge::GRAPH_SUCCESS) {
            TilingInfo tilingInfo;
            bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
            ASSERT_TRUE(tilingResult) << "prefix=" << prefix << ", case=" << caseName;
            EXPECT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), ParseU64(expectTilingKey))
                << "prefix=" << prefix << ", case=" << caseName;
            return;
        }

        ExecuteTestCase(tilingContextPara, expectStatus);
    }

    string caseName;
    bool enable = true;
    string prefix;
    string aicNum;
    string aivNum;
    string ubSize;
    string l1Size;
    string l2Size;
    string l0CSize;
    string l0ASize;
    string l0BSize;
    string btSize;
    string cubeFreq;
    string socVersion;
    string supportL0c2out;
    string supportL12BtBf16;
    string npuArch;
    string xShape;
    string wShape;
    string scaleShape;
    string biasShape;
    string pertokenScaleShape;
    string groupListShape;
    string sharedInputShape;
    string logitShape;
    string rowindexShape;
    string yShape;
    string xDtype;
    string wDtype;
    string scaleDtype;
    string biasDtype;
    string pertokenScaleDtype;
    string groupListDtype;
    string sharedInputDtype;
    string logitDtype;
    string rowindexDtype;
    string yDtype;
    string wFormat;
    string dtypeAttr;
    string sharedInputWeight;
    string sharedInputOffset;
    string transposeX;
    string transposeW;
    string outputBs;
    string groupListType;
    string tuningConfig;
    string expectResult;
    string expectTilingKey;
};

vector<GroupedMatmulFinalizeRoutingTilingCase> LoadCases(const string &csvPath)
{
    vector<GroupedMatmulFinalizeRoutingTilingCase> cases;
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        ADD_FAILURE() << "cannot open case file " << csvPath;
        return cases;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "caseName") {
            continue;
        }
        if (items.size() != kCsvColumnCount) {
            ADD_FAILURE() << "Bad csv row column count in " << csvPath << ": " << line;
            continue;
        }

        GroupedMatmulFinalizeRoutingTilingCase testCase;
        size_t idx = 0;
        testCase.caseName = Trim(items[idx++]);
        testCase.enable = ParseBool(items[idx++]);
        if (!testCase.enable) {
            continue;
        }
        testCase.prefix = Trim(items[idx++]);
        testCase.aicNum = Trim(items[idx++]);
        testCase.aivNum = Trim(items[idx++]);
        testCase.ubSize = Trim(items[idx++]);
        testCase.l1Size = Trim(items[idx++]);
        testCase.l2Size = Trim(items[idx++]);
        testCase.l0CSize = Trim(items[idx++]);
        testCase.l0ASize = Trim(items[idx++]);
        testCase.l0BSize = Trim(items[idx++]);
        testCase.btSize = Trim(items[idx++]);
        testCase.cubeFreq = Trim(items[idx++]);
        testCase.socVersion = Trim(items[idx++]);
        testCase.supportL0c2out = Trim(items[idx++]);
        testCase.supportL12BtBf16 = Trim(items[idx++]);
        testCase.npuArch = Trim(items[idx++]);
        testCase.xShape = Trim(items[idx++]);
        testCase.wShape = Trim(items[idx++]);
        testCase.scaleShape = Trim(items[idx++]);
        testCase.biasShape = Trim(items[idx++]);
        testCase.pertokenScaleShape = Trim(items[idx++]);
        testCase.groupListShape = Trim(items[idx++]);
        testCase.sharedInputShape = Trim(items[idx++]);
        testCase.logitShape = Trim(items[idx++]);
        testCase.rowindexShape = Trim(items[idx++]);
        testCase.yShape = Trim(items[idx++]);
        testCase.xDtype = Trim(items[idx++]);
        testCase.wDtype = Trim(items[idx++]);
        testCase.scaleDtype = Trim(items[idx++]);
        testCase.biasDtype = Trim(items[idx++]);
        testCase.pertokenScaleDtype = Trim(items[idx++]);
        testCase.groupListDtype = Trim(items[idx++]);
        testCase.sharedInputDtype = Trim(items[idx++]);
        testCase.logitDtype = Trim(items[idx++]);
        testCase.rowindexDtype = Trim(items[idx++]);
        testCase.yDtype = Trim(items[idx++]);
        testCase.wFormat = Trim(items[idx++]);
        testCase.dtypeAttr = Trim(items[idx++]);
        testCase.sharedInputWeight = Trim(items[idx++]);
        testCase.sharedInputOffset = Trim(items[idx++]);
        testCase.transposeX = Trim(items[idx++]);
        testCase.transposeW = Trim(items[idx++]);
        testCase.outputBs = Trim(items[idx++]);
        testCase.groupListType = Trim(items[idx++]);
        testCase.tuningConfig = Trim(items[idx++]);
        testCase.expectResult = Trim(items[idx++]);
        testCase.expectTilingKey = Trim(items[idx++]);
        cases.emplace_back(testCase);
    }

    EXPECT_FALSE(cases.empty()) << "No cases loaded from " << csvPath;
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingTilingCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

}  // namespace

namespace GroupedMatmulFinalizeRoutingTilingUT {

class TestGroupedMatmulFinalizeRoutingTiling : public testing::TestWithParam<GroupedMatmulFinalizeRoutingTilingCase> {
};

TEST_P(TestGroupedMatmulFinalizeRoutingTiling, generalTest)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    GROUPED_MATMUL_FINALIZE_ROUTING_TILING, TestGroupedMatmulFinalizeRoutingTiling,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_grouped_matmul_finalize_routing_tiling.csv"))), MakeParamName);

}  // namespace GroupedMatmulFinalizeRoutingTilingUT
