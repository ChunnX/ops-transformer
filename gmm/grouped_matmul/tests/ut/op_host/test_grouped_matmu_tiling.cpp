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
 * \file test_grouped_matmul.cpp
 * \brief
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

#include "../../../op_host/op_tiling/grouped_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_grouped_matmul_utils.h"

using namespace std;
using namespace ge;

class GroupedMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulTiling TearDown" << std::endl;
    }
};


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
    transform(lower.begin(), lower.end(), lower.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
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
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
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

gert::StorageShape MakeStorageShape(const vector<int64_t> &originDims, const vector<int64_t> &storageDims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildShape(originDims);
    shape.MutableStorageShape() = BuildShape(storageDims);
    return shape;
}

struct GroupedMatmulAscend950TensorDesc {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
};

enum class GroupedMatmulAscend950ExpectType {
    TILING_KEY,
    GRAPH_FAILED,
};

struct GroupedMatmulAscend950Case {
    string caseName;
    bool enable = true;
    string prefix;
    GroupedMatmulAscend950ExpectType expectType = GroupedMatmulAscend950ExpectType::TILING_KEY;
    int64_t expectTilingKey = 0;
    uint32_t aicNum = 24;
    uint32_t aivNum = 48;
    uint64_t ubSize = 196608;
    uint64_t l1Size = 524288;
    uint64_t l2Size = 196608;
    uint64_t l0CSize = 131072;
    uint64_t l0ASize = 65536;
    uint64_t l0BSize = 65536;
    int64_t splitItem = 2;
    int64_t dtypeAttr = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    int64_t tuningConfig = 0;
    vector<GroupedMatmulAscend950TensorDesc> inputs;
    GroupedMatmulAscend950TensorDesc output;

    void Run() const
    {
        optiling::GMMCompileInfo compileInfo = {
            aicNum,
            aivNum,
            ubSize,
            l1Size,
            l2Size,
            l0CSize,
            l0ASize,
            l0BSize,
            platform_ascendc::SocVersion::ASCEND950,
            NpuArch::DAV_3510,
        };
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul", BuildTensorDescs(inputs), {BuildTensorDesc(output)},
            {
                {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtypeAttr)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
                {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({tuningConfig})},
            },
            &compileInfo);

        if (expectType == GroupedMatmulAscend950ExpectType::GRAPH_FAILED) {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
            return;
        }

        TilingInfo tilingInfo;
        ExecuteTiling(tilingContextPara, tilingInfo);
        EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    }

private:
    static gert::TilingContextPara::TensorDescription BuildTensorDesc(const GroupedMatmulAscend950TensorDesc &tensor)
    {
        return {MakeStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype, tensor.format};
    }

    static vector<gert::TilingContextPara::TensorDescription> BuildTensorDescs(
        const vector<GroupedMatmulAscend950TensorDesc> &tensors)
    {
        vector<gert::TilingContextPara::TensorDescription> descs;
        descs.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            descs.emplace_back(BuildTensorDesc(tensor));
        }
        return descs;
    }
};

vector<GroupedMatmulAscend950Case> LoadGroupedMatmulAscend950Cases()
{
    vector<GroupedMatmulAscend950Case> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath + "gmm/grouped_matmul/tests/ut/op_host/test_grouped_matmu_tiling.csv");
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
        if (items.empty() || items[0] == "caseName" || items.size() < 61U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulAscend950Case tc;
        tc.caseName = Trim(items[idx++]);
        tc.enable = ParseBool(Trim(items[idx++]));
        if (!tc.enable) {
            continue;
        }
        tc.prefix = Trim(items[idx++]);
        tc.expectType = ParseBool(Trim(items[idx++])) ? GroupedMatmulAscend950ExpectType::TILING_KEY
                                                      : GroupedMatmulAscend950ExpectType::GRAPH_FAILED;
        tc.expectTilingKey = stoll(Trim(items[idx++]));
        tc.aicNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
        tc.aivNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
        tc.ubSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l1Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l2Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0CSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0ASize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0BSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.splitItem = stoll(Trim(items[idx++]));
        tc.dtypeAttr = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.transposeX = ParseBool(Trim(items[idx++]));
        tc.groupType = stoll(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        tc.actType = stoll(Trim(items[idx++]));
        tc.tuningConfig = stoll(Trim(items[idx++]));

        for (int tensorIdx = 0; tensorIdx < 9; ++tensorIdx) {
            tc.inputs.push_back({ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                                 ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))});
        }
        tc.output = {ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                     ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))};
        cases.push_back(tc);
    }
    return cases;
}

string MakeGroupedMatmulAscend950CaseName(const testing::TestParamInfo<GroupedMatmulAscend950Case> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

} // namespace

class GroupedMatmulAscend950Tiling : public testing::TestWithParam<GroupedMatmulAscend950Case> {
};

TEST_P(GroupedMatmulAscend950Tiling, generalTest)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_MATMUL_ASCEND950, GroupedMatmulAscend950Tiling,
                         testing::ValuesIn(LoadGroupedMatmulAscend950Cases()),
                         MakeGroupedMatmulAscend950CaseName);

namespace GroupedMatmulAscend910BUT {

struct GroupedMatmulAscend910BCase {
    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    int64_t expectTilingKey = 0;
    uint32_t aicNum = 24;
    uint32_t aivNum = 48;
    uint64_t ubSize = 196608;
    uint64_t l1Size = 524288;
    uint64_t l2Size = 196608;
    uint64_t l0CSize = 131072;
    uint64_t l0ASize = 65536;
    uint64_t l0BSize = 65536;
    int64_t splitItem = 3;
    int64_t dtypeAttr = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    string tuningConfig;
    vector<GroupedMatmulAscend950TensorDesc> inputs;
    GroupedMatmulAscend950TensorDesc output;

    void Run() const {
        optiling::GMMCompileInfo compileInfo = {
            aicNum,
            aivNum,
            ubSize,
            l1Size,
            l2Size,
            l0CSize,
            l0ASize,
            l0BSize,
            platform_ascendc::SocVersion::ASCEND910B,
            NpuArch::DAV_2201,
        };

        vector<int64_t> tuningConfigVec;
        if (!tuningConfig.empty() && tuningConfig != "NONE") {
            vector<string> tokens;
            SplitStr2Vec(tuningConfig, ":", tokens);
            for (const auto &token : tokens) {
                tuningConfigVec.push_back(stoll(Trim(token)));
            }
        } else {
            tuningConfigVec = {0};
        }

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul", BuildTensorDescs(inputs), {BuildTensorDesc(output)},
            {
                {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtypeAttr)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
                {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tuningConfigVec)},
            },
            &compileInfo);

        if (!expectSuccess) {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
            return;
        }

        TilingInfo tilingInfo;
        ExecuteTiling(tilingContextPara, tilingInfo);
        EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    }

private:
    static gert::TilingContextPara::TensorDescription BuildTensorDesc(const GroupedMatmulAscend950TensorDesc &tensor) {
        return {MakeStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype, tensor.format};
    }

    static vector<gert::TilingContextPara::TensorDescription> BuildTensorDescs(
        const vector<GroupedMatmulAscend950TensorDesc> &tensors) {
        vector<gert::TilingContextPara::TensorDescription> descs;
        descs.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            descs.emplace_back(BuildTensorDesc(tensor));
        }
        return descs;
    }
};

vector<GroupedMatmulAscend910BCase> LoadGroupedMatmulAscend910BCases() {
    vector<GroupedMatmulAscend910BCase> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath + "gmm/grouped_matmul/tests/ut/op_host/test_grouped_matmu_tiling.csv");
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
        if (items.empty() || items[0] == "caseName" || items.size() < 61U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulAscend910BCase tc;
        tc.caseName = Trim(items[idx++]);
        tc.enable = ParseBool(Trim(items[idx++]));
        if (!tc.enable) {
            continue;
        }
        tc.prefix = Trim(items[idx++]);
        tc.expectSuccess = ParseBool(Trim(items[idx++]));
        tc.expectTilingKey = stoll(Trim(items[idx++]));
        tc.aicNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
        tc.aivNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
        tc.ubSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l1Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l2Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0CSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0ASize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0BSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.splitItem = stoll(Trim(items[idx++]));
        tc.dtypeAttr = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.transposeX = ParseBool(Trim(items[idx++]));
        tc.groupType = stoll(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        tc.actType = stoll(Trim(items[idx++]));
        tc.tuningConfig = Trim(items[idx++]);

        for (int tensorIdx = 0; tensorIdx < 9; ++tensorIdx) {
            tc.inputs.push_back({ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                                 ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))});
        }
        tc.output = {ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                     ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))};
        cases.push_back(tc);
    }
    return cases;
}

string MakeGroupedMatmulAscend910BCaseName(const testing::TestParamInfo<GroupedMatmulAscend910BCase> &info) {
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

const vector<GroupedMatmulAscend910BCase> &GetGroupedMatmulAscend910BCsvCases() {
    static const vector<GroupedMatmulAscend910BCase> cases = LoadGroupedMatmulAscend910BCases();
    return cases;
}

class TestGroupedMatmulAscend910BTiling : public testing::TestWithParam<GroupedMatmulAscend910BCase> {
protected:
    static void SetUpTestCase() {
        std::cout << "TestGroupedMatmulAscend910BTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "TestGroupedMatmulAscend910BTiling TearDown" << std::endl;
    }
};

TEST_P(TestGroupedMatmulAscend910BTiling, csvDrivenCase) {
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_ASCEND910B_TILING_CSV, TestGroupedMatmulAscend910BTiling,
                         testing::ValuesIn(GetGroupedMatmulAscend910BCsvCases()), MakeGroupedMatmulAscend910BCaseName);

} // namespace GroupedMatmulAscend910BUT
