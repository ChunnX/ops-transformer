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
 * \file test_grouped_matmul_swiglu_quant_v2_infershape.cpp
 * \brief CSV-driven unit tests for grouped_matmul_swiglu_quant_v2 infershape.
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
#include "infer_datatype_context_faker.h"
#include "ut_ge_parse_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#define private public
#include "platform/platform_info.h"

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

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

ge::DataType ParseDtype(const string &dtype)
{
    return ops::ut::ParseGeDtype(dtype);
}

ge::Format ParseFormat(const string &format)
{
    return ops::ut::ParseGeFormat(format);
}

vector<int64_t> ParseDims(const string &value)
{
    return ops::ut::ParseDims(value, {}, true);
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

struct GroupedMatmulSwigluQuantV2InfershapeCase {
    bool ShouldRunInferShape() const
    {
        return caseType.find("dtype_only") == string::npos;
    }

    bool ShouldRunInferDtype() const
    {
        return caseType.find("dtype_only") != string::npos;
    }

    vector<gert::InfershapeContextPara::OpAttr> BuildShapeAttrs() const
    {
        return {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dequantMode)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(dequantDtype)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantDtype)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
        };
    }

    vector<pair<string, Ops::Transformer::AnyValue>> BuildDtypeAttrs() const
    {
        return {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dequantMode)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(dequantDtype)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantDtype)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
        };
    }

    void RunInferShape() const
    {
        SetupPlatformForCase(socVersion);
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmulSwigluQuantV2",
            {
                {MakeStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {MakeStorageShape(ParseDims(xScaleShape)), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat)},
                {MakeStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ParseFormat(groupListFormat)},
                {MakeStorageShape(ParseDims(weightShape)), ParseDtype(weightDtype), ParseFormat(weightFormat)},
                {MakeStorageShape(ParseDims(weightScaleShape)), ParseDtype(weightScaleDtype), ParseFormat(weightScaleFormat)},
                {MakeStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ParseFormat(biasFormat)},
                {MakeStorageShape(ParseDims(offsetShape)), ParseDtype(offsetDtype), ParseFormat(offsetFormat)},
                {MakeStorageShape(ParseDims(antiquantOffsetShape)), ParseDtype(antiquantOffsetDtype),
                 ParseFormat(antiquantOffsetFormat)},
            },
            {
                {MakeStorageShape(ParseDims("NONE")), ParseDtype(yDtype), ParseFormat(yFormat)},
                {MakeStorageShape(ParseDims("NONE")), ParseDtype(yScaleDtype), ParseFormat(yScaleFormat)},
            },
            BuildShapeAttrs());

        vector<vector<int64_t>> expectShape;
        expectShape.push_back(ParseDims(expectYShape));
        expectShape.push_back(ParseDims(expectYScaleShape));
        ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
    }

    void RunInferDtype() const
    {
        SetupPlatformForCase(socVersion);
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(spaceRegistry, nullptr);
        auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmulSwigluQuantV2");
        ASSERT_NE(opImpl, nullptr);
        ASSERT_NE(opImpl->infer_datatype, nullptr);

        ge::DataType xDataType = ParseDtype(xDtype);
        ge::DataType xScaleDataType = ParseDtype(xScaleDtype);
        ge::DataType groupListDataType = ParseDtype(groupListDtype);
        ge::DataType weightDataType = ParseDtype(weightDtype);
        ge::DataType weightScaleDataType = ParseDtype(weightScaleDtype);
        ge::DataType biasDataType = ParseDtype(biasDtype);
        ge::DataType offsetDataType = ParseDtype(offsetDtype);
        ge::DataType antiquantOffsetDataType = ParseDtype(antiquantOffsetDtype);
        vector<void *> inputDataTypes = {
            &xDataType,
            &xScaleDataType,
            &groupListDataType,
            &weightDataType,
            &weightScaleDataType,
            &biasDataType,
            &offsetDataType,
            &antiquantOffsetDataType,
        };

        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .SetOpType("GroupedMatmulSwigluQuantV2")
                                 .NodeIoNum(8, 2)
                                 .NodeOutputTd(0, ParseFormat(yFormat), ParseFormat(yFormat))
                                 .NodeOutputTd(1, ParseFormat(yScaleFormat), ParseFormat(yScaleFormat))
                                 .InputDataTypes(inputDataTypes)
                                 .NodeAttrs(BuildDtypeAttrs())
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        auto inferDtypeRet = opImpl->infer_datatype(context);
        ASSERT_EQ(inferDtypeRet, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
        if (inferDtypeRet != ge::GRAPH_SUCCESS) {
            return;
        }
        EXPECT_EQ(context->GetOutputDataType(0), ParseDtype(yDtype));
        EXPECT_EQ(context->GetOutputDataType(1), ParseDtype(yScaleDtype));
    }

    void Run() const
    {
        if (ShouldRunInferShape()) {
            RunInferShape();
        }
        if (ShouldRunInferDtype()) {
            RunInferDtype();
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    string caseType;
    bool expectSuccess = false;
    string expectYShape;
    string expectYScaleShape;

    string xShape;
    string xScaleShape;
    string groupListShape;
    string weightShape;
    string weightScaleShape;
    string biasShape;
    string offsetShape;
    string antiquantOffsetShape;

    string xDtype;
    string xScaleDtype;
    string groupListDtype;
    string weightDtype;
    string weightScaleDtype;
    string biasDtype;
    string offsetDtype;
    string antiquantOffsetDtype;
    string yDtype;
    string yScaleDtype;

    string xFormat;
    string xScaleFormat;
    string groupListFormat;
    string weightFormat;
    string weightScaleFormat;
    string biasFormat;
    string offsetFormat;
    string antiquantOffsetFormat;
    string yFormat;
    string yScaleFormat;

    int64_t dequantMode = 0;
    float dequantDtype = 0.0F;
    int64_t quantMode = 0;
    int64_t quantDtype = 0;
    bool transposeWeight = false;
    int64_t groupListType = 0;
};

vector<GroupedMatmulSwigluQuantV2InfershapeCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulSwigluQuantV2InfershapeCase> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath +
                   "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host/"
                   "test_grouped_matmul_swiglu_quant_v2_infershape.csv");
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 42U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulSwigluQuantV2InfershapeCase tc;
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
        tc.expectYScaleShape = Trim(items[idx++]);

        tc.xShape = Trim(items[idx++]);
        tc.xScaleShape = Trim(items[idx++]);
        tc.groupListShape = Trim(items[idx++]);
        tc.weightShape = Trim(items[idx++]);
        tc.weightScaleShape = Trim(items[idx++]);
        tc.biasShape = Trim(items[idx++]);
        tc.offsetShape = Trim(items[idx++]);
        tc.antiquantOffsetShape = Trim(items[idx++]);

        tc.xDtype = Trim(items[idx++]);
        tc.xScaleDtype = Trim(items[idx++]);
        tc.groupListDtype = Trim(items[idx++]);
        tc.weightDtype = Trim(items[idx++]);
        tc.weightScaleDtype = Trim(items[idx++]);
        tc.biasDtype = Trim(items[idx++]);
        tc.offsetDtype = Trim(items[idx++]);
        tc.antiquantOffsetDtype = Trim(items[idx++]);
        tc.yDtype = Trim(items[idx++]);
        tc.yScaleDtype = Trim(items[idx++]);

        tc.xFormat = Trim(items[idx++]);
        tc.xScaleFormat = Trim(items[idx++]);
        tc.groupListFormat = Trim(items[idx++]);
        tc.weightFormat = Trim(items[idx++]);
        tc.weightScaleFormat = Trim(items[idx++]);
        tc.biasFormat = Trim(items[idx++]);
        tc.offsetFormat = Trim(items[idx++]);
        tc.antiquantOffsetFormat = Trim(items[idx++]);
        tc.yFormat = Trim(items[idx++]);
        tc.yScaleFormat = Trim(items[idx++]);

        tc.dequantMode = stoll(Trim(items[idx++]));
        tc.dequantDtype = stof(Trim(items[idx++]));
        tc.quantMode = stoll(Trim(items[idx++]));
        tc.quantDtype = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        cases.push_back(tc);
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulSwigluQuantV2InfershapeCase> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}
} // namespace

namespace GroupedMatmulSwigluQuantV2InfershapeUT {
const vector<GroupedMatmulSwigluQuantV2InfershapeCase> &GetSwigluQuantV2InfershapeCsvCases()
{
    static const vector<GroupedMatmulSwigluQuantV2InfershapeCase> cases = [] {
        vector<GroupedMatmulSwigluQuantV2InfershapeCase> merged = LoadCases("Ascend950");
        vector<GroupedMatmulSwigluQuantV2InfershapeCase> ascend910b = LoadCases("Ascend910B");
        merged.insert(merged.end(), ascend910b.begin(), ascend910b.end());
        return merged;
    }();
    return cases;
}

class TestGroupedMatmulSwigluQuantV2Infershape
    : public testing::TestWithParam<GroupedMatmulSwigluQuantV2InfershapeCase> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedMatmulSwigluQuantV2Infershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V2_INFERSHAPE_CSV, TestGroupedMatmulSwigluQuantV2Infershape,
                         testing::ValuesIn(GetSwigluQuantV2InfershapeCsvCases()), MakeParamName);
} // namespace GroupedMatmulSwigluQuantV2InfershapeUT
