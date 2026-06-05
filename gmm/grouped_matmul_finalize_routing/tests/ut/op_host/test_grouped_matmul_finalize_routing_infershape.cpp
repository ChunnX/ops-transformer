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

#include <fstream>
#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "gmm_csv_ge_parse_utils.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

using namespace std;

namespace {
using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

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

struct GroupedMatmulFinalizeRoutingInfershapeCase {
    void Run() const
    {
        RunInferShape();
    }

    void RunInferShape() const
    {
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {ops::ut::MakeGertStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(wShape)), ParseDtype(wDtype), ParseFormat(wFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype),
                 ParseFormat(scaleFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ParseFormat(biasFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(pertokenScaleShape)), ParseDtype(pertokenScaleDtype),
                 ParseFormat(pertokenScaleFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype),
                 ParseFormat(groupListFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(sharedInputShape)), ParseDtype(sharedInputDtype),
                 ParseFormat(sharedInputFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(logitShape)), ParseDtype(logitDtype),
                 ParseFormat(logitFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(rowIndexShape)), ParseDtype(rowIndexDtype),
                 ParseFormat(rowIndexFormat)},
            },
            {
                {ops::ut::MakeGertStorageShape(ParseDims("NONE")), ParseDtype(yDtype), ParseFormat(yFormat)},
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

    void RunInferDtype() const
    {
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(spaceRegistry, nullptr);
        auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmulFinalizeRouting");
        ASSERT_NE(opImpl, nullptr);
        ASSERT_NE(opImpl->infer_datatype, nullptr);

        ge::DataType xDtypeVal = ParseDtype(xDtype);
        ge::DataType wDtypeVal = ParseDtype(wDtype);
        ge::DataType scaleDtypeVal = Trim(scaleShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(scaleDtype);
        ge::DataType biasDtypeVal = Trim(biasShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(biasDtype);
        ge::DataType pertokenScaleDtypeVal =
            Trim(pertokenScaleShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(pertokenScaleDtype);
        ge::DataType groupListDtypeVal = Trim(groupListShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(groupListDtype);
        ge::DataType sharedInputDtypeVal =
            Trim(sharedInputShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(sharedInputDtype);
        ge::DataType logitDtypeVal = Trim(logitShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(logitDtype);
        ge::DataType rowIndexDtypeVal = Trim(rowIndexShape) == "NONE" ? ge::DT_UNDEFINED : ParseDtype(rowIndexDtype);

        vector<void *> inputDataTypes = {
            &xDtypeVal,         &wDtypeVal,           &scaleDtypeVal, &biasDtypeVal,     &pertokenScaleDtypeVal,
            &groupListDtypeVal, &sharedInputDtypeVal, &logitDtypeVal, &rowIndexDtypeVal,
        };

        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .SetOpType("GroupedMatmulFinalizeRouting")
                                 .NodeIoNum(9, 1)
                                 .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes(inputDataTypes)
                                 .Attr("dtype", int64_t(dtype))
                                 .Attr("shared_input_weight", float(sharedInputWeight))
                                 .Attr("shared_input_offset", int64_t(sharedInputOffset))
                                 .Attr("transpose_x", bool(transposeX))
                                 .Attr("transpose_w", bool(transposeW))
                                 .Attr("output_bs", int64_t(outputBs))
                                 .Attr("group_list_type", int64_t(groupListType))
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        auto inferDtypeRet = opImpl->infer_datatype(context);
        EXPECT_EQ(inferDtypeRet, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED)
            << "InferDtype failed for case=" << caseName;
        if (inferDtypeRet == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
        }
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
    const string csvPath = ops::ut::ResolveCsvPath("test_grouped_matmul_finalize_routing_infershape.csv",
                                                   "gmm/grouped_matmul_finalize_routing/tests/ut/op_host", __FILE__);
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

const vector<GroupedMatmulFinalizeRoutingInfershapeCase> &GetCases()
{
    static const auto cases = LoadCases();
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingInfershapeCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedMatmulFinalizeRoutingInfershapeUT {
class TestGroupedMatmulFinalizeRoutingInfershape
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingInfershapeCase> {
protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
    }
};

TEST_P(TestGroupedMatmulFinalizeRoutingInfershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_INFERSHAPE_CSV, TestGroupedMatmulFinalizeRoutingInfershape,
                         testing::ValuesIn(GetCases()), MakeParamName);

class TestGroupedMatmulFinalizeRoutingInferDtype : public testing::Test {
protected:
};

ge::graphStatus RunInferDtypeWithDtypes(vector<void *> &inputDataTypes,
                                        vector<pair<string, Ops::Transformer::AnyValue>> &attrs)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmulFinalizeRouting");
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("GroupedMatmulFinalizeRouting")
                             .NodeIoNum(9, 1)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes(inputDataTypes)
                             .NodeAttrs(attrs)
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    return opImpl->infer_datatype(context);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W8A8ValidDtypes)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT8, scaleD = ge::DT_FLOAT;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_UNDEFINED, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W8A8WithBf16Scale)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT8, scaleD = ge::DT_BF16;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_UNDEFINED, riD = ge::DT_INT32;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W4A8ValidDtypes)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT4, scaleD = ge::DT_INT64;
    ge::DataType biasD = ge::DT_FLOAT, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, MXFP8ValidDtypes)
{
    ge::DataType xD = ge::DT_FLOAT8_E5M2, wD = ge::DT_FLOAT8_E5M2, scaleD = ge::DT_FLOAT8_E8M0;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_FLOAT8_E8M0;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, MXFP4ValidDtypes)
{
    ge::DataType xD = ge::DT_FLOAT4_E2M1, wD = ge::DT_FLOAT4_E2M1, scaleD = ge::DT_FLOAT8_E8M0;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_FLOAT8_E8M0;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W8A8WithSharedInput)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT8, scaleD = ge::DT_FLOAT;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_BF16;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W4A8WithSharedInput)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT4, scaleD = ge::DT_INT64;
    ge::DataType biasD = ge::DT_FLOAT, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_BF16;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, InvalidDtypeCompletelyWrong)
{
    ge::DataType xD = ge::DT_FLOAT16, wD = ge::DT_BF16, scaleD = ge::DT_FLOAT;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_UNDEFINED, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_FAILED);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W8A8WrongScaleDtype)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT8, scaleD = ge::DT_INT32;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_UNDEFINED, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_FAILED);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W4A8WrongBiasDtype)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT4, scaleD = ge::DT_INT64;
    ge::DataType biasD = ge::DT_INT32, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_FAILED);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W8A8WrongGroupListDtype)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT8, scaleD = ge::DT_FLOAT;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT32, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_UNDEFINED, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_FAILED);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W8A8WrongRowIndexDtype)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT8, scaleD = ge::DT_FLOAT;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_UNDEFINED, riD = ge::DT_FLOAT;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_FAILED);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, MXWrongScaleDtype)
{
    ge::DataType xD = ge::DT_FLOAT8_E5M2, wD = ge::DT_FLOAT8_E5M2, scaleD = ge::DT_FLOAT;
    ge::DataType biasD = ge::DT_UNDEFINED, ptD = ge::DT_FLOAT8_E8M0;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_FAILED);
}

TEST_F(TestGroupedMatmulFinalizeRoutingInferDtype, W4A8WithOffset)
{
    ge::DataType xD = ge::DT_INT8, wD = ge::DT_INT4, scaleD = ge::DT_INT64;
    ge::DataType biasD = ge::DT_FLOAT, ptD = ge::DT_UNDEFINED;
    ge::DataType glD = ge::DT_INT64, siD = ge::DT_UNDEFINED;
    ge::DataType logD = ge::DT_FLOAT, riD = ge::DT_INT64;
    vector<void *> inputDataTypes = {&xD, &wD, &scaleD, &biasD, &ptD, &glD, &siD, &logD, &riD};
    vector<pair<string, Ops::Transformer::AnyValue>> attrs = {
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    EXPECT_EQ(RunInferDtypeWithDtypes(inputDataTypes, attrs), ge::GRAPH_SUCCESS);
}
} // namespace GroupedMatmulFinalizeRoutingInfershapeUT
