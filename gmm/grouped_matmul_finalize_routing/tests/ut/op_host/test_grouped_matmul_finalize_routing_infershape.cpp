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
        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {ops::ut::MakeGertStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ParseFormat(xFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(wShape)), ParseDtype(wDtype), ParseFormat(wFormat)},
                {ops::ut::MakeGertStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype), ParseFormat(scaleFormat)},
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
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedMatmulFinalizeRoutingInfershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_INFERSHAPE_CSV, TestGroupedMatmulFinalizeRoutingInfershape,
                         testing::ValuesIn(GetCases()), MakeParamName);
} // namespace GroupedMatmulFinalizeRoutingInfershapeUT
