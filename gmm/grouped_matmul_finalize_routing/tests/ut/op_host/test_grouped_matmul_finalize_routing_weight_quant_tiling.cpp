/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_finalize_routing_weight_quant_tiling.cpp
 * \brief CSV-driven unit tests for MX-A8W4 weight quantization tiling.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace optiling
{
extern void DisablePatternCache();
extern void EnablePatternCache();
}  // namespace optiling

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 18U;

optiling::GroupedMatmulFinalizeRoutingCompileInfo kDefaultCompileInfo = {
    24,
    48,
    196608,
    524288,
    33554432,
    131072,
    65536,
    65536,
    131072,
    0,
    platform_ascendc::SocVersion::ASCEND950,
    false,
    true,
    NpuArch::DAV_3510,
};

ge::DataType ParseDtype(const string &dtype)
{
    const string trimmed = Trim(dtype);
    if (trimmed == "NULL" || trimmed == "EMPTY") {
        return ge::DT_UNDEFINED;
    }
    if (trimmed == "DT_FLOAT8_E4M3FN") {
        return ge::DT_FLOAT8_E4M3FN;
    }
    if (trimmed == "DT_FLOAT4_E2M1") {
        return ge::DT_FLOAT4_E2M1;
    }
    if (trimmed == "DT_FLOAT8_E8M0") {
        return ge::DT_FLOAT8_E8M0;
    }
    return ops::ut::ParseGeDtype(trimmed);
}

bool HasFlag(const string &testName, const string &flag)
{
    return testName.find(flag) != string::npos;
}

struct WeightQuantTilingCase {
    string testName;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t e = 0;
    int64_t bs = 0;
    string xDtype;
    string wDtype;
    string scaleDtype;
    string biasDtype;
    string perTokenScaleDtype;
    bool transposeW = false;
    string outputBs;
    int64_t groupListType = 0;
    int64_t expectTilingKey = 0;
    string expectResult;
    bool verifyBlockNum = false;
    string description;

    void Run() const
    {
        gert::StorageShape xShape = {{m, k}, {m, k}};
        gert::StorageShape wShape = {{e, n, k}, {e, (k + 31) / 32, (n + 15) / 16, 16, 32}};
        gert::StorageShape scaleShape = {{e, n, (k + 63) / 64, 2}, {e, n, (k + 63) / 64, 2}};
        gert::StorageShape biasShape = {{e, n}, {e, n}};
        gert::StorageShape perTokenScaleShape = {{m, (k + 63) / 64, 2}, {m, (k + 63) / 64, 2}};
        gert::StorageShape groupListShape = {{e}, {e}};
        gert::StorageShape sharedInputShape = {{bs, n}, {bs, n}};
        gert::StorageShape logitsShape = {{m}, {m}};
        gert::StorageShape rowIndexShape = {{m}, {m}};
        gert::StorageShape yShape = {{m, n}, {m, n}};

        if (HasFlag(testName, "NullScale")) {
            scaleShape = {{}, {}};
        }
        if (HasFlag(testName, "NullPertokenScale")) {
            perTokenScaleShape = {{}, {}};
        }
        if (HasFlag(testName, "NullRowIndex")) {
            rowIndexShape = {{}, {}};
        }
        if (HasFlag(testName, "NullGroupList")) {
            groupListShape = {{}, {}};
        }
        if (HasFlag(testName, "EmptyBias")) {
            biasShape = {{}, {}};
        }
        if (HasFlag(testName, "NullSharedInput")) {
            sharedInputShape = {{}, {}};
        }
        if (HasFlag(testName, "WrongXShape")) {
            xShape = {{m, k, 1}, {m, k, 1}};
        }
        if (HasFlag(testName, "WrongWShape")) {
            wShape = {{e, n}, {e, n}};
        }
        if (HasFlag(testName, "EMismatch")) {
            scaleShape = {{e + 1, n, (k + 63) / 64, 2}, {e + 1, n, (k + 63) / 64, 2}};
        }
        if (HasFlag(testName, "KMismatch")) {
            scaleShape = {{e, n, (k + 63) / 64 + 1, 2}, {e, n, (k + 63) / 64 + 1, 2}};
        }

        vector<gert::TilingContextPara::OpAttr> attrs = {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeW)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(Trim(outputBs) == "NULL" ? 0 : stoll(outputBs))},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        };

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {xShape, ParseDtype(xDtype), ge::FORMAT_ND},
                {wShape, ParseDtype(wDtype), ge::FORMAT_FRACTAL_NZ},
                {scaleShape, ParseDtype(scaleDtype), ge::FORMAT_ND},
                {biasShape, ParseDtype(biasDtype), ge::FORMAT_ND},
                {perTokenScaleShape, ParseDtype(perTokenScaleDtype), ge::FORMAT_ND},
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
                {sharedInputShape, ge::DT_BF16, ge::FORMAT_ND},
                {logitsShape, ge::DT_FLOAT, ge::FORMAT_ND},
                {rowIndexShape, ge::DT_INT64, ge::FORMAT_ND},
            },
            {
                {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            attrs,
            &kDefaultCompileInfo,
            "Ascend950");

        if (Trim(expectResult) != "SUCCESS") {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
            return;
        }

        TilingInfo tilingInfo;
        const bool result = ExecuteTiling(tilingContextPara, tilingInfo);
        EXPECT_TRUE(result) << description;
        EXPECT_EQ(static_cast<int64_t>(tilingInfo.tilingKey), expectTilingKey) << description;
        if (verifyBlockNum) {
            EXPECT_EQ(tilingInfo.blockNum, kDefaultCompileInfo.aicNum) << description;
        }
        VerifyDerivedFields(tilingInfo);
    }

    void VerifyDerivedFields(const TilingInfo &tilingInfo) const
    {
        ASSERT_NE(tilingInfo.tilingData, nullptr) << description;
        ASSERT_GT(tilingInfo.tilingDataSize, 0U) << description;

        const auto *weightQuantTiling =
            reinterpret_cast<const GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData *>(
                tilingInfo.tilingData.get());
        const int64_t outputBsValue = Trim(outputBs) == "NULL" ? (e > 0 ? m / e : 0) : stoll(outputBs);
        const uint64_t expectedSharedInputLen =
            HasFlag(testName, "NullSharedInput") ? 0U : static_cast<uint64_t>(bs);
        const uint64_t expectedInitSize =
            static_cast<uint64_t>(outputBsValue - static_cast<int64_t>(expectedSharedInputLen)) *
            static_cast<uint64_t>(n);

        EXPECT_EQ(weightQuantTiling->initSize, expectedInitSize) << testName << ": initSize mismatch";
        EXPECT_EQ(weightQuantTiling->nSize, static_cast<uint64_t>(n)) << testName << ": nSize mismatch";
        EXPECT_EQ(weightQuantTiling->sharedInputLen, expectedSharedInputLen)
            << testName << ": sharedInputLen mismatch";
    }
};

vector<WeightQuantTilingCase> LoadCases()
{
    vector<WeightQuantTilingCase> cases;
    const string csvPath = ops::ut::ResolveCsvPath("weight_quant_test_cases.csv",
                                                   "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/test_data",
                                                   __FILE__);
    ifstream file(csvPath, ios::in);
    EXPECT_TRUE(file.is_open()) << "cannot open case file " << csvPath;
    if (!file.is_open()) {
        return cases;
    }

    string line;
    size_t lineNo = 0U;
    while (getline(file, line)) {
        ++lineNo;
        const string trimmedLine = Trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }

        vector<string> fields;
        SplitStr2Vec(trimmedLine, ",", fields);
        if (fields.empty() || fields[0] == "test_name") {
            continue;
        }
        if (fields.size() != kCsvColumnCount) {
            ADD_FAILURE() << "Bad csv row column count in " << csvPath << ": " << trimmedLine;
            continue;
        }

        const string caseName = Trim(fields[0]);
        try {
            size_t idx = 0U;
            WeightQuantTilingCase tc;
            tc.testName = Trim(fields[idx++]);
            tc.m = stoll(Trim(fields[idx++]));
            tc.k = stoll(Trim(fields[idx++]));
            tc.n = stoll(Trim(fields[idx++]));
            tc.e = stoll(Trim(fields[idx++]));
            tc.bs = stoll(Trim(fields[idx++]));
            tc.xDtype = Trim(fields[idx++]);
            tc.wDtype = Trim(fields[idx++]);
            tc.scaleDtype = Trim(fields[idx++]);
            tc.biasDtype = Trim(fields[idx++]);
            tc.perTokenScaleDtype = Trim(fields[idx++]);
            tc.transposeW = ParseBool(Trim(fields[idx++]));
            tc.outputBs = Trim(fields[idx++]);
            tc.groupListType = stoll(Trim(fields[idx++]));
            tc.expectTilingKey = stoll(Trim(fields[idx++]));
            tc.expectResult = Trim(fields[idx++]);
            tc.verifyBlockNum = ParseBool(Trim(fields[idx++]));
            tc.description = Trim(fields[idx++]);
            cases.emplace_back(tc);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }

    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

const vector<WeightQuantTilingCase> &GetCases()
{
    static const vector<WeightQuantTilingCase> cases = LoadCases();
    return cases;
}

string MakeParamName(const testing::TestParamInfo<WeightQuantTilingCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.testName);
}

}  // namespace

namespace GroupedMatmulFinalizeRoutingWeightQuantTilingUT {

class TestGroupedMatmulFinalizeRoutingWeightQuantTiling
    : public testing::TestWithParam<WeightQuantTilingCase> {
protected:
    static void SetUpTestCase()
    {
        cout << "GroupedMatmulFinalizeRoutingWeightQuantTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "GroupedMatmulFinalizeRoutingWeightQuantTiling TearDown" << endl;
    }
};

TEST_P(TestGroupedMatmulFinalizeRoutingWeightQuantTiling, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_WEIGHT_QUANT_TILING_CSV, TestGroupedMatmulFinalizeRoutingWeightQuantTiling,
                         testing::ValuesIn(GetCases()), MakeParamName);

}  // namespace GroupedMatmulFinalizeRoutingWeightQuantTilingUT
