/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*! \file test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.cpp
 *  \brief CSV-driven ACLNN UT for the MXFP8 FRACTAL_NZ SwiGLU mode=2 path.
 */

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
using ops::ut::BuildAclTensorDescFromSpec;
using ops::ut::BuildAclTensorListDesc;
using ops::ut::MakeAclTensorDesc;
using ops::ut::ParseAclDtype;
using ops::ut::ParseAclFormat;
using ops::ut::ParseAclnnStatus;
using ops::ut::ParseI64List;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kColumnCount = 31;

void SetupAscend950()
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
}

struct V3Case {
    string caseName;
    aclnnStatus expected;
    string xSpec;
    string xDtype;
    string xFormat;
    string weightSpecs;
    string weightDtype;
    string weightFormat;
    string weightScaleSpecs;
    string weightScaleDtype;
    string weightScaleFormat;
    string xScaleSpec;
    string xScaleDtype;
    string groupListSpec;
    string groupListDtype;
    string outputSpec;
    string outputScaleSpec;
    int64_t dequantMode;
    int64_t dequantDtype;
    int64_t quantMode;
    int64_t groupListType;
    int64_t swigluMode;
    double clampLimit;
    double gluAlpha;
    double gluBias;
    string roundMode;
    int64_t scaleAlg;
    double dstTypeMax;
    string optionalInput;
    string tuningConfig;
    string nullArg;

    aclnnStatus Run() const
    {
        SetupAscend950();
        auto xDesc = BuildAclTensorDescFromSpec(xSpec, xDtype, xFormat);
        auto weightDesc = BuildAclTensorListDesc(weightSpecs, weightDtype, weightFormat);
        auto weightScaleDesc = BuildAclTensorListDesc(weightScaleSpecs, weightScaleDtype, weightScaleFormat);
        auto xScaleDesc = BuildAclTensorDescFromSpec(xScaleSpec, xScaleDtype, "ND");
        auto groupListDesc = BuildAclTensorDescFromSpec(groupListSpec, groupListDtype, "ND");
        auto outputDesc = BuildAclTensorDescFromSpec(outputSpec, "FLOAT8_E4M3FN", "ND");
        auto outputScaleDesc = BuildAclTensorDescFromSpec(outputScaleSpec, "FLOAT8_E8M0", "ND");

        vector<int64_t> tuningValues = ParseI64List(tuningConfig, "|");
        aclIntArray *tuningArray = tuningValues.empty() ? nullptr : aclCreateIntArray(tuningValues.data(), tuningValues.size());
        const char *roundModePtr = roundMode == "NULL" ? nullptr : roundMode.c_str();
        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;

        auto invoke = [&](auto xArg, auto weightArg, auto weightScaleArg, auto assistArg, auto biasArg, auto xScaleArg,
                          auto smoothArg, auto groupArg, auto outputArg, auto outputScaleArg) {
            auto ut = OP_API_UT(
                aclnnGroupedMatmulSwigluQuantWeightNzV3,
                INPUT(xArg, weightArg, weightScaleArg, assistArg, biasArg, xScaleArg, smoothArg, groupArg,
                      dequantMode, dequantDtype, quantMode, groupListType, tuningArray, swigluMode, clampLimit,
                      gluAlpha, gluBias, roundModePtr, scaleAlg, dstTypeMax),
                OUTPUT(outputArg, outputScaleArg));
            return ut.TestGetWorkspaceSize(&workspaceSize);
        };

        if (nullArg == "x") {
            ret = invoke(nullptr, weightDesc, weightScaleDesc, nullptr, nullptr, xScaleDesc, nullptr, groupListDesc,
                         outputDesc, outputScaleDesc);
        } else if (nullArg == "weight") {
            ret = invoke(xDesc, nullptr, weightScaleDesc, nullptr, nullptr, xScaleDesc, nullptr, groupListDesc,
                         outputDesc, outputScaleDesc);
        } else if (nullArg == "weight_scale") {
            ret = invoke(xDesc, weightDesc, nullptr, nullptr, nullptr, xScaleDesc, nullptr, groupListDesc, outputDesc,
                         outputScaleDesc);
        } else if (nullArg == "x_scale") {
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, nullptr, nullptr, nullptr, groupListDesc,
                         outputDesc, outputScaleDesc);
        } else if (nullArg == "group_list") {
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, nullptr, xScaleDesc, nullptr, nullptr, outputDesc,
                         outputScaleDesc);
        } else if (nullArg == "output") {
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, nullptr, xScaleDesc, nullptr, groupListDesc,
                         nullptr, outputScaleDesc);
        } else if (nullArg == "output_scale") {
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, nullptr, xScaleDesc, nullptr, groupListDesc,
                         outputDesc, nullptr);
        } else if (optionalInput == "assist") {
            TensorListDesc assistDesc(vector<TensorDesc>{MakeAclTensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)});
            ret = invoke(xDesc, weightDesc, weightScaleDesc, assistDesc, nullptr, xScaleDesc, nullptr, groupListDesc,
                         outputDesc, outputScaleDesc);
        } else if (optionalInput == "bias") {
            auto biasDesc = MakeAclTensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, biasDesc, xScaleDesc, nullptr, groupListDesc,
                         outputDesc, outputScaleDesc);
        } else if (optionalInput == "smooth") {
            auto smoothDesc = MakeAclTensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, nullptr, xScaleDesc, smoothDesc, groupListDesc,
                         outputDesc, outputScaleDesc);
        } else {
            ret = invoke(xDesc, weightDesc, weightScaleDesc, nullptr, nullptr, xScaleDesc, nullptr, groupListDesc,
                         outputDesc, outputScaleDesc);
        }
        if (tuningArray != nullptr) {
            aclDestroyIntArray(tuningArray);
        }
        return ret;
    }
};

vector<V3Case> LoadCases(const string &csvPath)
{
    ifstream in(csvPath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvPath;
    vector<V3Case> cases;
    string line;
    size_t lineNo = 0;
    while (getline(in, line)) {
        ++lineNo;
        if (line.empty() || lineNo == 1) {
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != kColumnCount) {
            ADD_FAILURE() << "Invalid CSV column count at line " << lineNo << ": " << cols.size();
            continue;
        }
        try {
            V3Case item;
            size_t i = 0;
            item.caseName = Trim(cols[i++]);
            item.expected = ParseAclnnStatus(cols[i++]);
            item.xSpec = Trim(cols[i++]); item.xDtype = Trim(cols[i++]); item.xFormat = Trim(cols[i++]);
            item.weightSpecs = Trim(cols[i++]); item.weightDtype = Trim(cols[i++]); item.weightFormat = Trim(cols[i++]);
            item.weightScaleSpecs = Trim(cols[i++]); item.weightScaleDtype = Trim(cols[i++]); item.weightScaleFormat = Trim(cols[i++]);
            item.xScaleSpec = Trim(cols[i++]); item.xScaleDtype = Trim(cols[i++]);
            item.groupListSpec = Trim(cols[i++]); item.groupListDtype = Trim(cols[i++]);
            item.outputSpec = Trim(cols[i++]); item.outputScaleSpec = Trim(cols[i++]);
            item.dequantMode = stoll(Trim(cols[i++])); item.dequantDtype = stoll(Trim(cols[i++]));
            item.quantMode = stoll(Trim(cols[i++])); item.groupListType = stoll(Trim(cols[i++]));
            item.swigluMode = stoll(Trim(cols[i++])); item.clampLimit = stod(Trim(cols[i++]));
            item.gluAlpha = stod(Trim(cols[i++])); item.gluBias = stod(Trim(cols[i++]));
            item.roundMode = Trim(cols[i++]); item.scaleAlg = stoll(Trim(cols[i++])); item.dstTypeMax = stod(Trim(cols[i++]));
            item.optionalInput = Trim(cols[i++]); item.tuningConfig = Trim(cols[i++]); item.nullArg = Trim(cols[i++]);
            cases.emplace_back(std::move(item));
        } catch (const exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, "", error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<V3Case> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test : public testing::TestWithParam<V3Case> {};

TEST_P(grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test, get_workspace_size)
{
    EXPECT_EQ(GetParam().Run(), GetParam().expected);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, direct_launch_rejects_null_executor)
{
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv,
    grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test,
    testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
        "test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.csv",
        "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_api", __FILE__))),
    BuildCaseName);
} // namespace
