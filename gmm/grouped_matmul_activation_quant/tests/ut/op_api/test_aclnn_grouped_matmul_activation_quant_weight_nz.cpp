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
 * \file test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp
 * \brief CSV-driven opapi UT for aclnnGroupedMatmulActivationQuantWeightNz.
 */

#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_activation_quant_weight_nz.h"
#include "gmm_csv_acl_parse_utils.h"
#include "gmm_soc_version_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {

using ops::ut::BuildAclTensorListDesc;
using ops::ut::ParseBool;
using ops::ut::ParseDims;
using ops::ut::ParseDimsList;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 50U;
constexpr int64_t kDefaultDequantDtype = 0L;
constexpr int64_t kDefaultYDtype = 36L;
constexpr int64_t kDefaultGroupListType = 0L;
constexpr int64_t kDefaultScaleAlg = 0L;
constexpr float kDefaultDstTypeMax = 0.0F;
constexpr const char *kActivationType = "gelu_tanh";
constexpr const char *kQuantMode = "MX";
constexpr const char *kRoundMode = "rint";

TensorDesc MakeTensorDesc(const string &shape, const string &dtype, const string &format,
                          const string &storageShape = "NONE", const string &stride = "NONE")
{
    return TensorDesc(ParseDims(shape), ops::ut::ParseAclDtype(dtype), ops::ut::ParseAclFormat(format),
                      ParseDims(stride), 0, ParseDims(storageShape))
        .ValueRange(-10, 10);
}

TensorListDesc MakeTensorListDesc(const string &shapes, const string &dtype, const string &format,
                                  const string &storageShapes = "NONE", const string &strides = "NONE")
{
    return BuildAclTensorListDesc(ParseDimsList(shapes), ops::ut::ParseAclDtype(dtype),
                                  ops::ut::ParseAclFormat(format), ParseDimsList(storageShapes),
                                  ParseDimsList(strides));
}

struct GroupedMatmulActivationQuantWeightNzOpApiCase {
    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        ops::ut::SetPlatformSocVersion(socVersion);

        TensorDesc xDesc = MakeTensorDesc(xShape, xDtype, xFormat, xStorageShape, xStride);
        TensorDesc groupListDesc = MakeTensorDesc(groupListShape, groupListDtype, groupListFormat);
        TensorListDesc weightDesc = MakeTensorListDesc(weightShapes, weightDtype, weightFormat,
                                                       weightStorageShapes, weightStrides);
        TensorListDesc weightScaleDesc = MakeTensorListDesc(weightScaleShapes, weightScaleDtype, weightScaleFormat,
                                                            weightScaleStorageShapes, weightScaleStrides);
        TensorDesc yDesc = MakeTensorDesc(yShape, yDtype, yFormat);
        TensorDesc yScaleDesc = MakeTensorDesc(yScaleShape, yScaleDtype, yScaleFormat);

        const char *activationTypePtr = activationTypeNull ? nullptr : activationType.c_str();
        const char *roundModePtr = roundModeNull ? nullptr : roundMode.c_str();

        vector<int64_t> tuningValues = ops::ut::ParseI64List(tuningConfig);
        aclIntArray *tuningConfigArr = tuningValues.empty() ? nullptr :
            aclCreateIntArray(tuningValues.data(), tuningValues.size());

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;
        if (biasNull && xScaleNull) {
            auto ut = OP_API_UT(aclnnGroupedMatmulActivationQuantWeightNz,
                                INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, nullptr, nullptr,
                                      activationTypePtr, dequantDtype, transposeWeight, groupListType,
                                      tuningConfigArr, quantMode.c_str(), attrYDtype, roundModePtr, scaleAlg,
                                      dstTypeMax),
                                OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (biasNull) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            auto ut = OP_API_UT(aclnnGroupedMatmulActivationQuantWeightNz,
                                INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, nullptr, xScaleDesc,
                                      activationTypePtr, dequantDtype, transposeWeight, groupListType,
                                      tuningConfigArr, quantMode.c_str(), attrYDtype, roundModePtr, scaleAlg,
                                      dstTypeMax),
                                OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (xScaleNull) {
            TensorListDesc biasDesc = MakeTensorListDesc(biasShapes, biasDtype, biasFormat);
            auto ut = OP_API_UT(aclnnGroupedMatmulActivationQuantWeightNz,
                                INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, biasDesc, nullptr,
                                      activationTypePtr, dequantDtype, transposeWeight, groupListType,
                                      tuningConfigArr, quantMode.c_str(), attrYDtype, roundModePtr, scaleAlg,
                                      dstTypeMax),
                                OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            TensorListDesc biasDesc = MakeTensorListDesc(biasShapes, biasDtype, biasFormat);
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            auto ut = OP_API_UT(aclnnGroupedMatmulActivationQuantWeightNz,
                                INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, biasDesc, xScaleDesc,
                                      activationTypePtr, dequantDtype, transposeWeight, groupListType,
                                      tuningConfigArr, quantMode.c_str(), attrYDtype, roundModePtr, scaleAlg,
                                      dstTypeMax),
                                OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        if (tuningConfigArr != nullptr) {
            aclDestroyIntArray(tuningConfigArr);
        }
        if (checkRet) {
            EXPECT_EQ(ret, ops::ut::ParseAclnnStatus(expectRet)) << "case=" << caseName;
        }
        if (expectWorkspaceZero) {
            EXPECT_EQ(workspaceSize, 0U) << "case=" << caseName;
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    bool checkRet = true;
    string expectRet;
    bool expectWorkspaceZero = false;
    string xShape;
    string xDtype;
    string xFormat;
    string xStorageShape;
    string xStride;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
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
    bool biasNull = true;
    string biasShapes;
    string biasDtype;
    string biasFormat;
    bool xScaleNull = false;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    bool activationTypeNull = false;
    string activationType;
    int64_t dequantDtype = kDefaultDequantDtype;
    bool transposeWeight = false;
    int64_t groupListType = kDefaultGroupListType;
    string tuningConfig;
    string quantMode;
    int64_t attrYDtype = kDefaultYDtype;
    bool roundModeNull = false;
    string roundMode;
    int64_t scaleAlg = kDefaultScaleAlg;
    float dstTypeMax = kDefaultDstTypeMax;
    string yShape;
    string yDtype;
    string yFormat;
    string yScaleShape;
    string yScaleDtype;
    string yScaleFormat;
};

vector<GroupedMatmulActivationQuantWeightNzOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulActivationQuantWeightNzOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    size_t lineNo = 0U;
    while (getline(in, line)) {
        ++lineNo;
        if (line.empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < kCsvColumnCount) {
            continue;
        }

        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            GroupedMatmulActivationQuantWeightNzOpApiCase c;
            size_t i = 0U;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.enable = ParseBool(Trim(cols[i++]));
            c.checkRet = ParseBool(Trim(cols[i++]));
            c.expectRet = Trim(cols[i++]);
            c.expectWorkspaceZero = ParseBool(Trim(cols[i++]));
            c.xShape = Trim(cols[i++]);
            c.xDtype = Trim(cols[i++]);
            c.xFormat = Trim(cols[i++]);
            c.xStorageShape = Trim(cols[i++]);
            c.xStride = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.groupListDtype = Trim(cols[i++]);
            c.groupListFormat = Trim(cols[i++]);
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
            c.biasNull = ParseBool(Trim(cols[i++]));
            c.biasShapes = Trim(cols[i++]);
            c.biasDtype = Trim(cols[i++]);
            c.biasFormat = Trim(cols[i++]);
            c.xScaleNull = ParseBool(Trim(cols[i++]));
            c.xScaleShape = Trim(cols[i++]);
            c.xScaleDtype = Trim(cols[i++]);
            c.xScaleFormat = Trim(cols[i++]);
            c.activationTypeNull = ParseBool(Trim(cols[i++]));
            c.activationType = Trim(cols[i++]);
            c.dequantDtype = stoll(Trim(cols[i++]));
            c.transposeWeight = ParseBool(Trim(cols[i++]));
            c.groupListType = stoll(Trim(cols[i++]));
            c.tuningConfig = Trim(cols[i++]);
            c.quantMode = Trim(cols[i++]);
            c.attrYDtype = stoll(Trim(cols[i++]));
            c.roundModeNull = ParseBool(Trim(cols[i++]));
            c.roundMode = Trim(cols[i++]);
            c.scaleAlg = stoll(Trim(cols[i++]));
            c.dstTypeMax = stod(Trim(cols[i++]));
            c.yShape = Trim(cols[i++]);
            c.yDtype = Trim(cols[i++]);
            c.yFormat = Trim(cols[i++]);
            c.yScaleShape = Trim(cols[i++]);
            c.yScaleDtype = Trim(cols[i++]);
            c.yScaleFormat = Trim(cols[i++]);
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulActivationQuantWeightNzOpApiCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class grouped_matmul_activation_quant_weight_nz_opapi_csv_test :
    public testing::TestWithParam<GroupedMatmulActivationQuantWeightNzOpApiCase> {};

TEST_P(grouped_matmul_activation_quant_weight_nz_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_activation_quant_weight_nz_opapi_csv,
    grouped_matmul_activation_quant_weight_nz_opapi_csv_test,
    testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
        "test_aclnn_grouped_matmul_activation_quant_weight_nz.csv",
        "gmm/grouped_matmul_activation_quant/tests/ut/op_api", __FILE__))),
    BuildCaseName);

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, workspace_output_ptr_null)
{
    EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, kActivationType, kDefaultDequantDtype,
                  false, kDefaultGroupListType, nullptr, kQuantMode, kDefaultYDtype, kRoundMode, kDefaultScaleAlg,
                  kDefaultDstTypeMax, nullptr, nullptr, nullptr, nullptr),
              ACLNN_SUCCESS);
}

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, required_tensorlist_null)
{
    TensorDesc xDesc = MakeTensorDesc("0:128", "FLOAT8_E4M3FN", "ND");
    TensorDesc groupListDesc = MakeTensorDesc("1", "INT64", "ND");
    TensorDesc xScaleDesc = MakeTensorDesc("0:2:2", "FLOAT8_E8M0", "ND");
    TensorDesc yDesc = MakeTensorDesc("0:64", "FLOAT8_E4M3FN", "ND");
    TensorDesc yScaleDesc = MakeTensorDesc("0:1:2", "FLOAT8_E8M0", "ND");
    auto x = xDesc.ToAclType();
    auto groupList = groupListDesc.ToAclType();
    auto xScale = xScaleDesc.ToAclType();
    auto y = yDesc.ToAclType();
    auto yScale = yScaleDesc.ToAclType();

    uint64_t workspaceSize = 0U;
    aclOpExecutor *executor = nullptr;
    EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                  x.get(), groupList.get(), nullptr, nullptr, nullptr, xScale.get(), kActivationType,
                  kDefaultDequantDtype, false, kDefaultGroupListType, nullptr, kQuantMode, kDefaultYDtype, kRoundMode,
                  kDefaultScaleAlg, kDefaultDstTypeMax, y.get(), yScale.get(), &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, empty_tensorlist_invalid)
{
    TensorDesc xDesc = MakeTensorDesc("0:128", "FLOAT8_E4M3FN", "ND");
    TensorDesc groupListDesc = MakeTensorDesc("1", "INT64", "ND");
    TensorDesc xScaleDesc = MakeTensorDesc("0:2:2", "FLOAT8_E8M0", "ND");
    TensorDesc yDesc = MakeTensorDesc("0:64", "FLOAT8_E4M3FN", "ND");
    TensorDesc yScaleDesc = MakeTensorDesc("0:1:2", "FLOAT8_E8M0", "ND");
    auto x = xDesc.ToAclType();
    auto groupList = groupListDesc.ToAclType();
    auto xScale = xScaleDesc.ToAclType();
    auto y = yDesc.ToAclType();
    auto yScale = yScaleDesc.ToAclType();
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> emptyList(aclCreateTensorList(nullptr, 0),
                                                                         aclDestroyTensorList);

    uint64_t workspaceSize = 0U;
    aclOpExecutor *executor = nullptr;
    EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                  x.get(), groupList.get(), emptyList.get(), emptyList.get(), nullptr, xScale.get(), kActivationType,
                  kDefaultDequantDtype, false, kDefaultGroupListType, nullptr, kQuantMode, kDefaultYDtype, kRoundMode,
                  kDefaultScaleAlg, kDefaultDstTypeMax, y.get(), yScale.get(), &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, tensorlist_element_null_invalid)
{
    TensorDesc xDesc = MakeTensorDesc("0:128", "FLOAT8_E4M3FN", "ND");
    TensorDesc groupListDesc = MakeTensorDesc("1", "INT64", "ND");
    TensorDesc xScaleDesc = MakeTensorDesc("0:2:2", "FLOAT8_E8M0", "ND");
    TensorDesc yDesc = MakeTensorDesc("0:64", "FLOAT8_E4M3FN", "ND");
    TensorDesc yScaleDesc = MakeTensorDesc("0:1:2", "FLOAT8_E8M0", "ND");
    TensorListDesc weightScaleDesc = MakeTensorListDesc("1:2:64:2", "FLOAT8_E8M0", "ND");
    auto x = xDesc.ToAclType();
    auto groupList = groupListDesc.ToAclType();
    auto weightScaleList = weightScaleDesc.ToAclType();
    auto xScale = xScaleDesc.ToAclType();
    auto y = yDesc.ToAclType();
    auto yScale = yScaleDesc.ToAclType();
    aclTensor *nullTensor = nullptr;
    aclTensorList *weightList = aclCreateTensorList(&nullTensor, 1);

    uint64_t workspaceSize = 0U;
    aclOpExecutor *executor = nullptr;
    EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                  x.get(), groupList.get(), weightList, weightScaleList.get(), nullptr, xScale.get(),
                  kActivationType, kDefaultDequantDtype, false, kDefaultGroupListType, nullptr, kQuantMode,
                  kDefaultYDtype, kRoundMode, kDefaultScaleAlg, kDefaultDstTypeMax, y.get(), yScale.get(),
                  &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, empty_bias_tensorlist_valid_empty_tensor)
{
    TensorDesc xDesc = MakeTensorDesc("0:128", "FLOAT8_E4M3FN", "ND");
    TensorDesc groupListDesc = MakeTensorDesc("1", "INT64", "ND");
    TensorListDesc weightDesc = MakeTensorListDesc("1:2:8:16:32", "FLOAT8_E4M3FN", "FRACTAL_NZ");
    TensorListDesc weightScaleDesc = MakeTensorListDesc("1:2:64:2", "FLOAT8_E8M0", "ND");
    TensorDesc xScaleDesc = MakeTensorDesc("0:2:2", "FLOAT8_E8M0", "ND");
    TensorDesc yDesc = MakeTensorDesc("0:64", "FLOAT8_E4M3FN", "ND");
    TensorDesc yScaleDesc = MakeTensorDesc("0:1:2", "FLOAT8_E8M0", "ND");
    auto x = xDesc.ToAclType();
    auto groupList = groupListDesc.ToAclType();
    auto weight = weightDesc.ToAclType();
    auto weightScale = weightScaleDesc.ToAclType();
    auto xScale = xScaleDesc.ToAclType();
    auto y = yDesc.ToAclType();
    auto yScale = yScaleDesc.ToAclType();
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> emptyBias(aclCreateTensorList(nullptr, 0),
                                                                         aclDestroyTensorList);

    uint64_t workspaceSize = 1U;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                  x.get(), groupList.get(), weight.get(), weightScale.get(), emptyBias.get(), xScale.get(),
                  kActivationType, kDefaultDequantDtype, false, kDefaultGroupListType, nullptr, kQuantMode,
                  kDefaultYDtype, kRoundMode, kDefaultScaleAlg, kDefaultDstTypeMax, y.get(), yScale.get(),
                  &workspaceSize, &executor),
              ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, output_tensor_null_invalid)
{
    TensorDesc xDesc = MakeTensorDesc("0:128", "FLOAT8_E4M3FN", "ND");
    TensorDesc groupListDesc = MakeTensorDesc("1", "INT64", "ND");
    TensorListDesc weightDesc = MakeTensorListDesc("1:2:8:16:32", "FLOAT8_E4M3FN", "FRACTAL_NZ");
    TensorListDesc weightScaleDesc = MakeTensorListDesc("1:2:64:2", "FLOAT8_E8M0", "ND");
    TensorDesc xScaleDesc = MakeTensorDesc("0:2:2", "FLOAT8_E8M0", "ND");
    TensorDesc yScaleDesc = MakeTensorDesc("0:1:2", "FLOAT8_E8M0", "ND");
    auto x = xDesc.ToAclType();
    auto groupList = groupListDesc.ToAclType();
    auto weight = weightDesc.ToAclType();
    auto weightScale = weightScaleDesc.ToAclType();
    auto xScale = xScaleDesc.ToAclType();
    auto yScale = yScaleDesc.ToAclType();

    uint64_t workspaceSize = 0U;
    aclOpExecutor *executor = nullptr;
    EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                  x.get(), groupList.get(), weight.get(), weightScale.get(), nullptr, xScale.get(), kActivationType,
                  kDefaultDequantDtype, false, kDefaultGroupListType, nullptr, kQuantMode, kDefaultYDtype, kRoundMode,
                  kDefaultScaleAlg, kDefaultDstTypeMax, nullptr, yScale.get(), &workspaceSize, &executor),
              ACLNN_SUCCESS);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST(grouped_matmul_activation_quant_weight_nz_opapi_direct_test, phase2_null_executor_path)
{
    EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNz(nullptr, 0, nullptr, nullptr), ACLNN_SUCCESS);
}

} // namespace
