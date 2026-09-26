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
 * \brief Shared CSV-driven opapi UT for GroupedMatmulSwigluQuant V2 and WeightNz V2/V3.
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"
#include "opdev/make_op_executor.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "gmm_csv_acl_parse_utils.h"

using namespace std;
using namespace op;

namespace {

using ops::ut::BuildAclTensorDescFromSpec;
using ops::ut::ParseAclnnStatus;
using ops::ut::ParseBool;
using ops::ut::ParseDims;
using ops::ut::ParseDimsList;
using ops::ut::ParseI64List;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;
using ops::ut::BuildAclTensorListDesc;
using ops::ut::MakeAclTensorDesc;

constexpr size_t kSwigluCsvColumnCount = 45;
constexpr size_t kSwigluV3CsvColumnCount = 31;

struct CsvExpectation {
    string caseName;
    bool checkRet{true};
    aclnnStatus expected{ACLNN_SUCCESS};
};

template <typename Case>
class SwigluOpApiCsvTest : public testing::TestWithParam<Case> {
protected:
    void CheckWorkspaceSize()
    {
        const auto &item = this->GetParam();
        const aclnnStatus actual = item.Run();
        if (item.checkRet) {
            EXPECT_EQ(actual, item.expected);
        }
    }
};

template <typename Case, typename ParseRow>
vector<Case> LoadSwigluCases(const string &csvPath, size_t columnCount, ParseRow parseRow)
{
    ifstream in(csvPath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvPath;
    vector<Case> cases;
    string line;
    bool headerSkipped = false;
    size_t lineNo = 0;
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
        if (cols.size() != columnCount) {
            ADD_FAILURE() << "Invalid CSV column count at line " << lineNo << ": " << cols.size();
            continue;
        }
        try {
            Case item;
            if (parseRow(cols, item)) {
                cases.emplace_back(std::move(item));
            }
        } catch (const exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, "", error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

template <typename Case>
string BuildCaseName(const testing::TestParamInfo<Case> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

void ReleaseCaseTensor(aclTensor *tensor)
{
    Release(tensor);
}
void ReleaseCaseTensorList(aclTensorList *tensors)
{
    Release(tensors);
}

// Retain the framework's TensorDesc/TensorListDesc ownership and null semantics without
// instantiating a different OP_API_UT tuple for every combination of optional inputs.
struct SwigluApiTensors {
    using Tensor = unique_ptr<aclTensor, void (*)(aclTensor *)>;
    using TensorList = unique_ptr<aclTensorList, void (*)(aclTensorList *)>;

    Tensor x{nullptr, ReleaseCaseTensor};
    TensorList weight{nullptr, ReleaseCaseTensorList};
    TensorList weightScale{nullptr, ReleaseCaseTensorList};
    TensorList weightAssist{nullptr, ReleaseCaseTensorList};
    Tensor bias{nullptr, ReleaseCaseTensor};
    Tensor xScale{nullptr, ReleaseCaseTensor};
    Tensor smoothScale{nullptr, ReleaseCaseTensor};
    Tensor groupList{nullptr, ReleaseCaseTensor};
    Tensor output{nullptr, ReleaseCaseTensor};
    Tensor outputScale{nullptr, ReleaseCaseTensor};
    unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> tuningConfig{nullptr, aclDestroyIntArray};

    void SetTuningConfig(const string &value)
    {
        const auto values = ParseI64List(value, "|");
        if (!values.empty()) {
            tuningConfig.reset(aclCreateIntArray(values.data(), values.size()));
        }
    }

    template <typename Query>
    aclnnStatus GetWorkspaceSize(Query query) const
    {
        struct QueryResult {
            uint64_t workspaceSize{0};
            aclOpExecutor *executor{nullptr};
            // Match OpApiUt::TestGetWorkspaceSize, including exceptional exits.
            ~QueryResult()
            {
                delete executor;
            }
        } result;
        // The executor must be destroyed before the input tensor resources.
        return query(&result.workspaceSize, &result.executor);
    }
};

aclDataType ParseDtype(const string &dtype)
{
    return ops::ut::ParseAclDtype(dtype);
}

aclFormat ParseFormat(const string &format)
{
    return ops::ut::ParseAclFormat(format);
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

struct SwigluOpApiCase : CsvExpectation {
    string socVersion;
    string api;
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

    aclnnStatus Run() const
    {
        SetupPlatformForCase(socVersion);
        auto weightShapesVec = ParseDimsList(weightShapes);
        auto weightStorageShapesVec = ParseDimsList(weightStorageShapes);
        auto weightFormatVal = ParseFormat(weightFormat);
        FallbackSingleNzWeightToNdStorage(api, weightFormatVal, weightShapesVec, weightStorageShapesVec);

        SwigluApiTensors tensors;
        tensors.x = MakeAclTensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat),
                                      ParseDims(xStorageShape), ParseDims(xStride))
                        .ToAclType();
        tensors.weight = BuildAclTensorListDesc(weightShapesVec, ParseDtype(weightDtype), weightFormatVal,
                                                weightStorageShapesVec, ParseDimsList(weightStrides))
                             .ToAclType();
        tensors.xScale =
            MakeAclTensorDesc(ParseDims(xScaleShape), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat)).ToAclType();
        tensors.groupList =
            MakeAclTensorDesc(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat))
                .ToAclType();
        tensors.output =
            MakeAclTensorDesc(ParseDims(out1Shape), ParseDtype(out1Dtype), ParseFormat(out1Format)).ToAclType();
        tensors.outputScale =
            MakeAclTensorDesc(ParseDims(out2Shape), ParseDtype(out2Dtype), ParseFormat(out2Format)).ToAclType();

        // An absent optional input is nullptr, not an empty TensorListDesc or a scalar tensor.
        const auto weightScaleShapesVec = ParseDimsList(weightScaleShapes);
        if (!weightScaleShapesVec.empty()) {
            tensors.weightScale =
                BuildAclTensorListDesc(weightScaleShapesVec, ParseDtype(weightScaleDtype),
                                       ParseFormat(weightScaleFormat), ParseDimsList(weightScaleStorageShapes),
                                       ParseDimsList(weightScaleStrides))
                    .ToAclType();
        }
        const auto weightAssistShapesVec = ParseDimsList(weightAssistShapes);
        if (!weightAssistShapesVec.empty()) {
            tensors.weightAssist = BuildAclTensorListDesc(weightAssistShapesVec, ParseDtype(weightAssistDtype),
                                                          ParseFormat(weightAssistFormat))
                                       .ToAclType();
        }
        const auto smoothScaleDims = ParseDims(smoothScaleShape);
        if (!smoothScaleDims.empty()) {
            tensors.smoothScale =
                MakeAclTensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat))
                    .ToAclType();
        }
        tensors.SetTuningConfig(tuningConfig);
        return tensors.GetWorkspaceSize([&](uint64_t *workspaceSize, aclOpExecutor **executor) {
            auto query = api == "WeightNzV2" ? aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize :
                                               aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize;
            return query(tensors.x.get(), tensors.weight.get(), tensors.weightScale.get(), tensors.weightAssist.get(),
                         nullptr, tensors.xScale.get(), tensors.smoothScale.get(), tensors.groupList.get(), dequantMode,
                         dequantDtype, quantMode, quantDtype, tensors.tuningConfig.get(), tensors.output.get(),
                         tensors.outputScale.get(), workspaceSize, executor);
        });
    }
};

bool ParseV2Case(const vector<string> &cols, SwigluOpApiCase &c)
{
    size_t i = 0;
    c.socVersion = Trim(cols[i++]);
    c.caseName = Trim(cols[i++]);
    c.api = Trim(cols[i++]);
    c.checkRet = ParseBool(cols[i++]);
    c.expected = static_cast<aclnnStatus>(stoull(Trim(cols[i++])));
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
    return c.socVersion == "Ascend950" || c.socVersion == "Ascend910B";
}

class grouped_matmul_swiglu_quant_v2_opapi_csv_test : public SwigluOpApiCsvTest<SwigluOpApiCase> {};

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, direct_launch_entry_error_case)
{
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantV2(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV2(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
}

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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    TensorDesc weight0 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
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
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_P(grouped_matmul_swiglu_quant_v2_opapi_csv_test, run_case)
{
    CheckWorkspaceSize();
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_swiglu_quant_v2_opapi_csv, grouped_matmul_swiglu_quant_v2_opapi_csv_test,
                         testing::ValuesIn(LoadSwigluCases<SwigluOpApiCase>(
                             ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_v2.csv",
                                                     "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_api", __FILE__),
                             kSwigluCsvColumnCount, ParseV2Case)),
                         BuildCaseName<SwigluOpApiCase>);

enum class Phase1NullPointerTarget {
    kWorkspaceSize,
    kExecutor,
    kWeightElement,
    kWeightScaleElement,
    kWeightElementWithInvalidCount,
};

aclnnStatus RunPhase1NullPointerCase(Phase1NullPointerTarget target)
{
    SetupPlatformForCase("Ascend950");
    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("1:128:128#16384:128:1#1:4:8:16:32", "FLOAT8_E4M3FN", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("1:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto groupList = BuildAclTensorDescFromSpec("1", "INT64", "ND").ToAclType();
    auto output = BuildAclTensorDescFromSpec("32:64", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto outputScale = BuildAclTensorDescFromSpec("32:1:2", "FLOAT8_E8M0", "ND").ToAclType();

    vector<aclTensor *> weightElements = {weightTensor.get()};
    vector<aclTensor *> weightScaleElements = {weightScaleTensor.get()};
    if (target == Phase1NullPointerTarget::kWeightElement) {
        weightElements = {nullptr};
    } else if (target == Phase1NullPointerTarget::kWeightScaleElement) {
        weightScaleElements = {nullptr};
    } else if (target == Phase1NullPointerTarget::kWeightElementWithInvalidCount) {
        weightElements.emplace_back(nullptr);
    }
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(
        aclCreateTensorList(weightElements.data(), weightElements.size()), aclDestroyTensorList);
    if (weight == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (target != Phase1NullPointerTarget::kWeightElement) {
        (void)weightTensor.release();
    }

    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(
        aclCreateTensorList(weightScaleElements.data(), weightScaleElements.size()), aclDestroyTensorList);
    if (weightScale == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (target != Phase1NullPointerTarget::kWeightScaleElement) {
        (void)weightScaleTensor.release();
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    uint64_t *workspaceSizeArg = target == Phase1NullPointerTarget::kWorkspaceSize ? nullptr : &workspaceSize;
    aclOpExecutor **executorArg = target == Phase1NullPointerTarget::kExecutor ? nullptr : &executor;
    const aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x.get(), weight.get(), weightScale.get(), nullptr, nullptr, xScale.get(), nullptr, groupList.get(), 2, 0, 2, 0,
        nullptr, 2, 7.0, 1.702, 1.0, "rint", 0, 0.0, output.get(), outputScale.get(), workspaceSizeArg, executorArg);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    return ret;
}

aclnnStatus RunHostGroupListCase(const vector<int64_t> &storageValues, int64_t stride, int64_t offset,
                                 int64_t groupListType)
{
    SetupPlatformForCase("Ascend950");
    auto inputExecutor = CREATE_EXECUTOR();
    if (inputExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    // The input executor owns the copied Host storage until the API executor has been destroyed.
    auto *groupList =
        inputExecutor->AllocHostTensor(storageValues.data(), storageValues.size(), op::DataType::DT_INT64);
    if (groupList == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    groupList->SetViewShape(op::Shape({2}));
    groupList->SetViewStrides(op::Strides({stride}));
    groupList->SetViewOffset(offset);

    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("2:128:128#16384:128:1#2:4:8:16:32", "FLOAT8_E4M3FN", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("2:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto output = BuildAclTensorDescFromSpec("32:64", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto outputScale = BuildAclTensorDescFromSpec("32:1:2", "FLOAT8_E8M0", "ND").ToAclType();

    const aclTensor *weightElements[] = {weightTensor.get()};
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(aclCreateTensorList(weightElements, 1),
                                                                      aclDestroyTensorList);
    if (weight == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    (void)weightTensor.release();
    const aclTensor *weightScaleElements[] = {weightScaleTensor.get()};
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(aclCreateTensorList(weightScaleElements, 1),
                                                                           aclDestroyTensorList);
    if (weightScale == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    (void)weightScaleTensor.release();

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x.get(), weight.get(), weightScale.get(), nullptr, nullptr, xScale.get(), nullptr, groupList, 2, 0, 2,
        groupListType, nullptr, 2, 7.0, 1.702, 1.0, "rint", 0, 0.0, output.get(), outputScale.get(), &workspaceSize,
        &executor);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    return ret;
}

aclOpExecutor *CreateTestExecutor()
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return nullptr;
    }
    aclOpExecutor *executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);
    return executor;
}

struct V3Case : CsvExpectation {
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
        SetupPlatformForCase("Ascend950");
        SwigluApiTensors tensors;
        if (nullArg != "x") {
            tensors.x = BuildAclTensorDescFromSpec(xSpec, xDtype, xFormat).ToAclType();
        }
        if (nullArg != "weight") {
            tensors.weight = BuildAclTensorListDesc(weightSpecs, weightDtype, weightFormat).ToAclType();
        }
        if (nullArg != "weight_scale") {
            tensors.weightScale =
                BuildAclTensorListDesc(weightScaleSpecs, weightScaleDtype, weightScaleFormat).ToAclType();
        }
        if (nullArg != "x_scale") {
            tensors.xScale = BuildAclTensorDescFromSpec(xScaleSpec, xScaleDtype, "ND").ToAclType();
        }
        if (nullArg != "group_list") {
            tensors.groupList = BuildAclTensorDescFromSpec(groupListSpec, groupListDtype, "ND").ToAclType();
        }
        if (nullArg != "output") {
            tensors.output = BuildAclTensorDescFromSpec(outputSpec, "FLOAT8_E4M3FN", "ND").ToAclType();
        }
        if (nullArg != "output_scale") {
            tensors.outputScale = BuildAclTensorDescFromSpec(outputScaleSpec, "FLOAT8_E8M0", "ND").ToAclType();
        }
        // NULL cases take precedence over the optional-input rejection cases, as in the original UT.
        if (tensors.x && tensors.weight && tensors.weightScale && tensors.xScale && tensors.groupList &&
            tensors.output && tensors.outputScale) {
            if (optionalInput == "assist") {
                tensors.weightAssist =
                    TensorListDesc(vector<TensorDesc>{MakeAclTensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)}).ToAclType();
            } else if (optionalInput == "bias") {
                tensors.bias = MakeAclTensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            } else if (optionalInput == "smooth") {
                tensors.smoothScale = MakeAclTensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            }
        }
        tensors.SetTuningConfig(tuningConfig);
        const char *roundModePtr = roundMode == "NULL" ? nullptr : roundMode.c_str();
        return tensors.GetWorkspaceSize([&](uint64_t *workspaceSize, aclOpExecutor **executor) {
            return aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
                tensors.x.get(), tensors.weight.get(), tensors.weightScale.get(), tensors.weightAssist.get(),
                tensors.bias.get(), tensors.xScale.get(), tensors.smoothScale.get(), tensors.groupList.get(),
                dequantMode, dequantDtype, quantMode, groupListType, tensors.tuningConfig.get(), swigluMode, clampLimit,
                gluAlpha, gluBias, roundModePtr, scaleAlg, dstTypeMax, tensors.output.get(), tensors.outputScale.get(),
                workspaceSize, executor);
        });
    }
};

bool ParseV3Case(const vector<string> &cols, V3Case &item)
{
    size_t i = 0;
    item.caseName = Trim(cols[i++]);
    item.expected = ParseAclnnStatus(cols[i++]);
    item.xSpec = Trim(cols[i++]);
    item.xDtype = Trim(cols[i++]);
    item.xFormat = Trim(cols[i++]);
    item.weightSpecs = Trim(cols[i++]);
    item.weightDtype = Trim(cols[i++]);
    item.weightFormat = Trim(cols[i++]);
    item.weightScaleSpecs = Trim(cols[i++]);
    item.weightScaleDtype = Trim(cols[i++]);
    item.weightScaleFormat = Trim(cols[i++]);
    item.xScaleSpec = Trim(cols[i++]);
    item.xScaleDtype = Trim(cols[i++]);
    item.groupListSpec = Trim(cols[i++]);
    item.groupListDtype = Trim(cols[i++]);
    item.outputSpec = Trim(cols[i++]);
    item.outputScaleSpec = Trim(cols[i++]);
    item.dequantMode = stoll(Trim(cols[i++]));
    item.dequantDtype = stoll(Trim(cols[i++]));
    item.quantMode = stoll(Trim(cols[i++]));
    item.groupListType = stoll(Trim(cols[i++]));
    item.swigluMode = stoll(Trim(cols[i++]));
    item.clampLimit = stod(Trim(cols[i++]));
    item.gluAlpha = stod(Trim(cols[i++]));
    item.gluBias = stod(Trim(cols[i++]));
    item.roundMode = Trim(cols[i++]);
    item.scaleAlg = stoll(Trim(cols[i++]));
    item.dstTypeMax = stod(Trim(cols[i++]));
    item.optionalInput = Trim(cols[i++]);
    item.tuningConfig = Trim(cols[i++]);
    item.nullArg = Trim(cols[i++]);
    return true;
}

class grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test : public SwigluOpApiCsvTest<V3Case> {};

TEST_P(grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test, get_workspace_size)
{
    CheckWorkspaceSize();
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase1_output_pointers_return_param_nullptr)
{
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWorkspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kExecutor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_ignore_padding)
{
    EXPECT_EQ(RunHostGroupListCase({16, 99, 32}, 2, 0, 0), ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({99, 16, 99, 32}, 2, 1, 0), ACLNN_SUCCESS);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_check_logical_values)
{
    EXPECT_EQ(RunHostGroupListCase({16, 20, -1}, 2, 0, 0), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunHostGroupListCase({16, 20, 8}, 2, 0, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_check_logical_sum)
{
    EXPECT_EQ(RunHostGroupListCase({12, 99, 20}, 2, 0, 1), ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({16, 1, 20}, 2, 0, 1), ACLNN_ERR_PARAM_INVALID);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_contiguous_values_are_unchanged)
{
    EXPECT_EQ(RunHostGroupListCase({16, 32}, 1, 0, 0), ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({12, 20}, 1, 0, 1), ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({16, -1}, 1, 0, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, tensor_list_elements_return_param_nullptr)
{
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWeightElement), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWeightScaleElement), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWeightElementWithInvalidCount),
              ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase2_executor_returns_param_nullptr)
{
    EXPECT_EQ(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase2_required_workspace_returns_param_nullptr)
{
    unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)> executor(CreateTestExecutor(),
                                                                           aclDestroyAclOpExecutor);
    ASSERT_NE(executor.get(), nullptr);
    EXPECT_EQ(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, 1, executor.get(), nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv,
                         grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test,
                         testing::ValuesIn(LoadSwigluCases<V3Case>(
                             ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.csv",
                                                     "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_api", __FILE__),
                             kSwigluV3CsvColumnCount, ParseV3Case)),
                         BuildCaseName<V3Case>);
} // namespace
