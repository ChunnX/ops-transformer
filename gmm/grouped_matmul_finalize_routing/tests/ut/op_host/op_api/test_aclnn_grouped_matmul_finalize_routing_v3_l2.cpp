/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_v3.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kV3CsvColumnCount = 42;

vector<int64_t> ParseDims(const string &value)
{
    return ops::ut::ParseDims(value);
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

aclnnStatus ParseStatus(const string &status)
{
    static const map<string, aclnnStatus> statusMap = {
        {"SUCCESS", ACLNN_SUCCESS},
        {"ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    auto it = statusMap.find(Trim(status));
    return it == statusMap.end() ? ACLNN_SUCCESS : it->second;
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, SocVersion> socMap = {
        {"Ascend910B", SocVersion::ASCEND910B},
        {"Ascend950", SocVersion::ASCEND950},
    };
    auto it = socMap.find(Trim(socVersion));
    SetPlatformSocVersion(it == socMap.end() ? SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t> &storageShape)
{
    return ops::ut::MakeAclTensorDesc(shape, dtype, format, storageShape).ValueRange(-1, 1);
}

struct GroupedMatmulFinalizeRoutingV3Case {
    void Run() const
    {
        SetupPlatformForCase(socVersion);

        TensorDesc x1 = MakeTensorDesc(ParseDims(x1Shape), ParseDtype(x1Dtype), ACL_FORMAT_ND, {});
        TensorDesc x2 =
            MakeTensorDesc(ParseDims(x2Shape), ParseDtype(x2Dtype), ParseFormat(x2Format), ParseDims(x2StorageShape));
        TensorDesc scale =
            MakeTensorDesc(ParseDims(scaleShape), ParseDtype(scaleDtype), ACL_FORMAT_ND, {}).ValueRange(0, 3);
        TensorDesc bias = MakeTensorDesc(ParseDims(biasShape), ParseDtype(biasDtype), ACL_FORMAT_ND, {});
        TensorDesc offset =
            MakeTensorDesc(ParseDims(offsetShape), ParseDtype(offsetDtype), ParseFormat(offsetFormat), {});
        TensorDesc antiQuantScale =
            MakeTensorDesc(ParseDims(antiQuantScaleShape), ParseDtype(antiQuantScaleDtype), ACL_FORMAT_ND, {});
        TensorDesc perTokenScale =
            MakeTensorDesc(ParseDims(perTokenScaleShape), ParseDtype(perTokenScaleDtype), ACL_FORMAT_ND, {})
                .ValueRange(0, 3);
        TensorDesc groupList =
            MakeTensorDesc(ParseDims(groupListShape), ACL_INT64, ACL_FORMAT_ND, {}).ValueRange(-1, 1);
        const vector<int64_t> groupListVals = ParseI64List(groupListValues);
        if (!groupListVals.empty()) {
            groupList.Value(groupListVals);
        }
        TensorDesc sharedInput =
            MakeTensorDesc(ParseDims(sharedInputShape), ParseDtype(sharedInputDtype), ACL_FORMAT_ND, {});
        TensorDesc logits = MakeTensorDesc(ParseDims(logitsShape), ParseDtype(logitsDtype), ACL_FORMAT_ND, {});
        TensorDesc rowIndex =
            MakeTensorDesc(ParseDims(rowIndexShape), ParseDtype(rowIndexDtype), ACL_FORMAT_ND, {});
        TensorDesc out = MakeTensorDesc(ParseDims(outShape), ParseDtype(outDtype), ACL_FORMAT_ND, {});

        const vector<int64_t> tuningVals = ParseI64List(tuningConfig);
        aclIntArray *tuningConfigPtr = nullptr;
        if (ParseBool(hasTuningConfig)) {
            tuningConfigPtr = aclCreateIntArray(const_cast<int64_t *>(tuningVals.data()), tuningVals.size());
        }

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;

        if (!ParseBool(hasOffset)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, nullptr, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasLogits)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, offset, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, nullptr, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasAntiQuantScale)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, offset, antiQuantScale, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(x1, x2, scale, bias, offset, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, ParseStatus(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string x1Shape;
    string x1Dtype;
    string x2Shape;
    string x2Dtype;
    string x2Format;
    string x2StorageShape;
    string scaleShape;
    string scaleDtype;
    string biasShape;
    string biasDtype;
    string offsetShape;
    string offsetDtype;
    string offsetFormat;
    string antiQuantScaleShape;
    string antiQuantScaleDtype;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string groupListShape;
    string groupListValues;
    string sharedInputShape;
    string sharedInputDtype;
    string logitsShape;
    string logitsDtype;
    string rowIndexShape;
    string rowIndexDtype;
    string outShape;
    string outDtype;
    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t groupListType = 1;
    string tuningConfig;
    string hasTuningConfig;
    string hasOffset;
    string hasAntiQuantScale;
    string hasLogits;
    string expectRet;
    string checkRet;
};

vector<GroupedMatmulFinalizeRoutingV3Case> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulFinalizeRoutingV3Case> cases;
    string line;
    bool skippedHeader = false;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!skippedHeader) {
            skippedHeader = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != kV3CsvColumnCount) {
            continue;
        }
        GroupedMatmulFinalizeRoutingV3Case c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.x1Shape = Trim(cols[i++]);
        c.x1Dtype = Trim(cols[i++]);
        c.x2Shape = Trim(cols[i++]);
        c.x2Dtype = Trim(cols[i++]);
        c.x2Format = Trim(cols[i++]);
        c.x2StorageShape = Trim(cols[i++]);
        c.scaleShape = Trim(cols[i++]);
        c.scaleDtype = Trim(cols[i++]);
        c.biasShape = Trim(cols[i++]);
        c.biasDtype = Trim(cols[i++]);
        c.offsetShape = Trim(cols[i++]);
        c.offsetDtype = Trim(cols[i++]);
        c.offsetFormat = Trim(cols[i++]);
        c.antiQuantScaleShape = Trim(cols[i++]);
        c.antiQuantScaleDtype = Trim(cols[i++]);
        c.perTokenScaleShape = Trim(cols[i++]);
        c.perTokenScaleDtype = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.sharedInputShape = Trim(cols[i++]);
        c.sharedInputDtype = Trim(cols[i++]);
        c.logitsShape = Trim(cols[i++]);
        c.logitsDtype = Trim(cols[i++]);
        c.rowIndexShape = Trim(cols[i++]);
        c.rowIndexDtype = Trim(cols[i++]);
        c.outShape = Trim(cols[i++]);
        c.outDtype = Trim(cols[i++]);
        c.dtype = stoll(Trim(cols[i++]));
        c.sharedInputWeight = stof(Trim(cols[i++]));
        c.sharedInputOffset = stoll(Trim(cols[i++]));
        c.transposeX = ParseBool(Trim(cols[i++]));
        c.transposeW = ParseBool(Trim(cols[i++]));
        c.groupListType = stoll(Trim(cols[i++]));
        c.tuningConfig = Trim(cols[i++]);
        c.hasTuningConfig = Trim(cols[i++]);
        c.hasOffset = Trim(cols[i++]);
        c.hasAntiQuantScale = Trim(cols[i++]);
        c.hasLogits = Trim(cols[i++]);
        c.expectRet = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        cases.emplace_back(c);
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingV3Case> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class grouped_matmul_finalize_routing_v3_csv_test
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingV3Case> {};

TEST_P(grouped_matmul_finalize_routing_v3_csv_test, run_case)
{
    GetParam().Run();
}

TEST(grouped_matmul_finalize_routing_v3_manual_test, shared_input_optional_on_ascend950)
{
    SetupPlatformForCase("Ascend950");

    TensorDesc x1({192, 2048}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2({4, 2048, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc scale({4, 32, 7168, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias({4, 7168}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale({192, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList({4}, ACL_INT64, ACL_FORMAT_ND);
    groupList.Value(vector<int64_t>{24, 24, 24, 24});
    TensorDesc logits({192}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc rowIndex({192}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out({24, 7168}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                        INPUT(x1, x2, scale, bias, nullptr, nullptr, nullptr, perTokenScale, groupList, nullptr,
                              logits, rowIndex, 0, 1.0F, 0, false, false, 1, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_finalize_routing_v3_csv,
    grouped_matmul_finalize_routing_v3_csv_test,
    testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
        "test_aclnn_grouped_matmul_finalize_routing_v3_l2.csv",
        "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api", __FILE__))),
    BuildCaseName);
}  // namespace
