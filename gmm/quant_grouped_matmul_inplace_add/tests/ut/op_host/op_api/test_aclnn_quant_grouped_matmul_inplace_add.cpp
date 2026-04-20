/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_quant_grouped_matmul_inplace_add.h"
#include "csv_test_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

namespace {

using csv_test_utils::FindFirstExistingPath;
using csv_test_utils::GetCwd;
using csv_test_utils::GetCurrentFileDir;
using csv_test_utils::GetExeDirPath;
using csv_test_utils::JoinPath;
using csv_test_utils::ParseBool;
using csv_test_utils::ParseDims;
using csv_test_utils::ParseI64List;
using csv_test_utils::SplitStr2Vec;
using csv_test_utils::Trim;

constexpr size_t kCsvColumnCount = 22;

string ResolveCsvPath(const string &csvName)
{
    const string cwd = GetCwd();
    const string codePath = []() {
        const char *env = std::getenv("CODE_PATH");
        return env == nullptr ? string() : string(env);
    }();

    vector<string> candidates;
    candidates.emplace_back(JoinPath(GetExeDirPath(), csvName));
    candidates.emplace_back(JoinPath(GetCurrentFileDir(__FILE__), csvName));
    candidates.emplace_back(JoinPath(cwd, csvName));
    candidates.emplace_back(
        JoinPath(cwd, "../../../../../gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host/op_api/" + csvName));
    if (!codePath.empty()) {
        candidates.emplace_back(
            JoinPath(codePath, "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host/op_api/" + csvName));
    }
    return FindFirstExistingPath(candidates);
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},
        {"INT64", ACL_INT64},
        {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"HIFLOAT8", ACL_HIFLOAT8},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclnnStatus ParseStatus(const string &value)
{
    static const map<string, aclnnStatus> statusMap = {
        {"ACLNN_SUCCESS", ACLNN_SUCCESS},
        {"ACLNN_ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
    };
    auto it = statusMap.find(value);
    if (it != statusMap.end()) {
        return it->second;
    }
    return static_cast<aclnnStatus>(stoll(value));
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

struct QGmmInplaceAddOpApiCase {
    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        SetupPlatformForCase(socVersion);

        TensorDesc x1Desc(ParseDims(x1Shape), ParseDtype(x1Dtype), ACL_FORMAT_ND, ParseDims(x1Stride));
        TensorDesc x2Desc(ParseDims(x2Shape), ParseDtype(x2Dtype), ACL_FORMAT_ND, ParseDims(x2Stride));
        TensorDesc scale2Desc(ParseDims(scale2Shape), ParseDtype(scale2Dtype), ACL_FORMAT_ND);
        TensorDesc yDesc(ParseDims(yShape), ParseDtype(yDtype), ACL_FORMAT_ND);
        TensorDesc scale1Desc(ParseDims(scale1Shape), ParseDtype(scale1Dtype), ACL_FORMAT_ND, ParseDims(scale1Stride));
        TensorDesc groupListDesc(ParseDims(groupListShape), ACL_INT64, ACL_FORMAT_ND);
        const vector<int64_t> groupVals = ParseI64List(groupListValues);
        if (!groupVals.empty()) {
            groupListDesc.Value(groupVals);
        }

        auto ut = OP_API_UT(aclnnQuantGroupedMatmulInplaceAdd,
                            INPUT(x1Desc, x2Desc, scale1Desc, scale2Desc, groupListDesc, yDesc, groupListType,
                                  groupSize),
                            OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
        if (checkRet) {
            EXPECT_EQ(ret, ParseStatus(expectRet)) << "case=" << caseName;
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string x1Shape;
    string x1Dtype;
    string x1Stride;
    string x2Shape;
    string x2Dtype;
    string x2Stride;
    string scale2Shape;
    string scale2Dtype;
    string yShape;
    string yDtype;
    string scale1Shape;
    string scale1Dtype;
    string scale1Stride;
    string groupListShape;
    string groupListValues;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
    string expectRet;
    bool checkRet = true;
};

vector<QGmmInplaceAddOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<QGmmInplaceAddOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    while (getline(in, line)) {
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

        QGmmInplaceAddOpApiCase c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.enable = ParseBool(Trim(cols[i++]));
        c.x1Shape = Trim(cols[i++]);
        c.x1Dtype = Trim(cols[i++]);
        c.x1Stride = Trim(cols[i++]);
        c.x2Shape = Trim(cols[i++]);
        c.x2Dtype = Trim(cols[i++]);
        c.x2Stride = Trim(cols[i++]);
        c.scale2Shape = Trim(cols[i++]);
        c.scale2Dtype = Trim(cols[i++]);
        c.yShape = Trim(cols[i++]);
        c.yDtype = Trim(cols[i++]);
        c.scale1Shape = Trim(cols[i++]);
        c.scale1Dtype = Trim(cols[i++]);
        c.scale1Stride = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.groupListType = stoll(Trim(cols[i++]));
        c.groupSize = stoll(Trim(cols[i++]));
        c.expectRet = Trim(cols[i++]);
        c.checkRet = ParseBool(Trim(cols[i++]));
        cases.emplace_back(c);
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<QGmmInplaceAddOpApiCase> &info)
{
    string name = info.param.caseName;
    for (char &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

class qgmm_inplace_add_opapi_csv_test : public testing::TestWithParam<QGmmInplaceAddOpApiCase> {};

TEST_P(qgmm_inplace_add_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    qgmm_inplace_add_opapi_csv,
    qgmm_inplace_add_opapi_csv_test,
    testing::ValuesIn(LoadCases(ResolveCsvPath("test_aclnn_quant_grouped_matmul_inplace_add.csv"))),
    BuildCaseName);

TEST(qgmm_inplace_add_opapi_direct_test, phase2_null_executor_path)
{
    // Cover phase2 API path: L2_DFX_PHASE_2 + CommonOpExecutorRun check branch.
    aclnnStatus ret = aclnnQuantGroupedMatmulInplaceAdd(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

} // namespace
