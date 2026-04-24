/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

static void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos < input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

string JoinPath(const string &dir, const string &file)
{
    if (dir.empty()) {
        return file;
    }
    char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

constexpr size_t kPathBufferSize = 4096;
constexpr size_t kGroupedMatmulCsvColumnCount = 31;
constexpr size_t kGroupedMatmulCsvExtendedColumnCount = 35;

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    string exePath(path, len);
    auto pos = exePath.find_last_of("\\/");
    return pos == string::npos ? string(".\\") : exePath.substr(0, pos + 1);
#else
    char path[kPathBufferSize] = {0};
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

string GetCurrentFileDir()
{
    string filePath = __FILE__;
    auto pos = filePath.find_last_of("\\/");
    return pos == string::npos ? string(".") : filePath.substr(0, pos);
}

string GetCwd()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetCurrentDirectoryA(MAX_PATH, path);
    return len == 0 ? string(".") : string(path, len);
#else
    char path[kPathBufferSize] = {0};
    if (getcwd(path, sizeof(path)) == nullptr) {
        return ".";
    }
    return string(path);
#endif
}

static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) return 0;
    if (a <= 0) return 0;
    return (a - 1) / b + 1;
}

aclDataType ParseDtype(const string &dtype)
{
    static const map<string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},             {"FLOAT16", ACL_FLOAT16},         {"BF16", ACL_BF16},
        {"INT8", ACL_INT8},               {"INT4", ACL_INT4},               {"INT32", ACL_INT32},
        {"INT64", ACL_INT64},             {"UINT64", ACL_UINT64},           {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2}, {"FLOAT8_E8M0", ACL_FLOAT8_E8M0}, {"HIFLOAT8", ACL_HIFLOAT8},
        {"FLOAT4_E2M1", ACL_FLOAT4_E2M1},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> formatMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
        {"FRACTAL_NZ_C0_32", ACL_FORMAT_FRACTAL_NZ_C0_32},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ACL_FORMAT_ND : it->second;
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

class l2_GroupedMatmulFinalizeRoutingWeightNzV2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test TearDown" << endl;
    }
};

// TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_add_value)
// {
//     int64_t m = 192;
//     int64_t k = 2048;
//     int64_t n = 7168;
//     int64_t e = 4;
//     int64_t bs = 24;
//     int64_t bsdp = 8;
//     TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc x2_desc =
//         TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
//     TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
//     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
//     TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     vector<int64_t> tuningConfigVal = { m / e };
//     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
//     auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
//                         INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
//                               groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
//                               1, tuningConfig),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_transpose_test)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { m / e };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, true, true,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_shared_input_null)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { m / e };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, nullptr, nullptr, 0, 1.0, 0, false, false, 1,
                              tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_bias_not_null)
// {
//     int64_t m = 192;
//     int64_t k = 2048;
//     int64_t n = 7168;
//     int64_t e = 4;
//     int64_t bs = 24;
//     int64_t bsdp = 8;
//     TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
//     TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
//     TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc bias_desc = TensorDesc({e, n}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
//     vector<int64_t> tuningConfigVal = { m / e };
//     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
//     auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
//                         INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc,
//                               groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
//                               1, tuningConfig),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_x2_not_nz)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { m / e };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_dtype_not_0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 1, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_scale_noteq_e)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e + 1, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);\
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_perTokenScale_noteq_m)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_shared_input_more_than_bs)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_group_list_type_not_1)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_view_shape_0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({0, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_x12_viewdims_wrong)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_shareinput_outputBS_noeq)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp + 1, n + 1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_logit_1stdim_noeq_xMdim)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_rowindex1stdim_noeq_xMdim)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_groupListdim_wrong)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_weightview0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({0}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_normal_case_nvalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7000;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({0}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_not_support_k_shape_case)
{
    int64_t m = 192;
    int64_t k = 4098;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_x1_type_not_same_with_x2)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_scale_shape_invalid_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e + 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_k_n_not_support_w8a8_case)
{
    int64_t m = 192;
    int64_t k = 2048 - 1;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 0 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_tuning_config_invalid_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { -5 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_offset_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc offset_desc = TensorDesc({e, 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, offset_desc, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_x2_not_int32_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    // NOTE: This test case is adapted from 'l2_GroupedMatmulFinalizeRoutingV3_test',
    // not sure what the original intention was.
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_scale_not_int64_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_scale_dim_not_3_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend910B2_test_w4a8_pertoken_scale_null_case)
// {
//     int64_t m = 192;
//     int64_t k = 2048;
//     int64_t n = 7168;
//     int64_t e = 4;
//     int64_t bs = 24;
//     int64_t bsdp = 8;
//     int64_t quantGroupSize = 256;

//     TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc x2_desc =
//         TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 64, k / 16, 16, 8}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
//     TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
//     TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     vector<int64_t> tuningConfigVal = { 10 };
//     aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
//     auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
//                         INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, nullptr, groupList_desc,
//                               shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, tuningConfig),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend950_test_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend950_test_invalid_case_scale_nullptr)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, nullptr, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet,ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test, ascend950_test_invalid_case_dim_mismatch)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1, 
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== MxA8W4 Test Cases (Ascend950 Only) ====================
struct GmmfrMxa8W4TestParam {
    void Run() const
    {
        SetupPlatformForCase(platform);
        int64_t ceil_k_64 = CeilDiv(k, 64);

        TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
        TensorDesc x2_desc;
        if (!transposeX2) {
            x2_desc = TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32,
                                 {static_cast<int64_t>(n) * k, 1, k}, 0, 
                                 {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
        } else {
            x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, 
                                 {}, 0,
                                 {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
        }
        TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
        TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
        TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
        TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
        TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
        TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc bias_desc = TensorDesc({e, n}, ACL_BF16, ACL_FORMAT_ND);

        auto biasOptional = nullptr;
        auto sharedInputOptional = nullptr;

        bool transposeX1 = false;

        uint64_t workspace_size = 0;
        aclnnStatus aclRet;

        if (hasBias && hasSharedInput) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr,
                                      perTokenScale_desc, groupList_desc, shared_input_desc,
                                      logits_desc, row_index_desc, 0, 1.0, sharedInputOffset, transposeX1, transposeX2, 
                                      groupListType, nullptr),
                                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (hasBias && !hasSharedInput) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr,
                                      perTokenScale_desc, groupList_desc, sharedInputOptional,
                                      logits_desc, row_index_desc, 0, 1.0, sharedInputOffset, transposeX1, transposeX2, 
                                      groupListType, nullptr),
                                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (!hasBias && hasSharedInput) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1_desc, x2_desc, scale_desc, biasOptional, nullptr, nullptr, nullptr,
                                      perTokenScale_desc, groupList_desc, shared_input_desc,
                                      logits_desc, row_index_desc, 0, 1.0, sharedInputOffset, transposeX1, transposeX2, 
                                      groupListType, nullptr),
                                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (!hasBias && !hasSharedInput) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1_desc, x2_desc, scale_desc, biasOptional, nullptr, nullptr, nullptr,
                                      perTokenScale_desc, groupList_desc, sharedInputOptional,
                                      logits_desc, row_index_desc, 0, 1.0, sharedInputOffset, transposeX1, transposeX2, 
                                      groupListType, nullptr),
                                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        }

        if (expectRet) {
            EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        } else {
            EXPECT_NE(aclRet, ACLNN_SUCCESS);
        }
    }

    static std::string ResolveUtCaseFilePath(const std::string& csvFileName)
    {
        const string cwd = GetCwd();
        const std::string defaultUtCaseRootPath = "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api";

        vector<string> candidates;
        const char *envPath = std::getenv("GMMFR_MXA8W4_UT_CASE_PATH");
        if (envPath != nullptr) {
            candidates.emplace_back(envPath);
        }
        candidates.emplace_back(JoinPath(GetExeDirPath(), csvFileName));
        candidates.emplace_back(JoinPath(GetCurrentFileDir(), csvFileName));
        candidates.emplace_back(JoinPath(cwd, csvFileName));
        candidates.emplace_back(JoinPath(cwd, "../../../../../" + defaultUtCaseRootPath + "/" + csvFileName));
        candidates.emplace_back(JoinPath(defaultUtCaseRootPath, csvFileName));

        for (const auto &path : candidates) {
            if (std::filesystem::exists(path)) {
                return path;
            }
        }
        return candidates.front();
    }

    static std::vector<GmmfrMxa8W4TestParam> GetMxa8W4Params(const std::string& csvPath)
    {
        std::vector<GmmfrMxa8W4TestParam> params;
        std::ifstream csvData(csvPath, std::ios::in);
        EXPECT_TRUE(csvData.is_open()) << "Failed to open CSV file: " << csvPath;

        std::string line;
        bool isHeader = true;
        while (std::getline(csvData, line)) {
            if (isHeader) {
                isHeader = false;
                continue;
            }
            if (line.empty() || line[0] == '\r' || line[0] == '\n') {
                continue;
            }
            std::vector<std::string> fields;
            SplitStr2Vec(line, ",", fields);
            if (fields.size() < 15) {
                std::cout << "Skipping line with " << fields.size() << " fields: " << line << std::endl;
                continue;
            }

            GmmfrMxa8W4TestParam param;
            try {
                size_t idx = 0;
                param.caseName = fields[idx++];
                param.platform = fields[idx++];
                param.enable = (fields[idx++] == "true");
                if (!param.enable) {
                    continue;
                }
                param.m = std::stol(fields[idx++]);
                param.k = std::stol(fields[idx++]);
                param.n = std::stol(fields[idx++]);
                param.e = std::stol(fields[idx++]);
                param.bs = std::stol(fields[idx++]);
                param.bsdp = std::stol(fields[idx++]);
                param.sharedInputOffset = std::stol(fields[idx++]);
                param.groupListType = std::stol(fields[idx++]);
                param.hasBias = (fields[idx++] == "true");
                param.transposeX2 = (fields[idx++] == "true");
                param.hasSharedInput = (fields[idx++] == "true");
                param.expectRet = (fields[idx++] == "true");
                param.comment = fields[idx++];
                params.push_back(param);
            } catch (const std::exception &e) {
                std::cout << "Error parsing line: " << line << " error: " << e.what() << std::endl;
            }
        }
        EXPECT_FALSE(params.empty()) << "No valid cases parsed from CSV: " << csvPath;
        return params;
    }

    std::string caseName;
    std::string platform;
    bool enable;

    int64_t m;
    int64_t k;
    int64_t n;
    int64_t e;
    int64_t bs;
    int64_t bsdp;
    int64_t sharedInputOffset;
    int64_t groupListType;
    bool hasBias;
    // transpose输入属性的值，传入false时通过stride构造transpose
    bool transposeX2;
    bool hasSharedInput;

    bool expectRet;
    std::string comment;
};

class GmmfrMxa8W4ParamTest : public testing::TestWithParam<GmmfrMxa8W4TestParam> {};

TEST_P(GmmfrMxa8W4ParamTest, gmmfr_weight_nz_v2_mxa8w4_test)
{
    GetParam().Run();
}

INSTANTIATE_TEST_CASE_P(MxA8W4_CSV, 
    GmmfrMxa8W4ParamTest, 
    testing::ValuesIn(GmmfrMxa8W4TestParam::GetMxa8W4Params(GmmfrMxa8W4TestParam::ResolveUtCaseFilePath("test_gmmfr_mxa8w4.csv")))
);

class l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950 SetUp" << endl;
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950 TearDown" << endl;
    }
};

// ==================== MxA8W4 Negative Test Cases ====================

// Validates: scale dimension must be 4 (e, n, ceil_k_64, 2), not 3
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    // scale维度错误：3维而非4维
    TensorDesc scale_desc = TensorDesc({e, 1, n}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: x2 tensor must use FRACTAL_NZ_C0_32 format, not ND
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x2_format_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // x2格式错误：ND而非NZ
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: sharedInputOffset + bsdp must not exceed output batch size
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_shared_input_offset_overflow)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 20;
    int64_t sharedInputOffset = 10;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    // sharedInputOffset + bsdp = 10 + 20 = 30 > outputBS = 24
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, sharedInputOffset,
                              false, true, 1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: k dimension must be 64-aligned
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_k_not_aligned)
{
    int64_t m = 192;
    int64_t k = 2050;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: n dimension must be 16-aligned
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_n_not_aligned)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7170;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: number of experts must not exceed 1024
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_experts_exceeded)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 1025;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: shared input batch size must not exceed output batch size
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_bsdp_exceed_output)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 30;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: perTokenScale must not be nullptr (required parameter)
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_pertoken_scale_nullptr)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Validates: x1 dtype must be ACL_FLOAT8_E4M3FN, not ACL_INT8
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x1_dtype_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: x2 dtype must be ACL_FLOAT4_E2M1, not ACL_INT8
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x2_dtype_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc =
        TensorDesc({e, n, k}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0, {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: scale dtype must be ACL_FLOAT8_E8M0, not ACL_FLOAT
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_dtype_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: x1 must be 2D tensor, not 3D
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x1_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, 1, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: perTokenScale must be 3D (m, ceil_k_64, 2), not 2D
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_pertoken_scale_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, 32}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: groupList must be 1D tensor, not 2D
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_group_list_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e, 1}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: output must be 2D tensor, not 3D
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_out_dim_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: scale shape must match (e, n, ceil_k_64, 2), not mismatched
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64 + 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: perTokenScale shape must match (m, ceil_k_64, 2), not mismatched
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_pertoken_scale_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64 + 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: bias shape must match (e, n), not mismatched
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_bias_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({e, n + 1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: x2 shape dimensions must be consistent with NZ storage format
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x2_shape_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: x1 must use ND format, not FRACTAL_NZ_C0_32
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_x1_format_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ_C0_32);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: scale must use ND format, not FRACTAL_NZ_C0_32
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_scale_format_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_FRACTAL_NZ_C0_32);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1,
                              nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Validates: torch pathway with non-transpose stride should fail
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_torch_pathway_non_transpose_invalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // 非转置的连续stride: (k*n, n, 1)
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {static_cast<int64_t>(k * n), n, 1}, 0,
                   {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    // transposeX2=false 且stride不是转置模式, 应该失败
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Torch通路测试用例: viewShape=(e,k,n), stride=(e*k,1,k)表示转置, transposeX2=false
// 注意: UT框架对TensorDesc stride支持有限, 以下用例暂时注释
// Torch通路的核心逻辑(通过stride检测转置)已通过torch_pathway_non_transpose_invalid负用例间接验证

// Torch通路正常用例: 验证通过stride检测转置时返回ACLNN_SUCCESS
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_torch_pathway_normal)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {static_cast<int64_t>(k * n), 1, k}, 0,
                   {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_torch_pathway_normal_m0)
{
    int64_t m = 0;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {static_cast<int64_t>(k * n), 1, k}, 0,
                   {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_torch_pathway_normal_n0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 0;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {static_cast<int64_t>(k * n), 1, k}, 0,
                   {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// Torch通路带bias用例: 验证通过stride检测转置且带bias时返回ACLNN_SUCCESS
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_torch_pathway_with_bias)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {static_cast<int64_t>(k * n), 1, k}, 0,
                   {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false,
                              1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 无sharedInput用例: 验证sharedInput为nullptr时返回ACLNN_SUCCESS
TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950, ascend950_test_mxa8w4_no_shared_input)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, n, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, {}, 0,
                                    {e, CeilDiv(k, 32), CeilDiv(n, 16), 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m, ceil_k_64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, perTokenScale_desc,
                              groupList_desc, nullptr, logits_desc, row_index_desc, 0, 1.0, 0, false, true, 1, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
