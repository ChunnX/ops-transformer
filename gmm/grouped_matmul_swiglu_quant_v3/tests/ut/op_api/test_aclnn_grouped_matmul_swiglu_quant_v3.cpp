/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Unit tests for the Ascend 950 MXFP8 weight-NZ V3 ACLNN interface.
 * The test matrix covers the public parameter surface and ACLNN descriptor
 * validation. It does not launch an NPU kernel.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

using namespace std;
using namespace op;

namespace {

constexpr int64_t kM = 8;
constexpr int64_t kN = 256;
constexpr int64_t kDefaultK = 128;

struct DescriptorCase {
    int64_t m = kM;
    int64_t k = kDefaultK;
    int64_t n = kN;
    bool transposeWeight = false;
    aclDataType xDtype = ACL_FLOAT8_E4M3FN;
    aclDataType weightDtype = ACL_FLOAT8_E4M3FN;
    aclDataType xScaleDtype = ACL_FLOAT8_E8M0;
    aclDataType weightScaleDtype = ACL_FLOAT8_E8M0;
    aclDataType groupListDtype = ACL_INT64;
    aclDataType outputDtype = ACL_FLOAT8_E4M3FN;
    aclDataType outputScaleDtype = ACL_FLOAT8_E8M0;
    aclFormat weightFormat = ACL_FORMAT_FRACTAL_NZ;
    int64_t mode = 0;
    bool activateLeft = true;
    double clampLimit = 7.0;
    double gluAlpha = 1.702;
    double gluBias = 1.0;
    int64_t groupListType = 0;
    int64_t dequantMode = 2;
    int64_t dequantDtype = 0;
    int64_t quantMode = 2;
    int64_t axis = -1;
    const char *roundMode = "rint";
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool duplicateWeight = false;
    bool duplicateWeightScale = false;
    bool wrongWeightStorage = false;
    bool withWeightAssist = false;
    bool withBias = false;
    bool withSmoothScale = false;
    bool withTuningConfig = false;
    bool useAscend950 = true;
};

struct DescriptorPack {
    TensorDesc x;
    TensorListDesc weight;
    TensorListDesc weightScale;
    TensorDesc xScale;
    TensorDesc groupList;
    TensorDesc output;
    TensorDesc outputScale;
    TensorDesc weightAssist;
    TensorDesc bias;
    TensorDesc smoothScale;
};

vector<int64_t> WeightStorage(const DescriptorCase &c)
{
    if (c.wrongWeightStorage) {
        return {1, 1, 1, 16, 32};
    }
    if (c.transposeWeight) {
        return {1, (c.k + 31) / 32, (c.n + 15) / 16, 16, 32};
    }
    return {1, (c.k + 15) / 16, (c.n + 31) / 32, 16, 32};
}

DescriptorPack MakeDescriptors(const DescriptorCase &c)
{
    const vector<int64_t> xShape = {c.m, c.k};
    const vector<int64_t> weightShape = c.transposeWeight ? vector<int64_t>{1, c.n, c.k}
                                                          : vector<int64_t>{1, c.k, c.n};
    const vector<int64_t> weightScaleShape =
        c.transposeWeight ? vector<int64_t>{1, c.n, (c.k + 63) / 64, 2}
                          : vector<int64_t>{1, (c.k + 63) / 64, c.n, 2};
    const vector<int64_t> xScaleShape = {c.m, (c.k + 63) / 64, 2};
    const vector<int64_t> outputShape = {c.m, c.n / 2};
    const vector<int64_t> outputScaleShape = {c.m, (c.n / 2 + 63) / 64, 2};
    const vector<int64_t> weightStrides = c.transposeWeight ? vector<int64_t>{c.n * c.k, 1, c.n}
                                                             : vector<int64_t>{c.n * c.k, c.n, 1};

    TensorDesc x = TensorDesc(xShape, c.xDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc weight = TensorDesc(weightShape, c.weightDtype, c.weightFormat, weightStrides, 0,
                                   WeightStorage(c))
                            .ValueRange(-2, 2);
    TensorDesc weightScale = TensorDesc(weightScaleShape, c.weightScaleDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc xScale = TensorDesc(xScaleShape, c.xScaleDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc groupList = TensorDesc({1}, c.groupListDtype, ACL_FORMAT_ND).Value(vector<int64_t>{c.m});
    TensorDesc output = TensorDesc(outputShape, c.outputDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc outputScale = TensorDesc(outputScaleShape, c.outputScaleDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc weightAssist = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc bias = TensorDesc({c.n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc smoothScale = TensorDesc({c.n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    return {x,
            c.duplicateWeight ? TensorListDesc(2, weight) : TensorListDesc({weight}),
            c.duplicateWeightScale ? TensorListDesc(2, weightScale) : TensorListDesc({weightScale}),
            xScale,
            groupList,
            output,
            outputScale,
            weightAssist,
            bias,
            smoothScale};
}

template <typename WeightAssistT, typename BiasT, typename SmoothScaleT>
aclnnStatus RunWorkspaceImpl(const DescriptorCase &c, const DescriptorPack &pack, const WeightAssistT &weightAssist,
                             const BiasT &bias, const SmoothScaleT &smoothScale, aclIntArray *tuningConfig)
{
    auto ut = OP_API_UT(
        aclnnGroupedMatmulSwigluQuantWeightNzV3,
        INPUT(pack.x, pack.weight, pack.weightScale, weightAssist, bias, pack.xScale, smoothScale, pack.groupList,
              c.dequantMode, c.dequantDtype, c.quantMode, c.groupListType, tuningConfig, -1, c.activateLeft, c.mode,
              c.clampLimit, c.gluAlpha, c.gluBias, c.axis, c.roundMode, c.scaleAlg, c.dstTypeMax),
        OUTPUT(pack.output, pack.outputScale));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

aclnnStatus RunWorkspace(const DescriptorCase &c)
{
    SetPlatformSocVersion(c.useAscend950 ? SocVersion::ASCEND950 : SocVersion::ASCEND910B);
    auto pack = MakeDescriptors(c);
    vector<int64_t> tuningValues = {0};
    aclIntArray *tuningConfig = c.withTuningConfig ? aclCreateIntArray(tuningValues.data(), tuningValues.size()) : nullptr;
    TensorListDesc weightAssistDesc({pack.weightAssist});

    aclnnStatus ret = ACL_SUCCESS;
    if (c.withWeightAssist) {
        ret = RunWorkspaceImpl(c, pack, weightAssistDesc, nullptr, nullptr, tuningConfig);
    } else if (c.withBias) {
        ret = RunWorkspaceImpl(c, pack, nullptr, pack.bias, nullptr, tuningConfig);
    } else if (c.withSmoothScale) {
        ret = RunWorkspaceImpl(c, pack, nullptr, nullptr, pack.smoothScale, tuningConfig);
    } else {
        ret = RunWorkspaceImpl(c, pack, nullptr, nullptr, nullptr, tuningConfig);
    }
    // CANN 9.2's op_api UT allocator invalidates the array after a rejected
    // call.  Do not destroy that rejected optional descriptor a second time;
    // the process-level UT owns the short-lived object until exit.
    if (tuningConfig != nullptr && !c.withTuningConfig) {
        aclDestroyIntArray(tuningConfig);
    }
    return ret;
}

string MakeName(const testing::TestParamInfo<DescriptorCase> &info)
{
    const auto &c = info.param;
    return string(c.transposeWeight ? "ZN" : "NZ") + "_M" + to_string(c.mode) + "_K" + to_string(c.k) +
           "_S" + to_string(c.scaleAlg) + "_L" + (c.activateLeft ? "1" : "0") + "_G" +
           to_string(c.groupListType);
}

vector<DescriptorCase> MakeNormalCases()
{
    vector<DescriptorCase> cases;
    for (bool transpose : {false, true}) {
        for (int64_t mode = 0; mode <= 3; ++mode) {
            for (int64_t k : {64, 128}) {
                for (int64_t scaleAlg : {0, 1}) {
                    for (bool activateLeft : {false, true}) {
                        for (int64_t groupListType : {0, 1}) {
                            DescriptorCase c;
                            c.transposeWeight = transpose;
                            c.mode = mode;
                            c.k = k;
                            c.scaleAlg = scaleAlg;
                            c.activateLeft = activateLeft;
                            c.groupListType = groupListType;
                            cases.emplace_back(c);
                        }
                    }
                }
            }
        }
    }
    return cases;
}

class GroupedMatmulSwigluQuantV3OpApiNormalTest : public testing::TestWithParam<DescriptorCase> {};

TEST_P(GroupedMatmulSwigluQuantV3OpApiNormalTest, AllSupportedParameterCombinations)
{
    EXPECT_EQ(RunWorkspace(GetParam()), ACL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V3_NORMAL, GroupedMatmulSwigluQuantV3OpApiNormalTest,
                         testing::ValuesIn(MakeNormalCases()), MakeName);

TEST(GroupedMatmulSwigluQuantV3OpApi, DirectLaunchRejectsNullExecutor)
{
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
}

TEST(GroupedMatmulSwigluQuantV3OpApi, WorkspaceRejectsNullRequiredInputs)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, nullptr, -1,
                  true, 0, 7.0, 1.702, 1.0, -1, "rint", 0, 0.0, nullptr, nullptr, &workspaceSize, &executor),
              ACL_SUCCESS);
}

TEST(GroupedMatmulSwigluQuantV3OpApi, Mode0AllowsZeroClampLimit)
{
    DescriptorCase c;
    c.mode = 0;
    c.clampLimit = 0.0;
    EXPECT_EQ(RunWorkspace(c), ACL_SUCCESS);
}

TEST(GroupedMatmulSwigluQuantV3OpApi, EmptyRoundModeUsesRint)
{
    DescriptorCase c;
    c.roundMode = "";
    EXPECT_EQ(RunWorkspace(c), ACL_SUCCESS);
}

class GroupedMatmulSwigluQuantV3OpApiInvalidTest : public testing::Test {};

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsNonNzWeight)
{
    DescriptorCase c;
    c.weightFormat = ACL_FORMAT_ND;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsWrongDtypes)
{
    const vector<pair<string, DescriptorCase>> cases = {
        {"x", [] { DescriptorCase c; c.xDtype = ACL_FLOAT16; return c; }()},
        {"weight", [] { DescriptorCase c; c.weightDtype = ACL_FLOAT8_E5M2; return c; }()},
        {"xScale", [] { DescriptorCase c; c.xScaleDtype = ACL_FLOAT16; return c; }()},
        {"weightScale", [] { DescriptorCase c; c.weightScaleDtype = ACL_FLOAT16; return c; }()},
        {"groupList", [] { DescriptorCase c; c.groupListDtype = ACL_INT32; return c; }()},
        {"output", [] { DescriptorCase c; c.outputDtype = ACL_FLOAT16; return c; }()},
        {"outputScale", [] { DescriptorCase c; c.outputScaleDtype = ACL_FLOAT16; return c; }()},
    };
    for (const auto &item : cases) {
        SCOPED_TRACE(item.first);
        EXPECT_EQ(RunWorkspace(item.second), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsWrongShapeOrStorage)
{
    DescriptorCase wrongStorage;
    wrongStorage.wrongWeightStorage = true;
    EXPECT_EQ(RunWorkspace(wrongStorage), ACLNN_ERR_PARAM_INVALID);

    DescriptorCase wrongN;
    wrongN.n = 255;
    EXPECT_EQ(RunWorkspace(wrongN), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsMultipleWeightTensors)
{
    DescriptorCase c;
    c.duplicateWeight = true;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
    c = DescriptorCase{};
    c.duplicateWeightScale = true;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsUnsupportedGmmAndQuantAttributes)
{
    const vector<pair<string, DescriptorCase>> cases = {
        {"dequantMode", [] { DescriptorCase c; c.dequantMode = 0; return c; }()},
        {"dequantDtype", [] { DescriptorCase c; c.dequantDtype = 1; return c; }()},
        {"quantMode", [] { DescriptorCase c; c.quantMode = 0; return c; }()},
        {"groupListType", [] { DescriptorCase c; c.groupListType = 2; return c; }()},
        {"axis", [] { DescriptorCase c; c.axis = 0; return c; }()},
        {"roundMode", [] { DescriptorCase c; c.roundMode = "floor"; return c; }()},
        {"scaleAlg", [] { DescriptorCase c; c.scaleAlg = 2; return c; }()},
        {"dstTypeMax", [] { DescriptorCase c; c.dstTypeMax = 1.0; return c; }()},
    };
    for (const auto &item : cases) {
        SCOPED_TRACE(item.first);
        EXPECT_EQ(RunWorkspace(item.second), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsUnsupportedSwiGluAttributes)
{
    const vector<pair<string, DescriptorCase>> cases = {
        {"swigluMode", [] { DescriptorCase c; c.mode = 4; return c; }()},
        {"clampLimit", [] { DescriptorCase c; c.mode = 1; c.clampLimit = 0.0; return c; }()},
        {"clampNaN", [] { DescriptorCase c; c.clampLimit = numeric_limits<double>::quiet_NaN(); return c; }()},
        {"gluAlphaNaN", [] { DescriptorCase c; c.gluAlpha = numeric_limits<double>::quiet_NaN(); return c; }()},
        {"gluBiasInf", [] { DescriptorCase c; c.gluBias = numeric_limits<double>::infinity(); return c; }()},
    };
    for (const auto &item : cases) {
        SCOPED_TRACE(item.first);
        EXPECT_EQ(RunWorkspace(item.second), ACLNN_ERR_PARAM_INVALID);
    }

    auto c = DescriptorCase{};
    auto pack = MakeDescriptors(c);
    auto ut = OP_API_UT(
        aclnnGroupedMatmulSwigluQuantWeightNzV3,
        INPUT(pack.x, pack.weight, pack.weightScale, nullptr, nullptr, pack.xScale, nullptr, pack.groupList, 2, 0, 2,
              0, nullptr, 0, true, 0, 7.0, 1.702, 1.0, -1, "rint", 0, 0.0),
        OUTPUT(pack.output, pack.outputScale));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(GroupedMatmulSwigluQuantV3OpApiInvalidTest, RejectsUnsupportedOptionalInputs)
{
    DescriptorCase c;
    c.withWeightAssist = true;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
    c = DescriptorCase{};
    c.withBias = true;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
    c = DescriptorCase{};
    c.withSmoothScale = true;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
    c = DescriptorCase{};
    c.withTuningConfig = true;
    EXPECT_EQ(RunWorkspace(c), ACLNN_ERR_PARAM_INVALID);
}

} // namespace
