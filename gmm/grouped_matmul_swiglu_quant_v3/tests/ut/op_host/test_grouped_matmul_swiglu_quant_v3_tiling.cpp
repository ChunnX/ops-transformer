/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host tiling UT for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v3_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v3_tiling_key.h"
#include "gmm_csv_ge_parse_utils.h"
#include "platform/platform_infos_def.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {

struct TensorDescParam {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
};

constexpr size_t kInputCount = 8;
constexpr size_t kOutputCount = 2;
constexpr uint32_t kAicNum = 24;
constexpr uint32_t kAivNum = 48;
constexpr uint64_t kUbSize = 201326592;
constexpr uint64_t kL1Size = 524288;
constexpr uint64_t kL2Size = 33554432;
constexpr uint64_t kL0CSize = 131072;
constexpr uint64_t kL0ASize = 65536;
constexpr uint64_t kL0BSize = 65536;

struct TilingCase {
    bool transpose = false;
    int64_t mode = 0;
    int64_t k = 128;
    int64_t scaleAlg = 0;
    int64_t groupListType = 0;
    bool expectSuccess = true;
    uint64_t expectedKey = GMM_SWIGLU_QUANT_V3_TENSOR_API_NZ;

    optiling::GMMCompileInfo CompileInfo() const
    {
        return {kAicNum,
                kAivNum,
                kUbSize,
                kL1Size,
                kL2Size,
                kL0CSize,
                kL0ASize,
                kL0BSize,
                platform_ascendc::SocVersion::ASCEND950,
                NpuArch::DAV_3510};
    }

    gert::TilingContextPara BuildContext(optiling::GMMCompileInfo *compileInfo) const
    {
        *compileInfo = CompileInfo();
        const int64_t m = 8;
        const int64_t n = 256;
        const int64_t kScale = (k + 63) / 64;
        const vector<int64_t> weightView = transpose ? vector<int64_t>{1, n, k} : vector<int64_t>{1, k, n};
        const vector<int64_t> weightStorage =
            transpose ? vector<int64_t>{1, (k + 31) / 32, (n + 15) / 16, 16, 32}
                      : vector<int64_t>{1, (k + 15) / 16, (n + 31) / 32, 16, 32};
        const vector<int64_t> weightScale =
            transpose ? vector<int64_t>{1, n, kScale, 2} : vector<int64_t>{1, kScale, n, 2};

        array<vector<TensorDescParam>, kInputCount> inputs;
        inputs[0] = {{{m, k}, {m, k}, DT_FLOAT8_E4M3FN, FORMAT_ND}};
        inputs[1] = {{{m, kScale, 2}, {m, kScale, 2}, DT_FLOAT8_E8M0, FORMAT_ND}};
        inputs[2] = {{{1}, {1}, DT_INT64, FORMAT_ND}};
        inputs[3] = {{weightView, weightStorage, DT_FLOAT8_E4M3FN, FORMAT_FRACTAL_NZ}};
        inputs[4] = {{weightScale, weightScale, DT_FLOAT8_E8M0, FORMAT_ND}};
        // The V3 interface deliberately supports only nullptr/empty optional inputs.
        inputs[5] = {};
        inputs[6] = {};
        inputs[7] = {};
        array<vector<TensorDescParam>, kOutputCount> outputs;
        outputs[0] = {{{m, n / 2}, {m, n / 2}, DT_FLOAT8_E4M3FN, FORMAT_ND}};
        outputs[1] = {{{m, (n / 2 + 63) / 64, 2}, {m, (n / 2 + 63) / 64, 2}, DT_FLOAT8_E8M0, FORMAT_ND}};

        vector<uint32_t> inputInstanceNum;
        vector<uint32_t> outputInstanceNum;
        auto build = [](const auto &groups, vector<uint32_t> &instanceNum) {
            vector<gert::TilingContextPara::TensorDescription> descs;
            for (const auto &group : groups) {
                instanceNum.push_back(static_cast<uint32_t>(group.size()));
                for (const auto &tensor : group) {
                    descs.emplace_back(ops::ut::MakeGertStorageShape(tensor.originDims, tensor.storageDims),
                                       tensor.dtype, tensor.format);
                }
            }
            return descs;
        };

        return gert::TilingContextPara(
            "GroupedMatmulSwigluQuantV3", build(inputs, inputInstanceNum), build(outputs, outputInstanceNum),
            {
                {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transpose)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0})},
                {"activate_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                {"activate_left", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                {"swiglu_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode)},
                {"clamp_limit", Ops::Transformer::AnyValue::CreateFrom<float>(7.0F)},
                {"glu_alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.702F)},
                {"glu_bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
                {"axis", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                {"round_mode", Ops::Transformer::AnyValue::CreateFrom<string>("rint")},
                {"scale_alg", Ops::Transformer::AnyValue::CreateFrom<int64_t>(scaleAlg)},
                {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(0.0F)},
            },
            inputInstanceNum, outputInstanceNum, compileInfo, "Ascend950", compileInfo->aicNum, compileInfo->ubSize);
    }

    void Run() const
    {
        auto compileInfo = CompileInfo();
        auto context = BuildContext(&compileInfo);
        if (!expectSuccess) {
            ExecuteTestCase(context, ge::GRAPH_FAILED);
            return;
        }
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey, expectedKey);
        ASSERT_GE(info.tilingDataSize, sizeof(optiling::GMMSwigluQuantV3TilingDataParams));
        const auto *data = reinterpret_cast<const optiling::GMMSwigluQuantV3TilingDataParams *>(info.tilingData.get());
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(data->gmmSwigluQuantParams.swigluMode, mode);
        EXPECT_EQ(data->gmmSwigluQuantParams.scaleAlg, static_cast<uint8_t>(scaleAlg));
        EXPECT_EQ(data->gmmSwigluQuantParams.axis, -1);
    }
};

vector<TilingCase> MakeCases()
{
    vector<TilingCase> cases;
    for (bool transpose : {false, true}) {
        for (int64_t mode = 0; mode <= 3; ++mode) {
            for (int64_t k : {64, 128}) {
                for (int64_t scaleAlg : {0, 1}) {
                    TilingCase c;
                    c.transpose = transpose;
                    c.mode = mode;
                    c.k = k;
                    c.scaleAlg = scaleAlg;
                    c.expectedKey = mode != 1 && k % 128 == 0
                                         ? (transpose ? GMM_SWIGLU_QUANT_V3_TENSOR_API_ZN
                                                      : GMM_SWIGLU_QUANT_V3_TENSOR_API_NZ)
                                         : (transpose ? GMM_SWIGLU_QUANT_V3_CGMCT_ZN
                                                      : GMM_SWIGLU_QUANT_V3_CGMCT_NZ);
                    cases.emplace_back(c);
                }
            }
        }
    }
    return cases;
}

string MakeName(const testing::TestParamInfo<TilingCase> &info)
{
    const auto &c = info.param;
    return string(c.transpose ? "ZN" : "NZ") + "_M" + to_string(c.mode) + "_K" + to_string(c.k) + "_S" +
           to_string(c.scaleAlg);
}

class GroupedMatmulSwigluQuantV3TilingTest : public testing::TestWithParam<TilingCase> {};

TEST_P(GroupedMatmulSwigluQuantV3TilingTest, RouteAndEncodeParameters)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMSQ_V3_TILING, GroupedMatmulSwigluQuantV3TilingTest, testing::ValuesIn(MakeCases()),
                         MakeName);

TEST(GroupedMatmulSwigluQuantV3TilingTest, RejectsNullRequiredInput)
{
    TilingCase c;
    auto compileInfo = c.CompileInfo();
    auto context = c.BuildContext(&compileInfo);
    // The first input is x. Clear its instance count by rebuilding a context
    // with no tensor descriptions; the registered tiler must fail safely.
    gert::TilingContextPara nullContext(
        "GroupedMatmulSwigluQuantV3", {}, {}, {}, vector<uint32_t>(kInputCount, 0), vector<uint32_t>(kOutputCount, 0),
        &compileInfo, "Ascend950", compileInfo.aicNum, compileInfo.ubSize);
    ExecuteTestCase(nullContext, ge::GRAPH_FAILED);
}

TEST(GroupedMatmulSwigluQuantV3TilingTest, RejectsInvalidSwiGluMode)
{
    TilingCase c;
    c.mode = 4;
    c.expectSuccess = false;
    auto compileInfo = c.CompileInfo();
    auto context = c.BuildContext(&compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(GroupedMatmulSwigluQuantV3TilingTest, RejectsScaleAlgTwo)
{
    TilingCase c;
    c.scaleAlg = 2;
    c.expectSuccess = false;
    auto compileInfo = c.CompileInfo();
    auto context = c.BuildContext(&compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

} // namespace
