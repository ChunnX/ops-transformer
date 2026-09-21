/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_weight_nz_v3_tiling_data.h"

namespace {
struct RouteCase {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t e;
    bool trans;
    int64_t key;
};

constexpr int64_t V3_CGMCT_NZ_KEY = 32;
constexpr int64_t V3_CGMCT_ZN_KEY = 33;
constexpr int64_t V3_TENSOR_API_NZ_KEY = 48;
constexpr int64_t V3_TENSOR_API_ZN_KEY = 49;

gert::TilingContextPara MakeContext(const RouteCase& c, optiling::GMMSwigluV2CompileInfo* info, int scaleAlg,
                                    int aiv = 64, int64_t swigluMode = 2)
{
    using Desc = gert::TilingContextPara::TensorDescription;
    using Value = Ops::Transformer::AnyValue;
    const int64_t s = (c.k + 63) / 64;
    const int64_t h = c.n / 2;
    const auto tensor = [](std::vector<int64_t> dims, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(dims, dims), dtype, ge::FORMAT_ND);
    };
    const std::vector<int64_t> w = c.trans ? std::vector<int64_t>{c.e, c.n, c.k} :
                                           std::vector<int64_t>{c.e, c.k, c.n};
    const std::vector<int64_t> ws = c.trans ? std::vector<int64_t>{c.e, c.n, s, 2} :
                                            std::vector<int64_t>{c.e, s, c.n, 2};
    const std::vector<int64_t> storage = c.trans ?
        std::vector<int64_t>{c.e, (c.k + 31) / 32, (c.n + 15) / 16, 16, 32} :
        std::vector<int64_t>{c.e, (c.n + 31) / 32, (c.k + 15) / 16, 16, 32};
    info->aicNum_ = 32;
    info->aivNum_ = static_cast<uint32_t>(aiv);
    info->ubSize_ = 262144;
    info->l1Size_ = 524288;
    info->l0ASize_ = 65536;
    info->l0BSize_ = 65536;
    info->l0CSize_ = 262144;
    info->npuArch_ = static_cast<int32_t>(NpuArch::DAV_3510);
    info->supportL12BtBf16 = true;
    gert::TilingContextPara context("GroupedMatmulSwigluQuantV2",
        {tensor({c.m, c.k}, ge::DT_FLOAT8_E4M3FN), tensor({c.m, s, 2}, ge::DT_FLOAT8_E8M0),
         tensor({c.e}, ge::DT_INT64),
         Desc(ops::ut::MakeGertStorageShape(w, storage), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ),
         tensor(ws, ge::DT_FLOAT8_E8M0)},
        {tensor({c.m, h}, ge::DT_FLOAT8_E4M3FN), tensor({c.m, (h + 63) / 64, 2}, ge::DT_FLOAT8_E8M0)},
        {{"dequant_mode", Value::CreateFrom<int64_t>(2)},
         {"dequant_dtype", Value::CreateFrom<int64_t>(0)},
         {"quant_mode", Value::CreateFrom<int64_t>(2)},
         {"quant_dtype", Value::CreateFrom<int64_t>(ge::DT_FLOAT8_E4M3FN)},
         {"transpose_weight", Value::CreateFrom<bool>(c.trans)},
         {"group_list_type", Value::CreateFrom<int64_t>(0)},
         {"tuning_config", Value::CreateFrom<std::vector<int64_t>>({})},
         {"swiglu_mode", Value::CreateFrom<int64_t>(swigluMode)},
         {"clamp_limit", Value::CreateFrom<float>(7.0F)},
         {"glu_alpha", Value::CreateFrom<float>(1.702F)},
         {"glu_bias", Value::CreateFrom<float>(1.0F)},
         {"round_mode", Value::CreateFrom<std::string>("rint")},
         {"scale_alg", Value::CreateFrom<int64_t>(scaleAlg)},
         {"dst_type_max", Value::CreateFrom<float>(0.0F)}},
        {1, 1, 1, 1, 1, 0, 0, 0}, {1, 1}, info, "Ascend950", 32, 262144);
    context.socInfoString_ = R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":)" +
        std::to_string(aiv) + R"(,"UB_SIZE":262144,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144}})";
    return context;
}

const ::GMMSwigluQuantV3TilingDataParams* GetTilingData(const TilingInfo& result)
{
    EXPECT_EQ(result.tilingDataSize, sizeof(::GMMSwigluQuantV3TilingDataParams));
    return reinterpret_cast<const ::GMMSwigluQuantV3TilingDataParams*>(result.tilingData.get());
}

TEST(GroupedMatmulSwigluQuantV3Route, V2CapabilityBoundaries)
{
    const std::vector<RouteCase> cases = {
        {512, 128, 128, 1, false, V3_CGMCT_NZ_KEY},
        {513, 128, 128, 1, false, V3_TENSOR_API_NZ_KEY},
        {128, 128, 128, 1, true, V3_CGMCT_ZN_KEY},
        {129, 128, 128, 1, true, V3_TENSOR_API_ZN_KEY},
        {1025, 128, 128, 2, false, V3_CGMCT_NZ_KEY},
        {1026, 128, 128, 2, false, V3_TENSOR_API_NZ_KEY},
        {257, 128, 128, 2, true, V3_CGMCT_ZN_KEY},
        {258, 128, 128, 2, true, V3_TENSOR_API_ZN_KEY},
        {1026, 128, 192, 2, false, V3_CGMCT_NZ_KEY},
        {258, 128, 192, 2, true, V3_CGMCT_ZN_KEY},
        {1026, 64, 128, 2, false, V3_CGMCT_NZ_KEY},
        {258, 64, 128, 2, true, V3_CGMCT_ZN_KEY},
        {1026, 192, 128, 2, false, V3_CGMCT_NZ_KEY},
        {258, 192, 128, 2, true, V3_CGMCT_ZN_KEY},
        {3000, 4096, 8192, 3, false, V3_TENSOR_API_NZ_KEY},
        {3000, 4096, 8192, 3, true, V3_TENSOR_API_ZN_KEY}};
    for (const auto& c : cases) {
        for (int scaleAlg : {0, 1}) {
            SCOPED_TRACE(::testing::Message() << c.m << "," << c.k << "," << c.n << "," << c.e
                                            << ",trans=" << c.trans << ",alg=" << scaleAlg);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(c, &info, scaleAlg);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, c.key);
            EXPECT_EQ(result.blockNum, 32U);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, TensorApiUsesSymmetricL1Budget)
{
    for (bool trans : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "trans=" << trans);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {8192, 6144, 4096, 8, trans, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY}, &info, 0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY);
        const auto* tilingData = GetTilingData(result);
        ASSERT_NE(tilingData, nullptr);
        const auto& mm = tilingData->mmTilingData;
        EXPECT_EQ(mm.baseM, 256);
        EXPECT_EQ(mm.baseN, 128);
        EXPECT_EQ(mm.baseK, 128);
        EXPECT_EQ(mm.depthA1, 4);
        EXPECT_EQ(mm.depthB1, 4);
        EXPECT_EQ(mm.stepKa, 2);
        EXPECT_EQ(mm.stepKb, 2);
        EXPECT_EQ(mm.dbL0C, 1);
        const auto mxTypePara = static_cast<uint32_t>(mm.mxTypePara);
        EXPECT_EQ(mxTypePara & 0xFFU, 24U);
        EXPECT_EQ((mxTypePara >> 8U) & 0xFFU, 24U);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, PerformanceShapesKeepNdCompatibleBasicTiles)
{
    const std::vector<int64_t> mValues = {
        128, 256, 512, 1024, 1536, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
        18432, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65535};
    for (const auto m : mValues) {
        SCOPED_TRACE(::testing::Message() << "M=" << m);
        const bool useTensorApi = m / 64 > 512;
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {m, 6144, 4096, 64, false, useTensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY}, &info, 0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, useTensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY);
        const auto* tilingData = GetTilingData(result);
        ASSERT_NE(tilingData, nullptr);
        const auto& mm = tilingData->mmTilingData;
        EXPECT_EQ(mm.baseM, useTensorApi ? 256 : 128);
        EXPECT_EQ(mm.baseN, useTensorApi ? 128 : 256);
        EXPECT_EQ(mm.baseK, 128);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, TensorApiKeepsKBoundaryScaleWindowWithinL1)
{
    for (int64_t k : {8192, 8320}) {
        for (bool trans : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "K=" << k << ",trans=" << trans);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(
                {8192, k, 4096, 8, trans, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY}, &info, 0);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            ASSERT_EQ(result.tilingKey, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY);
            const auto* tilingData = GetTilingData(result);
            ASSERT_NE(tilingData, nullptr);
            const auto& mm = tilingData->mmTilingData;
            EXPECT_EQ(mm.baseM, 256);
            EXPECT_EQ(mm.baseN, 128);
            EXPECT_EQ(mm.baseK, 128);
            EXPECT_EQ(mm.dbL0C, 1);
            const auto mxTypePara = static_cast<uint32_t>(mm.mxTypePara);
            EXPECT_EQ(mxTypePara & 0xFFU, 32U);
            EXPECT_EQ((mxTypePara >> 8U) & 0xFFU, 32U);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, CoreRatioFallsBack)
{
    for (bool trans : {false, true}) {
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {1026, 128, 128, 2, trans, trans ? V3_CGMCT_ZN_KEY : V3_CGMCT_NZ_KEY}, &info, 0, 32);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        EXPECT_EQ(result.tilingKey, trans ? V3_CGMCT_ZN_KEY : V3_CGMCT_NZ_KEY);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, LegacyModeUsesV2Template)
{
    for (bool trans : {false, true}) {
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext({512, 128, 128, 1, trans, trans ? 1 : 0}, &info, 0, 64, 0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        EXPECT_EQ(result.tilingKey, trans ? 1 : 0);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsUnsupportedModeAndPlatform)
{
    optiling::GMMSwigluV2CompileInfo invalidModeInfo{};
    auto invalidModeContext = MakeContext({512, 128, 128, 1, false, 0}, &invalidModeInfo, 0, 64, 1);
    TilingInfo result;
    EXPECT_FALSE(ExecuteTiling(invalidModeContext, result));

    optiling::GMMSwigluV2CompileInfo invalidPlatformInfo{};
    auto invalidPlatformContext = MakeContext(
        {512, 128, 128, 1, false, V3_CGMCT_NZ_KEY}, &invalidPlatformInfo, 0);
    invalidPlatformInfo.npuArch_ = static_cast<int32_t>(NpuArch::DAV_2201);
    EXPECT_FALSE(ExecuteTiling(invalidPlatformContext, result));
}
} // namespace
