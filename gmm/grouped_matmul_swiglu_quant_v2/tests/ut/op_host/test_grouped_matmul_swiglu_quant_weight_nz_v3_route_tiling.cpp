/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"
#include "../../../op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_data.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

namespace {
struct RouteCase {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t e;
    bool trans;
    int64_t key;
};

constexpr int64_t V3_CGMCT_NZ_KEY = 0;
constexpr int64_t V3_CGMCT_ZN_KEY = 1;
constexpr int64_t V3_TENSOR_API_NZ_KEY = 16;
constexpr int64_t V3_TENSOR_API_ZN_KEY = 17;

using CgmctTilingData = ::GMMSwigluQuantV2TilingDataParams;
using TensorApiTilingData = GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData;

gert::TilingContextPara MakeContext(const RouteCase &c, optiling::GMMSwigluV2CompileInfo *info, int scaleAlg,
                                    int aiv = 64, int64_t swigluMode = 2)
{
    using Desc = gert::TilingContextPara::TensorDescription;
    using Value = Ops::Transformer::AnyValue;
    const int64_t s = (c.k + 63) / 64;
    const int64_t h = c.n / 2;
    const auto tensor = [](std::vector<int64_t> dims, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(dims, dims), dtype, ge::FORMAT_ND);
    };
    const std::vector<int64_t> w = c.trans ? std::vector<int64_t>{c.e, c.n, c.k} : std::vector<int64_t>{c.e, c.k, c.n};
    const std::vector<int64_t> ws =
        c.trans ? std::vector<int64_t>{c.e, c.n, s, 2} : std::vector<int64_t>{c.e, s, c.n, 2};
    const std::vector<int64_t> storage = c.trans ? std::vector<int64_t>{c.e, (c.k + 31) / 32, (c.n + 15) / 16, 16, 32} :
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
    gert::TilingContextPara context(
        "GroupedMatmulSwigluQuantV2",
        {tensor({c.m, c.k}, ge::DT_FLOAT8_E4M3FN), tensor({c.m, s, 2}, ge::DT_FLOAT8_E8M0), tensor({c.e}, ge::DT_INT64),
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
    context.socInfoString_ =
        R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":)" + std::to_string(aiv) +
        R"(,"UB_SIZE":262144,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144}})";
    return context;
}

template <typename T>
const T *GetTilingData(const TilingInfo &result)
{
    EXPECT_EQ(result.tilingDataSize, sizeof(T));
    if (result.tilingDataSize != sizeof(T)) {
        return nullptr;
    }
    return reinterpret_cast<const T *>(result.tilingData.get());
}

const ::GMMSwigluQuantSwigluParams *GetSwigluParams(const TilingInfo &result)
{
    if (result.tilingKey == V3_CGMCT_NZ_KEY || result.tilingKey == V3_CGMCT_ZN_KEY) {
        const auto *data = GetTilingData<CgmctTilingData>(result);
        return data == nullptr ? nullptr : &data->swigluParams;
    }
    if (result.tilingKey == V3_TENSOR_API_NZ_KEY || result.tilingKey == V3_TENSOR_API_ZN_KEY) {
        const auto *data = GetTilingData<TensorApiTilingData>(result);
        return data == nullptr ? nullptr : &data->swigluParams;
    }
    ADD_FAILURE() << "Unexpected MXFP8 tiling key: " << result.tilingKey;
    return nullptr;
}

void ExpectSwigluParams(const ::GMMSwigluQuantSwigluParams &params, int64_t mode, uint8_t scaleAlg,
                       float clampLimit = 7.0F, float gluAlpha = 1.702F, float gluBias = 1.0F)
{
    EXPECT_EQ(params.swigluMode, mode);
    EXPECT_EQ(params.scaleAlg, scaleAlg);
    EXPECT_FLOAT_EQ(params.clampLimit, clampLimit);
    EXPECT_FLOAT_EQ(params.gluAlpha, gluAlpha);
    EXPECT_FLOAT_EQ(params.gluBias, gluBias);
    EXPECT_FLOAT_EQ(params.dstTypeMax, 0.0F);
    EXPECT_EQ(params.roundMode, 0U);
    EXPECT_EQ(params.reserved0, 0U);
    EXPECT_EQ(params.reserved1, 0U);
    EXPECT_EQ(params.reserved2, 0U);
    EXPECT_EQ(params.reserved3, 0U);
}

template <typename DeviceTiling, typename HostTiling>
void ExpectHostTailReset(HostTiling &host)
{
    ASSERT_EQ(host.GetDataSize(), sizeof(DeviceTiling));
    std::array<uint8_t, sizeof(DeviceTiling)> buffer;
    buffer.fill(0xA5U);
    ::GMMSwigluQuantSwigluParams previous;
    previous.swigluMode = 2;
    previous.clampLimit = 3.0F;
    previous.gluAlpha = 0.5F;
    previous.gluBias = -1.0F;
    previous.scaleAlg = 1;
    previous.dstTypeMax = 448.0F;
    previous.roundMode = 0xA5U;
    previous.reserved0 = 0xA5U;
    previous.reserved1 = 0xA5A5U;
    previous.reserved2 = 0xA5U;
    previous.reserved3 = 0xA5A5U;
    optiling::SetGMMSwigluQuantSwigluParams(host.swigluParams, previous);
    host.SaveToBuffer(buffer.data(), buffer.size());
    DeviceTiling decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(decoded));
    EXPECT_EQ(std::memcmp(&decoded.swigluParams, &previous, sizeof(previous)), 0);

    // Exercise a reused Host payload and a nonzero destination: resetting
    // legacy attributes must overwrite every field, including reserved bytes.
    buffer.fill(0xA5U);
    optiling::SetGMMSwigluQuantSwigluParams(host.swigluParams);
    host.SaveToBuffer(buffer.data(), buffer.size());
    std::memcpy(&decoded, buffer.data(), sizeof(decoded));
    ExpectSwigluParams(decoded.swigluParams, 0, 0);
}

TEST(GroupedMatmulSwigluQuantV3Route, SharedPayloadsPreserveLegacyPrefixes)
{
    EXPECT_EQ(sizeof(::GMMSwigluQuantSwigluParams), 32U);
    EXPECT_EQ(sizeof(::GMMSwigluQuantV2Params), 16U);
    EXPECT_EQ(offsetof(CgmctTilingData, mmTilingData), 16U);
    EXPECT_EQ(offsetof(CgmctTilingData, swigluParams), 16U + sizeof(::TCubeTiling));
    EXPECT_EQ(sizeof(CgmctTilingData), 16U + sizeof(::TCubeTiling) + 32U);
    EXPECT_EQ(sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiQuantParams), 24U);
    EXPECT_EQ(sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiMMTiling), 48U);
    EXPECT_EQ(offsetof(TensorApiTilingData, mmTilingData), 24U);
    EXPECT_EQ(offsetof(TensorApiTilingData, swigluParams), 72U);
    EXPECT_EQ(sizeof(TensorApiTilingData), 104U);

    optiling::GMMSwigluQuantTilingDataParams cgmctHost;
    optiling::GMMSwigluQuantV2TensorApiTilingData tensorApiHost;
    EXPECT_EQ(cgmctHost.GetDataSize(), sizeof(CgmctTilingData));
    EXPECT_EQ(tensorApiHost.GetDataSize(), sizeof(TensorApiTilingData));
}

TEST(GroupedMatmulSwigluQuantV3Route, HostTailResetOverwritesPreviousModeAndReservedBytes)
{
    optiling::GMMSwigluQuantTilingDataParams cgmctHost;
    ExpectHostTailReset<CgmctTilingData>(cgmctHost);
    optiling::GMMSwigluQuantV2TensorApiTilingData tensorApiHost;
    ExpectHostTailReset<TensorApiTilingData>(tensorApiHost);
}

TEST(GroupedMatmulSwigluQuantV3Route, AttributesShareBackendKeys)
{
    using Value = Ops::Transformer::AnyValue;
    const std::vector<RouteCase> cases = {{128, 128, 128, 1, false, V3_CGMCT_NZ_KEY},
                                          {128, 128, 128, 1, true, V3_CGMCT_ZN_KEY},
                                          {1024, 128, 128, 1, false, V3_TENSOR_API_NZ_KEY},
                                          {1024, 128, 128, 1, true, V3_TENSOR_API_ZN_KEY}};
    for (const auto &c : cases) {
        for (int scaleAlg : {0, 1}) {
            SCOPED_TRACE(::testing::Message() << "key=" << c.key << ",alg=" << scaleAlg);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(c, &info, scaleAlg);
            context.attrs_[optiling::GroupedMatmulSwigluQuantV3Tiling::ATTR_INDEX_CLAMP_LIMIT].attr_ =
                Value::CreateFrom<float>(3.0F);
            context.attrs_[optiling::GroupedMatmulSwigluQuantV3Tiling::ATTR_INDEX_GLU_ALPHA].attr_ =
                Value::CreateFrom<float>(0.5F);
            context.attrs_[optiling::GroupedMatmulSwigluQuantV3Tiling::ATTR_INDEX_GLU_BIAS].attr_ =
                Value::CreateFrom<float>(-1.0F);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, c.key);
            const auto *params = GetSwigluParams(result);
            ASSERT_NE(params, nullptr);
            ExpectSwigluParams(*params, 2, static_cast<uint8_t>(scaleAlg), 3.0F, 0.5F, -1.0F);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, V2CapabilityBoundaries)
{
    const std::vector<RouteCase> cases = {
        {512, 128, 128, 1, false, V3_CGMCT_NZ_KEY},         {513, 128, 128, 1, false, V3_TENSOR_API_NZ_KEY},
        {128, 128, 128, 1, true, V3_CGMCT_ZN_KEY},          {129, 128, 128, 1, true, V3_TENSOR_API_ZN_KEY},
        {1025, 128, 128, 2, false, V3_CGMCT_NZ_KEY},        {1026, 128, 128, 2, false, V3_TENSOR_API_NZ_KEY},
        {257, 128, 128, 2, true, V3_CGMCT_ZN_KEY},          {258, 128, 128, 2, true, V3_TENSOR_API_ZN_KEY},
        {1026, 128, 192, 2, false, V3_CGMCT_NZ_KEY},        {258, 128, 192, 2, true, V3_CGMCT_ZN_KEY},
        {1026, 64, 128, 2, false, V3_CGMCT_NZ_KEY},         {258, 64, 128, 2, true, V3_CGMCT_ZN_KEY},
        {1026, 192, 128, 2, false, V3_CGMCT_NZ_KEY},        {258, 192, 128, 2, true, V3_CGMCT_ZN_KEY},
        {3000, 4096, 8192, 3, false, V3_TENSOR_API_NZ_KEY}, {3000, 4096, 8192, 3, true, V3_TENSOR_API_ZN_KEY}};
    for (const auto &c : cases) {
        for (int scaleAlg : {0, 1}) {
            SCOPED_TRACE(::testing::Message() << c.m << "," << c.k << "," << c.n << "," << c.e << ",trans=" << c.trans
                                              << ",alg=" << scaleAlg);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(c, &info, scaleAlg);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, c.key);
            EXPECT_EQ(result.blockNum, 32U);
            const auto *params = GetSwigluParams(result);
            ASSERT_NE(params, nullptr);
            ExpectSwigluParams(*params, 2, static_cast<uint8_t>(scaleAlg));
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, ReportedUnalignedKCasesKeepCgmctScaleFactorsWithinL1)
{
    struct UnalignedKCase {
        RouteCase route;
        int scaleAlg;
    };
    const std::vector<UnalignedKCase> cases = {{{1, 6060, 1920, 5, false, V3_CGMCT_NZ_KEY}, 0},
                                               {{2815, 8208, 896, 7, true, V3_CGMCT_ZN_KEY}, 0},
                                               {{68, 8066, 1280, 5, false, V3_CGMCT_NZ_KEY}, 0},
                                               {{1, 6558, 4352, 5, true, V3_CGMCT_ZN_KEY}, 0},
                                               {{128, 6452, 1408, 4, false, V3_CGMCT_NZ_KEY}, 1}};

    for (const auto &c : cases) {
        SCOPED_TRACE(::testing::Message() << c.route.m << "," << c.route.k << "," << c.route.n << "," << c.route.e
                                          << ",trans=" << c.route.trans << ",alg=" << c.scaleAlg);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(c.route, &info, c.scaleAlg);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, c.route.key);
        const auto *tilingData = GetTilingData<CgmctTilingData>(result);
        ASSERT_NE(tilingData, nullptr);
        const auto &mm = tilingData->mmTilingData;
        ASSERT_GT(mm.stepKa, 0);
        ASSERT_GT(mm.stepKb, 0);
        ASSERT_GT(mm.baseK, 0);

        const auto mxTypePara = static_cast<uint32_t>(mm.mxTypePara);
        const uint64_t scaleFactorA = mxTypePara & 0xFFU;
        const uint64_t scaleFactorB = (mxTypePara >> 8U) & 0xFFU;
        const uint64_t fullScaleWindowsA = std::max<int64_t>(1, c.route.k / (mm.stepKa * mm.baseK));
        const uint64_t fullScaleWindowsB = std::max<int64_t>(1, c.route.k / (mm.stepKb * mm.baseK));
        EXPECT_LE(scaleFactorA, fullScaleWindowsA);
        EXPECT_LE(scaleFactorB, fullScaleWindowsB);

        const uint64_t scaleK = ((mm.baseK + 31U) / 32U + 1U) / 2U * 2U;
        const uint64_t baseASize = mm.baseM * mm.baseK;
        const uint64_t baseBSize = mm.baseN * mm.baseK;
        const uint64_t baseScaleASize = mm.baseM * scaleK;
        const uint64_t baseScaleBSize = mm.baseN * scaleK;
        const uint64_t l1Size = mm.depthA1 * (baseASize + scaleFactorA * baseScaleASize) +
                                mm.depthB1 * (baseBSize + scaleFactorB * baseScaleBSize);
        EXPECT_LE(l1Size, info.l1Size_);
        if (c.route.m == 1 && c.route.k == 6060 && c.route.n == 1920 && !c.route.trans) {
            // This is the smallest reported failure.  The old post-budget
            // ceil adjustment changed factorB from 7 to 12 and required
            // 563200 bytes, overflowing the 512-KiB L1 by 38912 bytes.
            EXPECT_EQ(scaleFactorA, 1U);
            EXPECT_EQ(scaleFactorB, 7U);
            EXPECT_EQ(l1Size, 522240U);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, TensorApiUsesSymmetricL1Budget)
{
    for (bool trans : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "trans=" << trans);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context =
            MakeContext({8192, 6144, 4096, 8, trans, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY}, &info, 0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY);
        const auto *tilingData = GetTilingData<TensorApiTilingData>(result);
        ASSERT_NE(tilingData, nullptr);
        const auto &mm = tilingData->mmTilingData;
        EXPECT_EQ(mm.baseM, 256);
        EXPECT_EQ(mm.baseN, 128);
        EXPECT_EQ(mm.baseK, 128);
        // The compact Tensor API protocol carries the actual L1 windows,
        // not the TCube step/depth encoding used by the CGMCT family.
        EXPECT_EQ(mm.kAL1, 256U);
        EXPECT_EQ(mm.kBL1, 256U);
        EXPECT_EQ(mm.scaleKAL1, 6144U);
        EXPECT_EQ(mm.scaleKBL1, 6144U);
        EXPECT_EQ(mm.dbL0C, 1);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, PerformanceShapesKeepNdCompatibleBasicTiles)
{
    const std::vector<int64_t> mValues = {128,   256,   512,   1024,  1536,  2048,  4096,  6144,  8192,
                                          10240, 12288, 14336, 16384, 18432, 20480, 24576, 28672, 32768,
                                          36864, 40960, 45056, 49152, 53248, 57344, 61440, 65535};
    for (const auto m : mValues) {
        SCOPED_TRACE(::testing::Message() << "M=" << m);
        const bool useTensorApi = m / 64 > 512;
        optiling::GMMSwigluV2CompileInfo info{};
        auto context =
            MakeContext({m, 6144, 4096, 64, false, useTensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY}, &info, 0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, useTensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY);
        if (useTensorApi) {
            const auto *tilingData = GetTilingData<TensorApiTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            EXPECT_EQ(tilingData->mmTilingData.baseM, 256U);
            EXPECT_EQ(tilingData->mmTilingData.baseN, 128U);
            EXPECT_EQ(tilingData->mmTilingData.baseK, 128U);
        } else {
            const auto *tilingData = GetTilingData<CgmctTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            EXPECT_EQ(tilingData->mmTilingData.baseM, 128);
            EXPECT_EQ(tilingData->mmTilingData.baseN, 256);
            EXPECT_EQ(tilingData->mmTilingData.baseK, 128);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, TensorApiKeepsKBoundaryScaleWindowWithinL1)
{
    for (int64_t k : {8192, 8320}) {
        for (bool trans : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "K=" << k << ",trans=" << trans);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context =
                MakeContext({8192, k, 4096, 8, trans, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY}, &info, 0);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            ASSERT_EQ(result.tilingKey, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY);
            const auto *tilingData = GetTilingData<TensorApiTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            const auto &mm = tilingData->mmTilingData;
            EXPECT_EQ(mm.baseM, 256);
            EXPECT_EQ(mm.baseN, 128);
            EXPECT_EQ(mm.baseK, 128);
            EXPECT_EQ(mm.dbL0C, 1);
            EXPECT_EQ(mm.kAL1, 256U);
            EXPECT_EQ(mm.kBL1, 256U);
            EXPECT_EQ(mm.scaleKAL1, 8192U);
            EXPECT_EQ(mm.scaleKBL1, 8192U);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, CoreRatioFallsBack)
{
    for (bool trans : {false, true}) {
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext({1026, 128, 128, 2, trans, trans ? V3_CGMCT_ZN_KEY : V3_CGMCT_NZ_KEY}, &info, 0, 32);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        EXPECT_EQ(result.tilingKey, trans ? V3_CGMCT_ZN_KEY : V3_CGMCT_NZ_KEY);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, LegacyModeUsesV2Template)
{
    for (bool tensorApi : {false, true}) {
        for (bool trans : {false, true}) {
            for (bool explicitMode : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "tensorApi=" << tensorApi << ",trans=" << trans
                                                  << ",explicitMode=" << explicitMode);
                const int64_t key = (tensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY) + (trans ? 1 : 0);
                optiling::GMMSwigluV2CompileInfo info{};
                auto context = MakeContext({1024, 128, 128, 1, trans, key}, &info, 0, 64, 0);
                if (tensorApi) {
                    // Legacy Tensor API accepts ND weights, whereas V3 uses
                    // the same key family for its supported weight-NZ input.
                    const std::vector<int64_t> weightShape = {1, 128, 128};
                    context.inputTensorDesc_[3] = gert::TilingContextPara::TensorDescription(
                        ops::ut::MakeGertStorageShape(weightShape, weightShape), ge::DT_FLOAT8_E4M3FN,
                        ge::FORMAT_ND);
                }
                if (!explicitMode) {
                    context.attrs_.erase(context.attrs_.begin() +
                                             optiling::GroupedMatmulSwigluQuantV3Tiling::ATTR_INDEX_SWIGLU_MODE,
                                         context.attrs_.end());
                }
                TilingInfo result;
                ASSERT_TRUE(ExecuteTiling(context, result));
                EXPECT_EQ(result.tilingKey, key);
                const auto *params = GetSwigluParams(result);
                ASSERT_NE(params, nullptr);
                ExpectSwigluParams(*params, 0, 0);
            }
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsUnsupportedModeAndPlatform)
{
    optiling::GMMSwigluV2CompileInfo invalidModeInfo{};
    auto invalidModeContext = MakeContext({512, 128, 128, 1, false, 0}, &invalidModeInfo, 0, 64, 1);
    TilingInfo result;
    EXPECT_FALSE(ExecuteTiling(invalidModeContext, result));

    optiling::GMMSwigluV2CompileInfo invalidPlatformInfo{};
    auto invalidPlatformContext = MakeContext({512, 128, 128, 1, false, V3_CGMCT_NZ_KEY}, &invalidPlatformInfo, 0);
    invalidPlatformInfo.npuArch_ = static_cast<int32_t>(NpuArch::DAV_2201);
    EXPECT_FALSE(ExecuteTiling(invalidPlatformContext, result));
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsMissingRequiredInputs)
{
    constexpr std::array<const char *, 5> INPUT_NAMES = {"x", "xScale", "groupList", "weight", "weightScale"};
    for (size_t inputIndex = 0; inputIndex < INPUT_NAMES.size(); ++inputIndex) {
        SCOPED_TRACE(INPUT_NAMES[inputIndex]);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext({512, 128, 128, 1, false, V3_CGMCT_NZ_KEY}, &info, 0);
        context.inputTensorDesc_.erase(context.inputTensorDesc_.begin() + inputIndex);
        context.inputInstanceNum_[inputIndex] = 0;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsMissingRequiredOutputs)
{
    constexpr std::array<const char *, 2> OUTPUT_NAMES = {"y", "yScale"};
    for (size_t outputIndex = 0; outputIndex < OUTPUT_NAMES.size(); ++outputIndex) {
        SCOPED_TRACE(OUTPUT_NAMES[outputIndex]);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext({512, 128, 128, 1, false, V3_CGMCT_NZ_KEY}, &info, 0);
        context.outputTensorDesc_.erase(context.outputTensorDesc_.begin() + outputIndex);
        context.outputInstanceNum_[outputIndex] = 0;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}
} // namespace
