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
#include "../../../op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_data.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

namespace {
constexpr int64_t SPLIT_SWIGLU_MODE = 2;
constexpr int SCALE_ALG_CUBLAS = 1;
constexpr int64_t SINGLE_INPUT_COUNT = 1;
constexpr int64_t MX_GROUP_SIZE = 64;
constexpr int64_t MX_HALF_GROUP_SIZE = 32;
constexpr int64_t SCALE_PAIR_SIZE = 2;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
constexpr int64_t NZ_K0 = 16;
constexpr int64_t NZ_C0 = 32;
constexpr int64_t CEIL_DIV_ADJUSTMENT = 1;
constexpr uint32_t AIC_CORE_COUNT = 32;
constexpr uint32_t AIV_CORE_COUNT = 64;
constexpr uint64_t UB_BYTES = 262144;
constexpr uint64_t L1_BYTES = 524288;
constexpr uint64_t L0_INPUT_BYTES = 65536;
constexpr uint64_t L0C_BYTES = 262144;
constexpr float DEFAULT_CLAMP_LIMIT = 7.0F;
constexpr float DEFAULT_GLU_ALPHA = 1.702F;
constexpr float DEFAULT_GLU_BIAS = 1.0F;
constexpr float CUSTOM_CLAMP_LIMIT = 3.0F;
constexpr float CUSTOM_GLU_ALPHA = 0.5F;
constexpr float CUSTOM_GLU_BIAS = -1.0F;
constexpr float FP8_MAX_FINITE = 448.0F;
constexpr uint8_t DIRTY_BYTE_PATTERN = 0xA5U;
constexpr uint16_t DIRTY_HALFWORD_PATTERN = 0xA5A5U;
constexpr uint32_t SCALE_FACTOR_B_SHIFT = 8U;
constexpr uint64_t SCALE_FACTOR_MASK = 0xFFU;
constexpr uint64_t MIN_SCALE_WINDOW_COUNT = 1;
constexpr uint64_t CGMCT_REGRESSION_FACTOR_B = 7;
constexpr uint64_t CGMCT_REGRESSION_L1_BYTES = 522240;
constexpr uint32_t EXPECTED_TENSOR_BASE_M = 256;
constexpr uint32_t EXPECTED_TENSOR_BASE_N = 128;
constexpr uint32_t EXPECTED_TENSOR_BASE_K = 128;
constexpr uint32_t EXPECTED_TENSOR_L1_K = 256;
constexpr uint32_t EXPECTED_CGMCT_BASE_M = 128;
constexpr uint32_t EXPECTED_CGMCT_BASE_N = 256;
constexpr uint32_t EXPECTED_CGMCT_BASE_K = 128;
constexpr uint8_t SINGLE_L0C_BUFFER = 1;
constexpr int64_t TRANSPOSE_KEY_OFFSET = 1;
constexpr int64_t ROUTE_GROUPS = 64;
constexpr int64_t LARGE_ROUTE_M = 8192;
constexpr int64_t LARGE_ROUTE_GROUPS = 8;
constexpr int64_t MEDIUM_ROUTE_M = 1024;
constexpr int64_t EMPTY_GROUP_ROUTE_M = 1;
constexpr int64_t UNALIGNED_REGRESSION_M = 1;
constexpr int64_t UNALIGNED_REGRESSION_K = 6060;
constexpr int64_t UNALIGNED_REGRESSION_N = 1920;
constexpr int64_t INVALID_SWIGLU_MODE = 1;
constexpr int64_t ROUTE_MATRIX_EXTENT = 128;
constexpr int64_t TWO_ROUTE_GROUPS = 2;
constexpr int64_t BOUNDARY_ROUTE_M = 1026;
constexpr int64_t CGMCT_ROUTE_M = 512;
constexpr int64_t SINGLE_GROUP_COUNT = 1;
constexpr int64_t DTYPE_OVERFLOW_SHIFT = 32;
constexpr int64_t DTYPE_OVERFLOW_UNIT = 1LL;
constexpr std::array<int64_t, 2> LONG_K_CASES = {8192, 8320};
constexpr std::array<int64_t, 5> MODE_REUSE_SEQUENCE = {0, 2, 0, 2, 0};
constexpr std::array<int64_t, 2> MODE_REUSE_M_VALUES = {1024, 1};
constexpr int64_t ROUTE_K = 6144;
constexpr int64_t ROUTE_N = 4096;
constexpr int64_t ROUTE_AVERAGE_M_LIMIT = 512;
constexpr size_t WEIGHT_INPUT_INDEX = 3;
constexpr size_t INPUT_NAME_COUNT = 5;
constexpr size_t OUTPUT_NAME_COUNT = 2;
constexpr size_t EXPECTED_SWIGLU_PARAMS_BYTES = 32U;
constexpr size_t EXPECTED_CGMCT_PARAMS_BYTES = 16U;
constexpr size_t EXPECTED_TENSOR_PARAMS_BYTES = 24U;
constexpr size_t EXPECTED_TENSOR_MM_BYTES = 48U;
constexpr size_t EXPECTED_TENSOR_SWIGLU_OFFSET = 72U;
constexpr size_t EXPECTED_TENSOR_TILING_BYTES = 104U;
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
                                    int aiv = AIV_CORE_COUNT, int64_t swigluMode = SPLIT_SWIGLU_MODE)
{
    using Desc = gert::TilingContextPara::TensorDescription;
    using Value = Ops::Transformer::AnyValue;
    const int64_t s = (c.k + MX_GROUP_SIZE - CEIL_DIV_ADJUSTMENT) / MX_GROUP_SIZE;
    const int64_t h = c.n / SWIGLU_SPLIT_FACTOR;
    const auto tensor = [](std::vector<int64_t> dims, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(dims, dims), dtype, ge::FORMAT_ND);
    };
    const std::vector<int64_t> w = c.trans ? std::vector<int64_t>{c.e, c.n, c.k} : std::vector<int64_t>{c.e, c.k, c.n};
    const std::vector<int64_t> ws = c.trans ? std::vector<int64_t>{c.e, c.n, s, SCALE_PAIR_SIZE} :
                                              std::vector<int64_t>{c.e, s, c.n, SCALE_PAIR_SIZE};
    const std::vector<int64_t> storage =
        c.trans ? std::vector<int64_t>{c.e, (c.k + NZ_C0 - CEIL_DIV_ADJUSTMENT) / NZ_C0,
                                       (c.n + NZ_K0 - CEIL_DIV_ADJUSTMENT) / NZ_K0, NZ_K0, NZ_C0} :
                  std::vector<int64_t>{c.e, (c.n + NZ_C0 - CEIL_DIV_ADJUSTMENT) / NZ_C0,
                                       (c.k + NZ_K0 - CEIL_DIV_ADJUSTMENT) / NZ_K0, NZ_K0, NZ_C0};
    info->aicNum_ = AIC_CORE_COUNT;
    info->aivNum_ = static_cast<uint32_t>(aiv);
    info->ubSize_ = UB_BYTES;
    info->l1Size_ = L1_BYTES;
    info->l0ASize_ = L0_INPUT_BYTES;
    info->l0BSize_ = L0_INPUT_BYTES;
    info->l0CSize_ = L0C_BYTES;
    info->npuArch_ = static_cast<int32_t>(NpuArch::DAV_3510);
    info->supportL12BtBf16 = true;
    gert::TilingContextPara context(
        "GroupedMatmulSwigluQuantV2",
        {tensor({c.m, c.k}, ge::DT_FLOAT8_E4M3FN), tensor({c.m, s, SCALE_PAIR_SIZE}, ge::DT_FLOAT8_E8M0),
         tensor({c.e}, ge::DT_INT64),
         Desc(ops::ut::MakeGertStorageShape(w, storage), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ),
         tensor(ws, ge::DT_FLOAT8_E8M0)},
        {tensor({c.m, h}, ge::DT_FLOAT8_E4M3FN),
         tensor({c.m, (h + MX_GROUP_SIZE - CEIL_DIV_ADJUSTMENT) / MX_GROUP_SIZE, SCALE_PAIR_SIZE}, ge::DT_FLOAT8_E8M0)},
        {{"dequant_mode", Value::CreateFrom<int64_t>(SPLIT_SWIGLU_MODE)},
         {"dequant_dtype", Value::CreateFrom<int64_t>(0)},
         {"quant_mode", Value::CreateFrom<int64_t>(SPLIT_SWIGLU_MODE)},
         {"quant_dtype", Value::CreateFrom<int64_t>(ge::DT_FLOAT8_E4M3FN)},
         {"transpose_weight", Value::CreateFrom<bool>(c.trans)},
         {"group_list_type", Value::CreateFrom<int64_t>(0)},
         {"tuning_config", Value::CreateFrom<std::vector<int64_t>>({})},
         {"swiglu_mode", Value::CreateFrom<int64_t>(swigluMode)},
         {"clamp_limit", Value::CreateFrom<float>(DEFAULT_CLAMP_LIMIT)},
         {"glu_alpha", Value::CreateFrom<float>(DEFAULT_GLU_ALPHA)},
         {"glu_bias", Value::CreateFrom<float>(DEFAULT_GLU_BIAS)},
         {"round_mode", Value::CreateFrom<std::string>("rint")},
         {"scale_alg", Value::CreateFrom<int64_t>(scaleAlg)},
         {"dst_type_max", Value::CreateFrom<float>(0.0F)}},
        {SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, 0, 0, 0},
        {SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT}, info, "Ascend950", AIC_CORE_COUNT, UB_BYTES);
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
                        float clampLimit = DEFAULT_CLAMP_LIMIT, float gluAlpha = DEFAULT_GLU_ALPHA,
                        float gluBias = DEFAULT_GLU_BIAS)
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
    buffer.fill(DIRTY_BYTE_PATTERN);
    ::GMMSwigluQuantSwigluParams previous;
    previous.swigluMode = SPLIT_SWIGLU_MODE;
    previous.clampLimit = CUSTOM_CLAMP_LIMIT;
    previous.gluAlpha = CUSTOM_GLU_ALPHA;
    previous.gluBias = CUSTOM_GLU_BIAS;
    previous.scaleAlg = SCALE_ALG_CUBLAS;
    previous.dstTypeMax = FP8_MAX_FINITE;
    previous.roundMode = DIRTY_BYTE_PATTERN;
    previous.reserved0 = DIRTY_BYTE_PATTERN;
    previous.reserved1 = DIRTY_HALFWORD_PATTERN;
    previous.reserved2 = DIRTY_BYTE_PATTERN;
    previous.reserved3 = DIRTY_HALFWORD_PATTERN;
    optiling::SetGMMSwigluQuantSwigluParams(host.swigluParams, previous);
    host.SaveToBuffer(buffer.data(), buffer.size());
    DeviceTiling decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(decoded));
    EXPECT_EQ(std::memcmp(&decoded.swigluParams, &previous, sizeof(previous)), 0);

    // Exercise a reused Host payload and a nonzero destination: resetting
    // legacy attributes must overwrite every field, including reserved bytes.
    buffer.fill(DIRTY_BYTE_PATTERN);
    optiling::SetGMMSwigluQuantSwigluParams(host.swigluParams);
    host.SaveToBuffer(buffer.data(), buffer.size());
    std::memcpy(&decoded, buffer.data(), sizeof(decoded));
    ExpectSwigluParams(decoded.swigluParams, 0, 0);
}

TEST(GroupedMatmulSwigluQuantV3Route, SharedPayloadsPreserveLegacyPrefixes)
{
    EXPECT_EQ(sizeof(::GMMSwigluQuantSwigluParams), EXPECTED_SWIGLU_PARAMS_BYTES);
    EXPECT_EQ(sizeof(::GMMSwigluQuantV2Params), EXPECTED_CGMCT_PARAMS_BYTES);
    EXPECT_EQ(offsetof(CgmctTilingData, mmTilingData), EXPECTED_CGMCT_PARAMS_BYTES);
    EXPECT_EQ(offsetof(CgmctTilingData, swigluParams), EXPECTED_CGMCT_PARAMS_BYTES + sizeof(::TCubeTiling));
    EXPECT_EQ(sizeof(CgmctTilingData),
              EXPECTED_CGMCT_PARAMS_BYTES + sizeof(::TCubeTiling) + EXPECTED_SWIGLU_PARAMS_BYTES);
    EXPECT_EQ(sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiQuantParams), EXPECTED_TENSOR_PARAMS_BYTES);
    EXPECT_EQ(sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiMMTiling), EXPECTED_TENSOR_MM_BYTES);
    EXPECT_EQ(offsetof(TensorApiTilingData, mmTilingData), EXPECTED_TENSOR_PARAMS_BYTES);
    EXPECT_EQ(offsetof(TensorApiTilingData, swigluParams), EXPECTED_TENSOR_SWIGLU_OFFSET);
    EXPECT_EQ(sizeof(TensorApiTilingData), EXPECTED_TENSOR_TILING_BYTES);

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
        for (int scaleAlg : {0, SCALE_ALG_CUBLAS}) {
            SCOPED_TRACE(::testing::Message() << "key=" << c.key << ",alg=" << scaleAlg);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(c, &info, scaleAlg);
            context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_CLAMP_LIMIT].attr_ =
                Value::CreateFrom<float>(CUSTOM_CLAMP_LIMIT);
            context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GLU_ALPHA].attr_ =
                Value::CreateFrom<float>(CUSTOM_GLU_ALPHA);
            context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GLU_BIAS].attr_ =
                Value::CreateFrom<float>(CUSTOM_GLU_BIAS);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, c.key);
            const auto *params = GetSwigluParams(result);
            ASSERT_NE(params, nullptr);
            ExpectSwigluParams(*params, SPLIT_SWIGLU_MODE, static_cast<uint8_t>(scaleAlg), CUSTOM_CLAMP_LIMIT,
                               CUSTOM_GLU_ALPHA, CUSTOM_GLU_BIAS);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, V2CapabilityBoundaries)
{
    const std::vector<RouteCase> cases = {
        {CGMCT_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, false, V3_CGMCT_NZ_KEY},
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
    for (const auto &c : cases) {
        for (int scaleAlg : {0, SCALE_ALG_CUBLAS}) {
            SCOPED_TRACE(::testing::Message() << c.m << "," << c.k << "," << c.n << "," << c.e << ",trans=" << c.trans
                                              << ",alg=" << scaleAlg);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(c, &info, scaleAlg);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, c.key);
            EXPECT_EQ(result.blockNum, AIC_CORE_COUNT);
            const auto *params = GetSwigluParams(result);
            ASSERT_NE(params, nullptr);
            ExpectSwigluParams(*params, SPLIT_SWIGLU_MODE, static_cast<uint8_t>(scaleAlg));
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
        const uint64_t scaleFactorA = mxTypePara & SCALE_FACTOR_MASK;
        const uint64_t scaleFactorB = (mxTypePara >> SCALE_FACTOR_B_SHIFT) & SCALE_FACTOR_MASK;
        const uint64_t fullScaleWindowsA =
            std::max<int64_t>(MIN_SCALE_WINDOW_COUNT, c.route.k / (mm.stepKa * mm.baseK));
        const uint64_t fullScaleWindowsB =
            std::max<int64_t>(MIN_SCALE_WINDOW_COUNT, c.route.k / (mm.stepKb * mm.baseK));
        EXPECT_LE(scaleFactorA, fullScaleWindowsA);
        EXPECT_LE(scaleFactorB, fullScaleWindowsB);

        const uint64_t scaleK =
            ((mm.baseK + MX_HALF_GROUP_SIZE - CEIL_DIV_ADJUSTMENT) / MX_HALF_GROUP_SIZE + CEIL_DIV_ADJUSTMENT) /
            SCALE_PAIR_SIZE * SCALE_PAIR_SIZE;
        const uint64_t baseASize = mm.baseM * mm.baseK;
        const uint64_t baseBSize = mm.baseN * mm.baseK;
        const uint64_t baseScaleASize = mm.baseM * scaleK;
        const uint64_t baseScaleBSize = mm.baseN * scaleK;
        const uint64_t l1Size = mm.depthA1 * (baseASize + scaleFactorA * baseScaleASize) +
                                mm.depthB1 * (baseBSize + scaleFactorB * baseScaleBSize);
        EXPECT_LE(l1Size, info.l1Size_);
        if (c.route.m == UNALIGNED_REGRESSION_M && c.route.k == UNALIGNED_REGRESSION_K &&
            c.route.n == UNALIGNED_REGRESSION_N && !c.route.trans) {
            // This is the smallest reported failure.  The old post-budget
            // ceil adjustment changed factorB from 7 to 12 and required
            // 563200 bytes, overflowing the 512-KiB L1 by 38912 bytes.
            EXPECT_EQ(scaleFactorA, MIN_SCALE_WINDOW_COUNT);
            EXPECT_EQ(scaleFactorB, CGMCT_REGRESSION_FACTOR_B);
            EXPECT_EQ(l1Size, CGMCT_REGRESSION_L1_BYTES);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, TensorApiUsesSymmetricL1Budget)
{
    for (bool trans : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "trans=" << trans);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext({LARGE_ROUTE_M, ROUTE_K, ROUTE_N, LARGE_ROUTE_GROUPS, trans,
                                    trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY},
                                   &info, 0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY);
        const auto *tilingData = GetTilingData<TensorApiTilingData>(result);
        ASSERT_NE(tilingData, nullptr);
        const auto &mm = tilingData->mmTilingData;
        EXPECT_EQ(mm.baseM, EXPECTED_TENSOR_BASE_M);
        EXPECT_EQ(mm.baseN, EXPECTED_TENSOR_BASE_N);
        EXPECT_EQ(mm.baseK, EXPECTED_TENSOR_BASE_K);
        // The compact Tensor API protocol carries the actual L1 windows,
        // not the TCube step/depth encoding used by the CGMCT family.
        EXPECT_EQ(mm.kAL1, EXPECTED_TENSOR_L1_K);
        EXPECT_EQ(mm.kBL1, EXPECTED_TENSOR_L1_K);
        EXPECT_EQ(mm.scaleKAL1, ROUTE_K);
        EXPECT_EQ(mm.scaleKBL1, ROUTE_K);
        EXPECT_EQ(mm.dbL0C, SINGLE_L0C_BUFFER);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, PerformanceShapesKeepNdCompatibleBasicTiles)
{
    const std::vector<int64_t> mValues = {128,   256,   512,   1024,  1536,  2048,  4096,  6144,  8192,
                                          10240, 12288, 14336, 16384, 18432, 20480, 24576, 28672, 32768,
                                          36864, 40960, 45056, 49152, 53248, 57344, 61440, 65535};
    for (const auto m : mValues) {
        SCOPED_TRACE(::testing::Message() << "M=" << m);
        const bool useTensorApi = m / ROUTE_GROUPS > ROUTE_AVERAGE_M_LIMIT;
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {m, ROUTE_K, ROUTE_N, ROUTE_GROUPS, false, useTensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY}, &info,
            0);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        ASSERT_EQ(result.tilingKey, useTensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY);
        if (useTensorApi) {
            const auto *tilingData = GetTilingData<TensorApiTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            EXPECT_EQ(tilingData->mmTilingData.baseM, EXPECTED_TENSOR_BASE_M);
            EXPECT_EQ(tilingData->mmTilingData.baseN, EXPECTED_TENSOR_BASE_N);
            EXPECT_EQ(tilingData->mmTilingData.baseK, EXPECTED_TENSOR_BASE_K);
        } else {
            const auto *tilingData = GetTilingData<CgmctTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            EXPECT_EQ(tilingData->mmTilingData.baseM, EXPECTED_CGMCT_BASE_M);
            EXPECT_EQ(tilingData->mmTilingData.baseN, EXPECTED_CGMCT_BASE_N);
            EXPECT_EQ(tilingData->mmTilingData.baseK, EXPECTED_CGMCT_BASE_K);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, TensorApiKeepsKBoundaryScaleWindowWithinL1)
{
    for (int64_t k : LONG_K_CASES) {
        for (bool trans : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "K=" << k << ",trans=" << trans);
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext({LARGE_ROUTE_M, k, ROUTE_N, LARGE_ROUTE_GROUPS, trans,
                                        trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY},
                                       &info, 0);
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            ASSERT_EQ(result.tilingKey, trans ? V3_TENSOR_API_ZN_KEY : V3_TENSOR_API_NZ_KEY);
            const auto *tilingData = GetTilingData<TensorApiTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            const auto &mm = tilingData->mmTilingData;
            EXPECT_EQ(mm.baseM, EXPECTED_TENSOR_BASE_M);
            EXPECT_EQ(mm.baseN, EXPECTED_TENSOR_BASE_N);
            EXPECT_EQ(mm.baseK, EXPECTED_TENSOR_BASE_K);
            EXPECT_EQ(mm.dbL0C, SINGLE_L0C_BUFFER);
            EXPECT_EQ(mm.kAL1, EXPECTED_TENSOR_L1_K);
            EXPECT_EQ(mm.kBL1, EXPECTED_TENSOR_L1_K);
            EXPECT_EQ(mm.scaleKAL1, LARGE_ROUTE_M);
            EXPECT_EQ(mm.scaleKBL1, LARGE_ROUTE_M);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, CoreRatioFallsBack)
{
    for (bool trans : {false, true}) {
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext({BOUNDARY_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, TWO_ROUTE_GROUPS, trans,
                                    trans ? V3_CGMCT_ZN_KEY : V3_CGMCT_NZ_KEY},
                                   &info, 0, AIC_CORE_COUNT);
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
                SCOPED_TRACE(::testing::Message()
                             << "tensorApi=" << tensorApi << ",trans=" << trans << ",explicitMode=" << explicitMode);
                const int64_t key =
                    (tensorApi ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY) + (trans ? TRANSPOSE_KEY_OFFSET : 0);
                optiling::GMMSwigluV2CompileInfo info{};
                auto context = MakeContext(
                    {MEDIUM_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, trans, key}, &info,
                    0, AIV_CORE_COUNT, 0);
                if (tensorApi) {
                    // Legacy Tensor API accepts ND weights, whereas V3 uses
                    // the same key family for its supported weight-NZ input.
                    const std::vector<int64_t> weightShape = {SINGLE_GROUP_COUNT, ROUTE_MATRIX_EXTENT,
                                                              ROUTE_MATRIX_EXTENT};
                    context.inputTensorDesc_[WEIGHT_INPUT_INDEX] = gert::TilingContextPara::TensorDescription(
                        ops::ut::MakeGertStorageShape(weightShape, weightShape), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
                }
                if (!explicitMode) {
                    context.attrs_.erase(
                        context.attrs_.begin() + optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_SWIGLU_MODE,
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

TEST(GroupedMatmulSwigluQuantV3Route, SharedTemplateResetsEpilogueAndKernelRoute)
{
    // V2 NZ and V3 now reuse template 2. Alternate the epilogue and kernel
    // route to check that cross-call input state does not leak into legacy V2.
    for (bool trans : {false, true}) {
        for (int64_t mode : MODE_REUSE_SEQUENCE) {
            for (int64_t m : MODE_REUSE_M_VALUES) {
                SCOPED_TRACE(::testing::Message() << "trans=" << trans << ",mode=" << mode << ",M=" << m);
                const int scaleAlg = mode == SPLIT_SWIGLU_MODE ? SCALE_ALG_CUBLAS : 0;
                const int64_t key =
                    (mode == SPLIT_SWIGLU_MODE && m == MEDIUM_ROUTE_M ? V3_TENSOR_API_NZ_KEY : V3_CGMCT_NZ_KEY) +
                    (trans ? TRANSPOSE_KEY_OFFSET : 0);
                optiling::GMMSwigluV2CompileInfo info{};
                auto context =
                    MakeContext({m, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, trans, key}, &info,
                                scaleAlg, AIV_CORE_COUNT, mode);
                if (mode == SPLIT_SWIGLU_MODE) {
                    context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_CLAMP_LIMIT].attr_ =
                        Ops::Transformer::AnyValue::CreateFrom<float>(CUSTOM_CLAMP_LIMIT);
                    context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GLU_ALPHA].attr_ =
                        Ops::Transformer::AnyValue::CreateFrom<float>(CUSTOM_GLU_ALPHA);
                    context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GLU_BIAS].attr_ =
                        Ops::Transformer::AnyValue::CreateFrom<float>(CUSTOM_GLU_BIAS);
                }
                TilingInfo result;
                ASSERT_TRUE(ExecuteTiling(context, result));
                EXPECT_EQ(result.tilingKey, key);
                const auto *params = GetSwigluParams(result);
                ASSERT_NE(params, nullptr);
                if (mode == SPLIT_SWIGLU_MODE) {
                    ExpectSwigluParams(*params, mode, scaleAlg, CUSTOM_CLAMP_LIMIT, CUSTOM_GLU_ALPHA, CUSTOM_GLU_BIAS);
                } else {
                    ExpectSwigluParams(*params, 0, 0);
                }
            }
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsUnsupportedModeAndPlatform)
{
    optiling::GMMSwigluV2CompileInfo invalidModeInfo{};
    auto invalidModeContext =
        MakeContext({CGMCT_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, false, 0},
                    &invalidModeInfo, 0, AIV_CORE_COUNT, INVALID_SWIGLU_MODE);
    TilingInfo result;
    EXPECT_FALSE(ExecuteTiling(invalidModeContext, result));

    optiling::GMMSwigluV2CompileInfo invalidPlatformInfo{};
    auto invalidPlatformContext = MakeContext(
        {CGMCT_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, false, V3_CGMCT_NZ_KEY},
        &invalidPlatformInfo, 0);
    invalidPlatformInfo.npuArch_ = static_cast<int32_t>(NpuArch::DAV_2201);
    EXPECT_FALSE(ExecuteTiling(invalidPlatformContext, result));
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsDequantDtypeBeforeEnumNarrowing)
{
    for (int64_t dtype : {DTYPE_OVERFLOW_UNIT << DTYPE_OVERFLOW_SHIFT, -(DTYPE_OVERFLOW_UNIT << DTYPE_OVERFLOW_SHIFT),
                          DTYPE_OVERFLOW_UNIT}) {
        SCOPED_TRACE(::testing::Message() << "dequant_dtype=" << dtype);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {CGMCT_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, false, V3_CGMCT_NZ_KEY},
            &info, 0);
        context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_DEQUANT_DTYPE].attr_ =
            Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtype);
        TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(context, result));
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsMissingRequiredInputs)
{
    constexpr std::array<const char *, INPUT_NAME_COUNT> INPUT_NAMES = {"x", "xScale", "groupList", "weight",
                                                                        "weightScale"};
    for (size_t inputIndex = 0; inputIndex < INPUT_NAMES.size(); ++inputIndex) {
        SCOPED_TRACE(INPUT_NAMES[inputIndex]);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {CGMCT_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, false, V3_CGMCT_NZ_KEY},
            &info, 0);
        context.inputTensorDesc_.erase(context.inputTensorDesc_.begin() + inputIndex);
        context.inputInstanceNum_[inputIndex] = 0;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, RejectsMissingRequiredOutputs)
{
    constexpr std::array<const char *, OUTPUT_NAME_COUNT> OUTPUT_NAMES = {"y", "yScale"};
    for (size_t outputIndex = 0; outputIndex < OUTPUT_NAMES.size(); ++outputIndex) {
        SCOPED_TRACE(OUTPUT_NAMES[outputIndex]);
        optiling::GMMSwigluV2CompileInfo info{};
        auto context = MakeContext(
            {CGMCT_ROUTE_M, ROUTE_MATRIX_EXTENT, ROUTE_MATRIX_EXTENT, SINGLE_GROUP_COUNT, false, V3_CGMCT_NZ_KEY},
            &info, 0);
        context.outputTensorDesc_.erase(context.outputTensorDesc_.begin() + outputIndex);
        context.outputInstanceNum_[outputIndex] = 0;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}
} // namespace
