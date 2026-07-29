/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/mhc_pre_sinkhorn_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcPreSinkhornTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreSinkhornTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreSinkhornTiling TearDown" << std::endl;
    }
};

TEST_F(MhcPreSinkhornTiling, test_tiling_B1_S1024_N4_C512)
{
    int64_t B = 1;
    int64_t S = 1024;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N}, {numIters * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N, N}, {numIters * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingData =
        "1024 24 4 512 8 26 10 26 10 0 1 1 1 512 512 20 3856831416675743677 0 0 0 0 1 2048 0 0 0 0 128 256 4 1 0 "
        "1024 512 256 0 0 0 0 0 0 0 0 40 0 20 21 0 0 0 1 128 8 128 128 26 10 26 10 40 1 1 1 256 128 256 4 40 20 "
        "26 10 1 32 8 256 256 20971520 274877906945 8796093022232 274877908992 8796093022232 137438953536 "
        "8589934720 4294967298 1 0 422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 "
        "0 0 0 ";
    std::vector<size_t> expectWorkspaces = {171966464};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_3d_S1024_N4_C512)
{
    int64_t T = 1024;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, T, N}, {numIters * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, T, N, N}, {numIters * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingData =
        "1024 24 4 512 8 26 10 26 10 0 1 1 1 512 512 20 3856831416675743677 0 0 0 0 1 2048 0 0 0 0 128 256 4 1 0 "
        "1024 512 256 0 0 0 0 0 0 0 0 40 0 20 21 0 0 0 1 128 8 128 128 26 10 26 10 40 1 1 1 256 128 256 4 40 20 "
        "26 10 1 32 8 256 256 20971520 274877906945 8796093022232 274877908992 8796093022232 137438953536 "
        "8589934720 4294967298 1 0 422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 "
        "0 0 0 ";
    std::vector<size_t> expectWorkspaces = {171966464};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_B2_S128_N4_C256)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N}, {numIters * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{numIters * 2, B, S, N, N}, {numIters * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingData =
        "256 24 4 256 8 7 4 7 4 0 1 1 1 256 256 20 3856831416675743677 0 0 0 0 1 1024 0 0 0 0 128 256 1 1 0 "
        "1024 256 256 0 0 0 0 0 0 0 0 37 0 20 21 0 0 0 1 128 2 128 128 7 4 7 4 37 1 1 1 256 128 256 1 40 20 7 7 "
        "1 32 4 256 256 10485760 274877906945 4398046511128 274877907968 4398046511128 137438953536 8589934720 "
        "4294967304 1 0 844424930131968 8192 4294967297 4294967297 34359738369 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {161480704};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_no_backward_B1_S256_N4_C512)
{
    int64_t B = 1;
    int64_t S = 256;
    int64_t N = 4;
    int64_t C = 512;
    int64_t numIters = 20;
    float hcEps = 1e-6f;
    float normEps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhorn",
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(N)},
            {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numIters)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingData =
        "256 24 4 512 8 7 4 7 4 0 1 1 1 512 512 20 3856831416675743677 0 0 0 0 1 2048 0 0 0 0 128 256 1 1 0 "
        "1024 256 256 0 0 0 0 0 0 0 0 37 0 20 21 0 0 0 0 128 2 128 128 7 4 7 4 37 1 1 1 256 128 256 1 40 20 7 7 "
        "1 32 8 256 256 20971520 274877906945 8796093022232 274877908992 8796093022232 137438953536 8589934720 "
        "4294967298 1 0 422212465065984 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {171992064};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
