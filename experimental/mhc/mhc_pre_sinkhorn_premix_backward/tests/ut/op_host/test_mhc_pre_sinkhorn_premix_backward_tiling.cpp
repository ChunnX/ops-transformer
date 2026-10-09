/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/mhc_pre_sinkhorn_premix_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include <tiling/tiling_api.h>
#include "experimental/mhc/mhc_pre_sinkhorn_premix_backward/op_kernel/arch35/mhc_pre_sinkhorn_premix_backward_data_arch35.h"

using namespace std;

class MhcPreSinkhornPremixBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreSinkhornPremixBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreSinkhornPremixBackwardTiling TearDown" << std::endl;
    }
};

template <typename T>
static string TilingData2Str(void *buf, size_t size)
{
    string result;
    const T *data = reinterpret_cast<const T *>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B2_S128_N4_C256)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B1_S35_N1_C512)
{
    int64_t B = 1;
    int64_t S = 35;
    int64_t N = 1;
    int64_t C = 512;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B2_S64_N2_C256)
{
    int64_t B = 2;
    int64_t S = 64;
    int64_t N = 2;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B1_S64_N8_C128)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t N = 8;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B1_S64_N9_C128_fail)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t N = 9;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B1_S512_N4_C128)
{
    int64_t B = 1;
    int64_t S = 512;
    int64_t N = 4;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B4_S64_N4_C512)
{
    int64_t B = 4;
    int64_t S = 64;
    int64_t N = 4;
    int64_t C = 512;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_T256_N4_C256_3d)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t T = B * S;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_T64_N8_C128_3d)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t T = B * S;
    int64_t N = 8;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_T64_N9_C128_3d_fail)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t T = B * S;
    int64_t N = 9;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B2_S128_N4_C256_ascend950_key0)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, "Ascend950", "", 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B2_S128_N4_C256_ascend950_deterministic_key1)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, "Ascend950", "", 4096, 1);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_T256_N4_C256_3d_ascend950_key1)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t T = B * S;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornPremixBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, "Ascend950", "", 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

// bf16 gradX fp32 装配分流的回归断言（Ascend950, one_stage key0 路由）：
// 1) x 为 DT_BF16 时 workspace 必须大于同形状 fp16，且追加量精确等于
//    bs*n*c*sizeof(float) + numTiles * AlignUp(tile, 8) * sizeof(float)
//    （GetWorkspaceSize 的 gradXM fp32 plane + sSpill 追加，与 kernel InitGM 布局一致）；
// 2) mm1 切分几何（序列化 TCubeTiling）在 bf16/fp16 两种分流下必须逐字节一致：
//    C dtype 不参与 TCubeTiling 序列化与切分（实测多组 shape bf16/fp16 mm1 全字段相同），
//    CType 切 DT_FLOAT 与 workspace 追加共用同一 isBf16_ 探测（x==DT_BF16），
//    本用例以 workspace 精确断言 + tile 不变守卫该分流整体生效。
TEST_F(MhcPreSinkhornPremixBackwardTiling, test_tiling_B2_S128_N4_C256_ascend950_bf16_gradx_fp32)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    auto makePara = [&](ge::DataType mainDtype) {
        optiling::MhcPreSinkhornPremixBackwardCompileInfo compileInfo = {};
        return gert::TilingContextPara(
            "MhcPreSinkhornPremixBackward",
            {
                {{{B, S, C}, {B, S, C}}, mainDtype, ge::FORMAT_ND},
                {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{B, S, N, C}, {B, S, N, C}}, mainDtype, ge::FORMAT_ND},
                {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {{{B, S, N, C}, {B, S, N, C}}, mainDtype, ge::FORMAT_ND},
                {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
            },
            &compileInfo, "Ascend950", "", 4096);
    };

    TilingInfo infoBf16;
    TilingInfo infoFp16;
    ASSERT_TRUE(ExecuteTiling(makePara(ge::DT_BF16), infoBf16));
    ASSERT_TRUE(ExecuteTiling(makePara(ge::DT_FLOAT16), infoFp16));
    EXPECT_EQ(infoBf16.tilingKey, 0);
    EXPECT_EQ(infoFp16.tilingKey, 0);

    // 1) workspace: bf16 > fp16, 追加量精确对账
    ASSERT_EQ(infoBf16.workspaceSizes.size(), 1U);
    ASSERT_EQ(infoFp16.workspaceSizes.size(), 1U);
    EXPECT_GT(infoBf16.workspaceSizes[0], infoFp16.workspaceSizes[0]);
    ASSERT_GE(infoBf16.tilingDataSize, sizeof(MhcPreSinkhornPremixBackwardArch35TilingData));
    ASSERT_GE(infoFp16.tilingDataSize, sizeof(MhcPreSinkhornPremixBackwardArch35TilingData));
    auto *tdBf16 = reinterpret_cast<const MhcPreSinkhornPremixBackwardArch35TilingData *>(infoBf16.tilingData.get());
    auto *tdFp16 = reinterpret_cast<const MhcPreSinkhornPremixBackwardArch35TilingData *>(infoFp16.tilingData.get());
    const int64_t tile = tdBf16->tileSize;
    const int64_t spillSlotElems = (tile + 7) / 8 * 8;
    const int64_t numTiles = (B * S + tile - 1) / tile;
    const int64_t expectAppend = B * S * N * C * static_cast<int64_t>(sizeof(float)) +
                                 numTiles * spillSlotElems * static_cast<int64_t>(sizeof(float));
    EXPECT_EQ(infoBf16.workspaceSizes[0] - infoFp16.workspaceSizes[0], expectAppend);

    // 2) mm1 切分几何守卫：C dtype 切换（bf16 分流）不得扰动序列化切分
    EXPECT_EQ(tdFp16->tileSize, tile);
    EXPECT_EQ(memcmp(&tdBf16->mm1TilingData, &tdFp16->mm1TilingData, sizeof(TCubeTiling)), 0);
    EXPECT_EQ(memcmp(&tdBf16->mm2TilingData, &tdFp16->mm2TilingData, sizeof(TCubeTiling)), 0);
}
