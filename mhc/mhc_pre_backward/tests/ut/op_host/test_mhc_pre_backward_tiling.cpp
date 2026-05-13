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
#include "../../../op_host/op_tiling/arch35/mhc_pre_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcPreBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreBackwardTiling TearDown" << std::endl;
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

/*
 * 网络用例1：B=2, S=4096, n=4，D=1536, x数据类型bf16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case01_B2_S4096_n4_D1536_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 1536;
    uint32_t nD = n * D;                 // 6144
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},          // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 网络用例2：B=2, S=4096, n=4，D=2048, x数据类型fp16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case02_B2_S4096_n4_D2048_FP16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 2048;
    uint32_t nD = n * D;                 // 8192
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 网络用例3：B=2, S=4096, n=4，D=6144, x数据类型bf16
 * alpha [0.5 0.5 0.5]，hc_eps 200
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case03_B2_S4096_n4_D6144_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 6144;
    uint32_t nD = n * D;                 // 24576
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},          // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 兼容性用例1：B=2, S=4096, n=4，D=1536, x数据类型bf16
 * alpha [0.6 0.6 0.6]，hc_eps 300
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case04_B2_S4096_n4_D1536_BF16_Compat1)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 1536;
    uint32_t nD = n * D;                 // 6144
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},          // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 兼容性用例2：B=2, S=4096, n=4，D=2048, x数据类型fp16
 * alpha [0.4 0.4 0.4]，hc_eps 400
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case05_B2_S4096_n4_D2048_FP16_Compat2)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 2048;
    uint32_t nD = n * D;                 // 8192
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 兼容性用例3：B=2, S=4096, n=4，D=6144, x数据类型bf16
 * alpha [0.2 0.2 0.2]，hc_eps 250
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case06_B2_S4096_n4_D6144_BF16_Compat3)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t D = 6144;
    uint32_t nD = n * D;                 // 24576
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},          // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 边界测试用例1：B=1, S=1, n=4，D=64, x数据类型bf16
 * 最小尺寸测试
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case07_B1_S1_n4_D64_BF16_Boundary)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 4;
    uint32_t D = 64;
    uint32_t nD = n * D;                 // 4
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},          // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 边界测试用例2：B=1, S=65535, n=8，D=16384, x数据类型fp16
 * 大尺寸测试
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case08_B1_S65535_n8_D8192_FP16_Boundary)
{
    uint32_t B = 1;
    uint32_t S = 65535;
    uint32_t n = 8;
    uint32_t D = 16384;
    uint32_t nD = n * D;
    uint32_t fusionSize = n * n + 2 * n; // 80
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 不同n值测试用例1：B=2, S=2048, n=6，D=1024, x数据类型bf16
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case09_B2_S2048_n6_D1024_BF16)
{
    uint32_t B = 2;
    uint32_t S = 2048;
    uint32_t n = 6;
    uint32_t D = 1024;
    uint32_t nD = n * D;                 // 6144
    uint32_t fusionSize = n * n + 2 * n; // 48
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_BF16, ge::FORMAT_ND},                    // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_BF16, ge::FORMAT_ND},          // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * hc_eps极小值测试：B=4, S=1024, n=4，D=512, x数据类型fp16
 * hc_eps设置为0（默认值）
 * 预期结果：成功
 */
TEST_F(MhcPreBackwardTiling, Ut_Check_Case10_B4_S1024_n4_D512_FP16_HcEpsZero)
{
    uint32_t B = 4;
    uint32_t S = 1024;
    uint32_t n = 4;
    uint32_t D = 512;
    uint32_t nD = n * D;                 // 2048
    uint32_t fusionSize = n * n + 2 * n; // 24
    float hcEps = 0.000001f;

    optiling::MhcPreBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreBackward",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},     // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha
            {{{B, S, D}, {B, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // gradHIn
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // gradHPost
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // gradHRes
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // invRms
            {{{B, S, fusionSize}, {B, S, fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hMix
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPre
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // hPost
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // gamma (optional)
        },
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // gradX
            {{{fusionSize, nD}, {fusionSize, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // gradPhi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // gradAlpha
            {{{fusionSize}, {fusionSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // gradBias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // gradGamma (optional)
        },
        {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}}, &compileInfo);

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}