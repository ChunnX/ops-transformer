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
#include "../../../op_host/op_tiling/arch35/mhc_pre_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcPreTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreTiling TearDown" << std::endl;
    }
};

template <typename T>
static string TilingData2Str(void* buf, size_t size) {
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

/*
* 测试用例1：B=1，S=1，n=4，d=1，x的数据类型为bf16
* alpha的值为[0.1, 0.1, 0.1]，norm_eps值为0.000001，hc_eps值为0.000001
* 预期结果：失败
*/
TEST_F(MhcPreTiling, Ut_Check_Case01_B1_S1_n4_d1_BF16)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 4;
    uint32_t d = 1;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 4
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.1f, 0.1f, 0.1f};  // alpha = [0.1, 0.1, 0.1]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // alpha (fixed size 3)
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},          // bias
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}           // gamma (optional)
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},             // hin
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},            // h_post
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},      // h_res
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // inv_rms (optional)
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // h_mix (optional)
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}             // h_pre (optional)
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/*
* 测试用例2：B=1，S=1，n=6，d=65535，x的数据类型为float16
* alpha的值为[1.0, 1.0, 1.0]，norm_eps值为20.0，hc_eps值为200
* 预期结果：失败
*/
TEST_F(MhcPreTiling, Ut_Check_Case02_B1_S1_n6_d65535_FP16)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 6;
    uint32_t d = 65535;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 393210
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.5f, 0.5f, 0.5f};  // alpha = [0.5, 0.5, 0.5]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/*
* 测试用例3：B=65535，S=1，n=8，d=1，x的数据类型为bf16
* alpha的值为[3.0, 0.3, 0.03]，norm_eps值为1024.0，hc_eps值为2
* 预期结果：失败
*/
TEST_F(MhcPreTiling, Ut_Check_Case03_B65535_S1_n8_d1_BF16)
{
    uint32_t B = 65535;
    uint32_t S = 1;
    uint32_t n = 8;
    uint32_t d = 1;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 8
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.3f, -0.3f, -0.3f};  // alpha = [-0.3, -0.3, -0.3]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/*
* 测试用例4：B=2，S=4096，n=4，d=1536，x的数据类型为bf16
* alpha的值为[0.5, 0.5, 0.5]，norm_eps值为3.0，hc_eps值为200
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case04_B2_S4096_n4_d1536_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.2f, 0.2f, 0.2f};  // alpha = [0.2, 0.2, 0.2]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例5：B=2，S=4096，n=6，d=2048，x的数据类型为float16
* alpha的值为[3.0, 10.0, 100.0]，norm_eps值为1.0，hc_eps值为20
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case05_B2_S4096_n6_d2048_FP16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 6;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 12288
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.6f, 0.6f, 0.6f};  // alpha = [0.6, 0.6, 0.6]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例6：B=2，S=4096，n=8，d=6144，x的数据类型为bf16
* alpha的值为[0.5, 0.5, 0.5]，norm_eps值为3.0，hc_eps值为200
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case06_B2_S4096_n8_d6144_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 8;
    uint32_t d = 6144;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 49152
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.5f, -0.5f, -0.5f};  // alpha = [-0.5, -0.5, -0.5]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例7：B=256，S=1024，n=4，d=2048，x的数据类型为bf16
* alpha的值为[0.2, 1.5, 100.0]，norm_eps值为50.0，hc_eps值为1000
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case07_B256_S1024_n4_d2048_BF16)
{
    uint32_t B = 256;
    uint32_t S = 1024;
    uint32_t n = 4;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 8192
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.3f, 0.3f, 0.3f};  // alpha = [0.3, 0.3, 0.3]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例8：B=20，S=4096，n=6，d=1024，x的数据类型为float16
* alpha的值为[10.0, 5.0, 20.0]，norm_eps值为60.0，hc_eps值为384
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case08_B20_S4096_n6_d1024_FP16)
{
    uint32_t B = 20;
    uint32_t S = 4096;
    uint32_t n = 6;
    uint32_t d = 1024;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.7f, 0.7f, 0.7f};  // alpha = [0.7, 0.7, 0.7]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例9：B=8，S=512，n=8，d=768，x的数据类型为float16
* alpha的值为[0.8, 0.8, 0.8]，norm_eps值为0.00001，hc_eps值为100
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case09_B8_S512_n8_d768_FP16)
{
    uint32_t B = 8;
    uint32_t S = 512;
    uint32_t n = 8;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.2f, -0.2f, -0.2f};  // alpha = [-0.2, -0.2, -0.2]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例10：B=32，S=256，n=4，d=512，x的数据类型为bf16
* alpha的值为[1.2, 1.5, 2.0]，norm_eps值为0.000001，hc_eps值为500
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case10_B32_S256_n4_d512_BF16)
{
    uint32_t B = 32;
    uint32_t S = 256;
    uint32_t n = 4;
    uint32_t d = 512;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 2048
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.4f, 0.4f, 0.4f};  // alpha = [0.4, 0.4, 0.4]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例11：B=1，S=8192，n=6，d=256，x的数据类型为float16
* alpha的值为[0.3, 0.3, 0.3]，norm_eps值为0.1，hc_eps值为50
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case11_B1_S8192_n6_d256_FP16)
{
    uint32_t B = 1;
    uint32_t S = 8192;
    uint32_t n = 6;
    uint32_t d = 256;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 1536
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.8f, 0.8f, 0.8f};  // alpha = [0.8, 0.8, 0.8]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例12：B=128，S=64，n=8，d=1024，x的数据类型为bf16
* alpha的值为[2.5, 3.0, 3.5]，norm_eps值为0.01，hc_eps值为300
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case12_B128_S64_n8_d1024_BF16)
{
    uint32_t B = 128;
    uint32_t S = 64;
    uint32_t n = 8;
    uint32_t d = 1024;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 8192
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.4f, -0.4f, -0.4f};  // alpha = [-0.4, -0.4, -0.4]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例13：B=4，S=2048，n=4，d=3072，x的数据类型为float16
* alpha的值为[1.0, 2.0, 3.0]，norm_eps值为10.0，hc_eps值为150
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case13_B4_S2048_n4_d3072_FP16)
{
    uint32_t B = 4;
    uint32_t S = 2048;
    uint32_t n = 4;
    uint32_t d = 3072;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 12288
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.15f, 0.15f, 0.15f};  // alpha = [0.15, 0.15, 0.15]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例14：B=16，S=1024，n=6，d=128，x的数据类型为bf16
* alpha的值为[0.1, 1.0, 10.0]，norm_eps值为0.001，hc_eps值为1000
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case14_B16_S1024_n6_d128_BF16)
{
    uint32_t B = 16;
    uint32_t S = 1024;
    uint32_t n = 6;
    uint32_t d = 128;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 768
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.65f, 0.65f, 0.65f};  // alpha = [0.65, 0.65, 0.65]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例15：B=64，S=128，n=8，d=4096，x的数据类型为float16
* alpha的值为[5.0, 5.0, 5.0]，norm_eps值为100.0，hc_eps值为50
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case15_B64_S128_n8_d4096_FP16)
{
    uint32_t B = 64;
    uint32_t S = 128;
    uint32_t n = 8;
    uint32_t d = 4096;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 32768
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.35f, -0.35f, -0.35f};  // alpha = [-0.35, -0.35, -0.35]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例16：B=2，S=32768，n=4，d=512，x的数据类型为bf16
* alpha的值为[0.01, 0.1, 1.0]，norm_eps值为0.5，hc_eps值为800
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case16_B2_S32768_n4_d512_BF16)
{
    uint32_t B = 2;
    uint32_t S = 32768;
    uint32_t n = 4;
    uint32_t d = 512;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 2048
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.25f, 0.25f, 0.25f};  // alpha = [0.25, 0.25, 0.25]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例17：B=48，S=512，n=6，d=2048，x的数据类型为float16
* alpha的值为[0.2, 0.4, 0.8]，norm_eps值为5.0，hc_eps值为250
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case17_B48_S512_n6_d2048_FP16)
{
    uint32_t B = 48;
    uint32_t S = 512;
    uint32_t n = 6;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 12288
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.75f, 0.75f, 0.75f};  // alpha = [0.75, 0.75, 0.75]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例18：B=12，S=1536，n=8，d=1536，x的数据类型为bf16
* alpha的值为[1.5, 2.5, 4.0]，norm_eps值为2.0，hc_eps值为120
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case18_B12_S1536_n8_d1536_BF16)
{
    uint32_t B = 12;
    uint32_t S = 1536;
    uint32_t n = 8;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 12288
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.25f, -0.25f, -0.25f};  // alpha = [-0.25, -0.25, -0.25]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例19：B=1，S=1，n=4，d=128，x的数据类型为float16
* alpha的值为[0.05, 0.05, 0.05]，norm_eps值为0.1，hc_eps值为100
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case19_B1_S1_n4_d128_FP16)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 4;
    uint32_t d = 128;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 512
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.35f, 0.35f, 0.35f};  // alpha = [0.35, 0.35, 0.35]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例20：B=1024，S=32，n=6，d=768，x的数据类型为bf16
* alpha的值为[0.7, 0.7, 0.7]，norm_eps值为0.001，hc_eps值为600
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case20_B1024_S32_n6_d768_BF16)
{
    uint32_t B = 1024;
    uint32_t S = 32;
    uint32_t n = 6;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 4608
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.85f, 0.85f, 0.85f};  // alpha = [0.85, 0.85, 0.85]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例21：t = 1024，n=6，d=768，x的数据类型为bf16
* alpha的值为[0.85, 0.85, 0.85]，norm_eps值为0.000001，hc_eps值为0.000001
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case21_T1024_n6_d768_BF16)
{
    uint32_t T = 1024;
    uint32_t n = 6;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 4608
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.85f, 0.85f, 0.85f};  // alpha = [0.85, 0.85, 0.85]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{T, n, d}, {T, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{T, d}, {T, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, phi_dim1}, {T, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}
/*
* 测试用例22：B=2，S=4096，n=4，d=1536，x的数据类型为bf16，gamma为空（可选输入）
* alpha的值为[0.2, 0.2, 0.2]，norm_eps值为0.000001，hc_eps值为0.000001
* 预期结果：成功
* 说明：gamma为可选输入，此用例测试gamma为空时的正确性
*/
TEST_F(MhcPreTiling, Ut_Check_Case22_B2_S4096_n4_d1536_NoGamma)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.2f, 0.2f, 0.2f};
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    // 不包含gamma输入，只有4个输入：x, phi, alpha, bias（gamma为可选输入，此处不填）
    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},              // x
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // alpha (fixed size 3)
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND}                   // bias
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},                    // hin
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // h_post
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},             // h_res
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // inv_rms (optional)
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},     // h_mix (optional)
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // h_pre (optional)
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例23：outFlag=1，SPLIT_BS模式，BSND layout
* B=2，S=4096，n=4，d=1536，x的数据类型为bf16
* totalLength=8192 > 512 → SPLIT_BS
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case23_OutFlag1_SplitBS_BSND_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.3f, 0.3f, 0.3f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例24：outFlag=1，SPLIT_ND模式，BSND layout
* B=1，S=64，n=6，d=2048，x的数据类型为float16
* totalLength=64 <= 512 → SPLIT_ND
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case24_OutFlag1_SplitND_BSND_FP16)
{
    uint32_t B = 1;
    uint32_t S = 64;
    uint32_t n = 6;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n;  // 48
    uint32_t phi_dim1 = n * d;          // 12288
    uint32_t bias_dim = n * n + 2 * n;  // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.5f, 0.5f, 0.5f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例25：outFlag=1，SPLIT_BS模式，TND layout
* T=1024，n=4，d=1536，x的数据类型为bf16
* totalLength=1024 > 512 → SPLIT_BS
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case25_OutFlag1_SplitBS_TND_BF16)
{
    uint32_t T = 1024;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.4f, 0.4f, 0.4f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{T, n, d}, {T, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{T, d}, {T, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, phi_dim1}, {T, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例26：outFlag=1，SPLIT_ND模式，TND layout
* T=256，n=8，d=768，x的数据类型为float16
* totalLength=256 <= 512 → SPLIT_ND
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case26_OutFlag1_SplitND_TND_FP16)
{
    uint32_t T = 256;
    uint32_t n = 8;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n;  // 80
    uint32_t phi_dim1 = n * d;          // 6144
    uint32_t bias_dim = n * n + 2 * n;  // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.6f, 0.6f, 0.6f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{T, n, d}, {T, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{T, d}, {T, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, phi_dim1}, {T, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}

/*
* 测试用例27：outFlag=1，长序列 totalLength>65536，BSND layout
* B=1，S=70000，n=4，d=512，x的数据类型为bf16
* totalLength=70000 > 65536 → SPLIT_BS
* 预期结果：成功
*/
TEST_F(MhcPreTiling, Ut_Check_Case27_OutFlag1_LongSeq_BSND_BF16)
{
    uint32_t B = 1;
    uint32_t S = 70000;
    uint32_t n = 4;
    uint32_t d = 512;
    uint32_t phi_dim0 = n * n + 2 * n;  // 24
    uint32_t phi_dim1 = n * d;          // 2048
    uint32_t bias_dim = n * n + 2 * n;  // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.5f, 0.5f, 0.5f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, phi_dim1}, {B, S, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
            {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}
        },
        &compileInfo
    );

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int32_t>);
}