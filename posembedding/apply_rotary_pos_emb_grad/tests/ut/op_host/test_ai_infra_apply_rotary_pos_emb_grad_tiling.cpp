/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <initializer_list>
#include <iostream>
#include <string>
#include <gtest/gtest.h>
#include "../../../op_host/apply_rotary_pos_emb_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
constexpr const char *OP_TYPE = "AiInfraApplyRotaryPosEmbGrad";
constexpr uint64_t EXPECT_TILING_KEY = 0;
const std::vector<size_t> EXPECT_WORKSPACES = {16 * 1024 * 1024};

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

gert::TilingContextPara::TensorDescription Td(std::initializer_list<int64_t> shape, ge::DataType dtype)
{
    return {gert::StorageShape(shape, shape), dtype, ge::FORMAT_ND};
}

gert::TilingContextPara::TensorDescription OptTd(ge::DataType dtype)
{
    return {gert::StorageShape(), dtype, ge::FORMAT_ND};
}

std::vector<gert::TilingContextPara::OpAttr> Attrs(const char *mode, int64_t layout)
{
    return {{"rotary_mode", AiInfraOps::Transformer::AnyValue::CreateFrom<string>(mode)},
            {"layout", AiInfraOps::Transformer::AnyValue::CreateFrom<int64_t>(layout)}};
}

// ======== 通用 shape 常量 ========
// D=64 (16 的倍数，满足 dimD % 16 == 0)

// BSND 4D: [B,S,N,D]; Nq=2, Nk=3 (GQA)
const std::initializer_list<int64_t> BSND_GQE = {4, 11, 2, 64};
const std::initializer_list<int64_t> BSND_GKE = {4, 11, 3, 64};
const std::initializer_list<int64_t> BSND_COS = {4, 11, 1, 64};
const std::initializer_list<int64_t> BSND_SIN = {4, 11, 1, 64};

// TND 3D: [S,N,D]; Nq=2, Nk=3
const std::initializer_list<int64_t> TND_GQE = {11, 2, 64};
const std::initializer_list<int64_t> TND_GKE = {11, 3, 64};
const std::initializer_list<int64_t> TND_COS = {11, 1, 64};
const std::initializer_list<int64_t> TND_SIN = {11, 1, 64};

// SBND 4D: [S,B,N,D]; Nq=2, Nk=3
const std::initializer_list<int64_t> SBND_GQE = {11, 4, 2, 64};
const std::initializer_list<int64_t> SBND_GKE = {11, 4, 3, 64};
const std::initializer_list<int64_t> SBND_COS = {11, 4, 1, 64};
const std::initializer_list<int64_t> SBND_SIN = {11, 4, 1, 64};
}  // namespace

class TestAiInfraApplyRotaryPosEmbGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestAiInfraApplyRotaryPosEmbGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TestAiInfraApplyRotaryPosEmbGradTiling TearDown" << std::endl; }
};

// =====================================================================================
// 第一组: 合法输入 —— optional 路径, 精确校验 tilingData
// 覆盖 dtype {fp16, bf16, fp32} × layout {BSND, TND, SBND}
// BSND: Nq=2, Nk=3 (GQA); TND/SBND: Nq=2, Nk=2
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_fp16_bsnd_opt_right)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    string expectTilingData =
        "1 1 5632 8448 2816 4 11 2 3 64 4 1 32 128 128 0 128 0 128 3456 3456 22 1 2 3 5632 8448 64 192 320 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, EXPECT_TILING_KEY, expectTilingData, EXPECT_WORKSPACES, 0,
                    TilingData2Str<int64_t>);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_bf16_bsnd_opt_right)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_BF16), Td(BSND_GKE, ge::DT_BF16), Td(BSND_COS, ge::DT_BF16),
         Td(BSND_SIN, ge::DT_BF16), Td(BSND_GQE, ge::DT_BF16), Td(BSND_GKE, ge::DT_BF16)},
        {Td(BSND_GQE, ge::DT_BF16), Td(BSND_GKE, ge::DT_BF16), Td(BSND_COS, ge::DT_BF16),
         Td(BSND_SIN, ge::DT_BF16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    string expectTilingData =
        "1 1 5632 8448 2816 4 11 2 3 64 4 1 32 128 128 0 128 0 128 3456 3456 22 1 2 3 5632 8448 64 192 320 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, EXPECT_TILING_KEY, expectTilingData, EXPECT_WORKSPACES, 0,
                    TilingData2Str<int64_t>);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_fp32_bsnd_opt_right)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT), Td(BSND_GKE, ge::DT_FLOAT), Td(BSND_COS, ge::DT_FLOAT),
         Td(BSND_SIN, ge::DT_FLOAT), Td(BSND_GQE, ge::DT_FLOAT), Td(BSND_GKE, ge::DT_FLOAT)},
        {Td(BSND_GQE, ge::DT_FLOAT), Td(BSND_GKE, ge::DT_FLOAT), Td(BSND_COS, ge::DT_FLOAT),
         Td(BSND_SIN, ge::DT_FLOAT)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    string expectTilingData =
        "1 1 5632 8448 2816 4 11 2 3 64 4 1 32 128 128 0 128 0 128 3648 3648 22 1 2 3 5632 8448 128 384 640 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, EXPECT_TILING_KEY, expectTilingData, EXPECT_WORKSPACES, 0,
                    TilingData2Str<int64_t>);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_fp16_tnd_opt_right)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td({11, 2, 64}, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_COS, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td({11, 2, 64}, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td({11, 2, 64}, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_COS, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    string expectTilingData =
        "1 4 1408 1408 704 1 11 2 2 64 1 1 32 64 64 0 64 0 64 3456 3456 11 1 2 2 1408 1408 64 192 192 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, EXPECT_TILING_KEY, expectTilingData, EXPECT_WORKSPACES, 0,
                    TilingData2Str<int64_t>);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_fp16_sbnd_opt_right)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td({11, 4, 2, 64}, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td(SBND_COS, ge::DT_FLOAT16), Td(SBND_GQE, ge::DT_FLOAT16), Td({11, 4, 2, 64}, ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td({11, 4, 2, 64}, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td(SBND_COS, ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    string expectTilingData =
        "1 2 5632 5632 2816 4 11 2 2 64 4 1 32 128 128 0 128 0 128 3456 3456 22 1 2 2 1 1 64 192 192 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, EXPECT_TILING_KEY, expectTilingData, EXPECT_WORKSPACES, 0,
                    TilingData2Str<int64_t>);
}

// =====================================================================================
// 第二组: CheckAttr —— 预期 GRAPH_FAILED
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_rotary_mode)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("full", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_layout_zero)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_layout_bnsd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 3), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_layout_five)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 5), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第三组: CheckGradQueryEmbed —— 预期 GRAPH_FAILED
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQueryEmbed_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td({11, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQueryEmbed_dimD_too_large)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td({4, 11, 2, 256}, ge::DT_FLOAT16), Td({4, 11, 3, 256}, ge::DT_FLOAT16),
         Td({4, 11, 1, 256}, ge::DT_FLOAT16), Td({4, 11, 1, 256}, ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({4, 11, 2, 256}, ge::DT_FLOAT16), Td({4, 11, 3, 256}, ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQueryEmbed_dimD_not_multiple_of_16)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td({4, 11, 2, 24}, ge::DT_FLOAT16), Td({4, 11, 3, 24}, ge::DT_FLOAT16),
         Td({4, 11, 1, 24}, ge::DT_FLOAT16), Td({4, 11, 1, 24}, ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({4, 11, 2, 24}, ge::DT_FLOAT16), Td({4, 11, 3, 24}, ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第四组: CheckGradKeyEmbed —— 预期 GRAPH_FAILED
// 覆盖 BSND × 4, TND(3D) × 2, SBND × 3 (共 9 个)
// =====================================================================================

// BSND: rank 不匹配 (gradQEmbed 4D, gradKEmbed 3D)
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({11, 3, 64}, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim0 (B) 不匹配: B=5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimB_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({5, 11, 3, 64}, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim1 (S) 不匹配: S=10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimS_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 10, 3, 64}, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim3 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimD_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 11, 3, 48}, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0 (S) 不匹配: S=10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td({10, 3, 64}, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td({11, 3, 48}, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim0 (S) 不匹配: S=10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimS_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td({10, 4, 3, 64}, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim1 (B) 不匹配: B=5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimB_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td({11, 5, 3, 64}, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim3 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dimD_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td({11, 4, 3, 48}, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第五组: CheckCos —— 预期 GRAPH_FAILED
// 覆盖 BSND × 5, TND(3D) × 3, SBND × 4 (共 12 个)
// =====================================================================================

// BSND: rank 不匹配 (2D vs 4D)
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({11, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim0 (B) 不匹配: B=5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimB_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({5, 11, 1, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim1 (S) 不匹配: S=10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimS_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({4, 10, 1, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim2 (N) != 1: N=2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_N_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({4, 11, 2, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim3 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimD_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({4, 11, 1, 48}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0 (S) 不匹配: S=10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td({10, 1, 64}, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1 (N) != 1: N=2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_N_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td({11, 2, 64}, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td({11, 1, 48}, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim0 (S) 不匹配: S=10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimS_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td({10, 4, 1, 64}, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim1 (B) 不匹配: B=5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimB_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td({11, 5, 1, 64}, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim2 (N) != 1: N=2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_N_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td({11, 4, 2, 64}, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim3 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dimD_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td({11, 4, 1, 48}, ge::DT_FLOAT16),
         Td(SBND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第六组: CheckSin —— 预期 GRAPH_FAILED
// 覆盖 BSND × 5, TND(3D) × 3, SBND × 4 (共 12 个)
// =====================================================================================

// BSND: rank 不匹配 (2D vs 4D)
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({11, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim0 (B) 不匹配: B=5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimB_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({5, 11, 1, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim1 (S) 不匹配: S=12 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimS_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({4, 12, 1, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim2 (N) != 1: N=2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_N_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({4, 11, 2, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: dim3 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimD_bsnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({4, 11, 1, 48}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0 (S) 不匹配: S=12 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td({12, 1, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1 (N) != 1: N=2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_N_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td({11, 2, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td({11, 1, 48}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim0 (S) 不匹配: S=12 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimS_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td({12, 4, 1, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim1 (B) 不匹配: B=5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimB_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td({11, 5, 1, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim2 (N) != 1: N=2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_N_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td({11, 4, 2, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// SBND: dim3 (D) 不匹配: D=48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dimD_sbnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), Td(SBND_COS, ge::DT_FLOAT16),
         Td({11, 4, 1, 48}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(SBND_GQE, ge::DT_FLOAT16), Td(SBND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 2), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第七组: CheckQuery —— 预期 GRAPH_FAILED
// 覆盖 rank + BSND(4D) × 4 + TND(3D) × 3 (共 8 个)
// =====================================================================================

// BSND: query rank=3 ≠ 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td({11, 2, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim0 不匹配: 5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dim0)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td({5, 11, 2, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim1 不匹配: 10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dim1)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td({4, 10, 2, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim2(N) 不匹配: 5 vs 2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dim2)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td({4, 11, 5, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim3(D) 不匹配: 48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dim3)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td({4, 11, 2, 48}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0(S) 不匹配: 10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td({10, 2, 64}, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1(N) 不匹配: 5 vs 2
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dimN_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td({11, 5, 64}, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2(D) 不匹配: 48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_query_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td({11, 2, 48}, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第八组: CheckKey —— 预期 GRAPH_FAILED
// 覆盖 成对×2 + rank + BSND(4D) × 4 + TND(3D) × 3 (共 10 个)
// =====================================================================================

// 仅提供 query，不提供 key
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_optional_query_only)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 仅提供 key，不提供 query
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_optional_key_only)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND: key rank=3 ≠ 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td({11, 3, 64}, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim0 不匹配: 5 vs 4
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dim0)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td({5, 11, 3, 64}, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim1 不匹配: 10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dim1)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 10, 3, 64}, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim2(N) 不匹配: 5 vs 3
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dim2)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 11, 5, 64}, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim3(D) 不匹配: 48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dim3)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 11, 3, 48}, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0(S) 不匹配: 10 vs 11
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td({10, 3, 64}, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1(N) 不匹配: 5 vs 3
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dimN_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td({11, 5, 64}, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2(D) 不匹配: 48 vs 64
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_key_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td({11, 3, 48}, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第九组: CheckInputDatatype —— 预期 GRAPH_FAILED
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQueryEmbed_dtype_int32)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_INT32), Td(BSND_GKE, ge::DT_INT32), Td(BSND_COS, ge::DT_INT32),
         Td(BSND_SIN, ge::DT_INT32), OptTd(ge::DT_INT32), OptTd(ge::DT_INT32)},
        {Td(BSND_GQE, ge::DT_INT32), Td(BSND_GKE, ge::DT_INT32), OptTd(ge::DT_INT32), OptTd(ge::DT_INT32)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKeyEmbed_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_cos_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_sin_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十组: CheckOptionalInputDatatype —— 预期 GRAPH_FAILED
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_optional_query_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_optional_key_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十一组: CheckOutput —— optional 一致性 (L631/L635)
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_optional_missing_gradCosSin)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十二组: CheckGradQuery —— 预期 GRAPH_FAILED
// 覆盖 rank + BSND(4D) × 4 + TND(3D) × 3 (共 8 个)
// =====================================================================================

// BSND: rank 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({11, 2, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim0 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dim0)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({5, 11, 2, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim1 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dim1)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({4, 10, 2, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim2(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dim2)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({4, 11, 5, 64}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim3(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dim3)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({4, 11, 2, 48}, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0(S) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({10, 2, 64}, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dimN_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({11, 5, 64}, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td({11, 2, 48}, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十三组: CheckGradKey —— 预期 GRAPH_FAILED
// 覆盖 rank + BSND(4D) × 4 + TND(3D) × 3 (共 8 个)
// =====================================================================================

// BSND: rank 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({11, 3, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim0 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dim0)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({5, 11, 3, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim1 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dim1)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 10, 3, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim2(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dim2)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 11, 5, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim3(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dim3)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td({4, 11, 3, 48}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16),
         OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0(S) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td({10, 3, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dimN_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td({11, 5, 64}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td({11, 3, 48}, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十四组: CheckGradCos —— 预期 GRAPH_FAILED (hasOptionalInputs 路径)
// 覆盖 rank + BSND(4D) × 4 + TND(3D) × 3 (共 8 个)
// =====================================================================================

// BSND: rank 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({11, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim0 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dim0)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({5, 11, 1, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim1 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dim1)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({4, 10, 1, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim2(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dim2)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({4, 11, 5, 64}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim3(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dim3)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td({4, 11, 1, 48}, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0(S) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td({10, 1, 64}, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dimN_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td({11, 5, 64}, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td({11, 1, 48}, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十五组: CheckGradSin —— 预期 GRAPH_FAILED (hasOptionalInputs 路径)
// 覆盖 rank + BSND(4D) × 4 + TND(3D) × 3 (共 8 个)
// =====================================================================================

// BSND: rank 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_rank)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({11, 64}, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim0 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dim0)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({5, 11, 1, 64}, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim1 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dim1)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({4, 10, 1, 64}, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim2(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dim2)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({4, 11, 5, 64}, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// BSND(4D): dim3(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dim3)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td({4, 11, 1, 48}, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim0(S) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dimS_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td({10, 1, 64}, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim1(N) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dimN_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td({11, 5, 64}, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// TND(3D): dim2(D) 不匹配
TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_dimD_tnd)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td(TND_SIN, ge::DT_FLOAT16), Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16)},
        {Td(TND_GQE, ge::DT_FLOAT16), Td(TND_GKE, ge::DT_FLOAT16), Td(TND_COS, ge::DT_FLOAT16),
         Td({11, 1, 48}, ge::DT_FLOAT16)},
        Attrs("half", 4), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十六组: CheckOutputDatatype —— 预期 GRAPH_FAILED
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradQuery_output_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT), Td(BSND_GKE, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradKey_output_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT), OptTd(ge::DT_FLOAT16), OptTd(ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// =====================================================================================
// 第十七组: CheckOptionalOutputDatatype —— 预期 GRAPH_FAILED
// =====================================================================================

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradCos_output_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT),
         Td(BSND_SIN, ge::DT_FLOAT16)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TestAiInfraApplyRotaryPosEmbGradTiling, arpe_grad_wrong_gradSin_output_dtype)
{
    optiling::AiInfraApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        OP_TYPE,
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT16), Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16)},
        {Td(BSND_GQE, ge::DT_FLOAT16), Td(BSND_GKE, ge::DT_FLOAT16), Td(BSND_COS, ge::DT_FLOAT16),
         Td(BSND_SIN, ge::DT_FLOAT)},
        Attrs("half", 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
