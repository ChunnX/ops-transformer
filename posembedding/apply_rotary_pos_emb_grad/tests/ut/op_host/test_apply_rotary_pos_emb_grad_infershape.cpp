/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ApplyRotaryPosEmbGrad InferShape UT
//
// InferShape 规则 (apply_rotary_pos_emb_grad_infershape.cpp):
//   grad_query  = grad_query_embed  (输入 0 → 输出 0)
//   grad_key    = grad_key_embed    (输入 1 → 输出 1)
//   grad_cos    = cos               (输入 2 → 输出 2, 可选)
//   grad_sin    = sin               (输入 3 → 输出 3, 可选)
// dtype 跟随对应输入。
//
// 覆盖场景:
//   1. fp16/bf16/fp32 三种 dtype (4 输出, 含 grad_cos/grad_sin)
//   2. TND 3D shape
//   3. nQ != nK (grad_query 与 grad_key 形状不同)
//   4. 仅 2 输出 (无 grad_cos/grad_sin, 对应 dCosFlag=0 场景)

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ApplyRotaryPosEmbGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradInferShape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradInferShape Test TearDown" << std::endl;
    }
};

// =====================================================================================
// 4 输出场景 (grad_query, grad_key, grad_cos, grad_sin)
// 输入: grad_query_embed, grad_key_embed, cos, sin, query, key
// =====================================================================================

// fp16 BSND 4D — 标准 4 输出
TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_fp16_bsnd_4out)
{
    gert::InfershapeContextPara para(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_query_embed
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_key_embed
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // cos
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // sin
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_query (待推导)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_key
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_cos
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_sin
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectShape = {
        {2, 64, 4, 128},   // grad_query  = grad_query_embed
        {2, 64, 4, 128},   // grad_key    = grad_key_embed
        {1, 64, 1, 128},   // grad_cos    = cos
        {1, 64, 1, 128},   // grad_sin    = sin
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// bf16 BSND 4D
TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_bf16_bsnd_4out)
{
    gert::InfershapeContextPara para(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectShape = {
        {2, 64, 4, 128},
        {2, 64, 4, 128},
        {1, 64, 1, 128},
        {1, 64, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// fp32 BSND 4D
TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_fp32_bsnd_4out)
{
    gert::InfershapeContextPara para(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{{}, {}}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectShape = {
        {2, 64, 4, 128},
        {2, 64, 4, 128},
        {1, 64, 1, 128},
        {1, 64, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// =====================================================================================
// TND 3D shape
// =====================================================================================

// fp16 TND 3D 4 输出
TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_fp16_tnd_4out)
{
    gert::InfershapeContextPara para(
        "ApplyRotaryPosEmbGrad",
        {
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // [T, N, D]
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // cos [T, 1, D]
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},  // TND
        });
    std::vector<std::vector<int64_t>> expectShape = {
        {64, 4, 128},
        {64, 4, 128},
        {64, 1, 128},
        {64, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// =====================================================================================
// nQ != nK — grad_query 与 grad_key 的 N 维不同
// =====================================================================================

// fp16 BSND nQ=8 nK=4 4 输出
TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_fp16_nq_neq_nk)
{
    gert::InfershapeContextPara para(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // nQ=8
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // nK=4
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectShape = {
        {2, 64, 8, 128},   // grad_query 跟随 nQ=8
        {2, 64, 4, 128},   // grad_key   跟随 nK=4
        {1, 64, 1, 128},
        {1, 64, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// =====================================================================================
// 2 输出场景 (仅 grad_query, grad_key; 无 grad_cos/grad_sin → 对应 dCosFlag=0)
// 输入仅 4 个: grad_query_embed, grad_key_embed, cos, sin (无 query/key)
// =====================================================================================

// fp16 BSND 2 输出
TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_fp16_bsnd_2out)
{
    gert::InfershapeContextPara para(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_query
            {{{{}, {}}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_key
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectShape = {
        {2, 64, 4, 128},   // grad_query = grad_query_embed
        {2, 64, 4, 128},   // grad_key   = grad_key_embed
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}
