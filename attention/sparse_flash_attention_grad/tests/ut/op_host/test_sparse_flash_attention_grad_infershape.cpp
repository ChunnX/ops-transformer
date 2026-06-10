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
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

class SparseFlashAttentionGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionGradProto TearDown" << std::endl;
    }
};

// =================== BSND 正例: 无 rope, dq=q / dk=k / dv=v ===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_bsnd_norope)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},          // query
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},        // key
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},        // value
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},             // sparse_indices
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},          // d_out
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},          // out
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // softmax_max
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // softmax_sum
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // actual_seq_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // actual_seq_kv
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // query_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                                     // key_rope
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // d_query
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // d_key
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // d_value
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                    // d_query_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                                     // d_key_rope
        },
        {
            {"scale_value",        Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
            {"sparse_block_size",  Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout",             Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"deterministic",      Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 64, 8, 128},
        {1, 128, 1, 128},
        {1, 128, 1, 128}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== BSND 正例: 带 rope, 额外推 dq_rope / dk_rope ===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_bsnd_rope)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 8, 64}, {1, 64, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},            // query_rope
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}           // key_rope
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"scale_value",        Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
            {"sparse_block_size",  Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout",             Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"deterministic",      Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 64, 8, 128},
        {1, 128, 1, 128},
        {1, 128, 1, 128},
        {1, 64, 8, 64},
        {1, 128, 1, 64}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== TND 正例 (小写 layout 也应通过, infershape 内部 toupper) ===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_tnd_lowercase)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 1, 4}, {64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"scale_value",        Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
            {"sparse_block_size",  Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout",             Ops::Transformer::AnyValue::CreateFrom<std::string>("tnd")}, // 小写 -> 内部 toupper
            {"sparse_mode",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"deterministic",      Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 8, 128},
        {128, 1, 128},
        {128, 1, 128}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// =================== 负例: 非法 layout "BSH" ===================
TEST_F(SparseFlashAttentionGradProto, SparseFlashAttentionGrad_infershape_invalid_layout)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttentionGrad",
        {
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 8, 64}, {1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"scale_value",        Ops::Transformer::AnyValue::CreateFrom<float>(0.0883883476f)},
            {"sparse_block_size",  Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout",             Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"sparse_mode",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens",        Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"deterministic",      Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
