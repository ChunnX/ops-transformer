/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"

// 测试类：专门测试 BSASelectBlockMask 的 InferShape 逻辑
class BSASelectBlockMaskInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- BSASelectBlockMask InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- BSASelectBlockMask InferShape UT TearDown ---" << std::endl;
    }
};

// ============================================================================
// 测试用例 1: BNSD Layout 下的 Shape 推导
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_bnsd_layout)
{
    int64_t b = 2, n = 8, s = 250, s_kv = 129, d = 128;
    int64_t blockShapeData[2] = {128, 64};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {// 0: query [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [B, N, S_kv, D]
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape (OPTIONAL)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL)
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         // 5: actual_seq_lengths_kv (OPTIONAL)
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6: actual_block_len_query (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {// 预期的输出列表占位 (block_sparse_mask_out)
         {{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {// 属性占位 (对齐 OpDef 里的 5 个属性顺序)
         {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    // 断言期待结果：mask_out = [B, N, Xblocks, Yblocks]
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, 2, 3} // 0: block_sparse_mask_out
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 2: TND Layout 下的 Shape 推导
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_tnd_layout)
{
    int64_t total_q = 1025, total_kv = 2049, n = 8, d = 128;
    int64_t batch = 1;
    int64_t blockShapeData[2] = {128, 64};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {// 补齐 8 个输入
         // 0: query [T, N, D]
         {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [T_kv, N, D]
         {{{total_kv, n, d}, {total_kv, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: block_shape (OPTIONAL)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL)
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         // 5: actual_seq_lengths_kv (OPTIONAL)
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6: actual_block_len_query (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    // TND 布局下：batchSize=1, numHeads=n, Xblocks=ceil(total_q/128), Yblocks=ceil(total_kv/64)
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, n, 9, 33} // 0: block_sparse_mask_out
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 3: BF16 数据类型
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_bf16_dtype)
{
    int64_t b = 1, n = 4, s = 513, s_kv = 514, d = 128;
    int64_t blockShapeData[2] = {128, 64};

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {// 0: query [B, N, S, D] BF16
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1: key [B, N, S_kv, D] BF16
         {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_BF16, ge::FORMAT_ND},
         // 2: block_shape (OPTIONAL)
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)blockShapeData},
         // 3: post_block_shape (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 4: actual_seq_lengths (OPTIONAL)
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         // 5: actual_seq_lengths_kv (OPTIONAL)
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6: actual_block_len_query (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 7: actual_block_len_key (OPTIONAL)
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}},
        {{{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, 5, 9} // 0: block_sparse_mask_out
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
