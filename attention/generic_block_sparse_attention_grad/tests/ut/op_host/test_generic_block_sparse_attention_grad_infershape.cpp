/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"

// 测试类：专门测试 GenericBlockSparseAttentionGrad 的 InferShape 逻辑
class GenericBlockSparseAttentionGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- GenericBlockSparseAttentionGrad InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- GenericBlockSparseAttentionGrad InferShape UT TearDown ---" << std::endl;
    }
};

// ============================================================================
// 测试用例 1: BNSD Layout 下的 Shape 推导
// ============================================================================
TEST_F(GenericBlockSparseAttentionGradInferShapeTest, infershape_bnsd_layout)
{
    int64_t b = 2, n = 8, n_kv = 4, s = 256, s_kv = 128, d = 128;

    gert::InfershapeContextPara infershapeContextPara(
        "GenericBlockSparseAttentionGrad",
        {// 输入顺序必须与 def.cpp 一致：query, key, value, dOut, out, lse, ...
         // 0: query [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: key [B, N_kv, S_kv, D]
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: value [B, N_kv, S_kv, D]
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3: dout [B, N, S, D]
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4: attentionOut (REQUIRED)
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 5: softmaxLse (REQUIRED, Float32)
         {{{b, n, s}, {b, n, s}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 6: rsvdBlockIdx (KV2Q)
         {{{b, n, 1, 2}, {b, n, 1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 7: rsvdBlockCount
         {{{b, n, 1}, {b, n, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         // 8: metadataOptional, 9: attenMaskOptional
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 10-11: cuSeqLengthsQ/KvOptional
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         // 12-13: sequsedQ/KvOptional
         {{{b}, {b}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 预期的输出列表占位 (dq, dk, dv)
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{1, 128})},
         {"isPackedGQA", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layoutKv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmaxPrecision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"winLeft", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"winRight", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}});

    // 断言期待结果：dQ=Q, dK=K, dV=V
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, s, d},       // 0: dQ 输出
        {b, n_kv, s_kv, d}, // 1: dK 输出
        {b, n_kv, s_kv, d}  // 2: dV 输出
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 2: TND Layout 下的 Shape 推导 (补齐所有输入防止 GE 框架报错)
// ============================================================================
TEST_F(GenericBlockSparseAttentionGradInferShapeTest, infershape_tnd_layout)
{
    int64_t total_q = 1024, total_kv = 2048, n = 8, n_kv = 2, d = 64;
    int64_t batch = 1;

    gert::InfershapeContextPara infershapeContextPara(
        "GenericBlockSparseAttentionGrad",
        {
            // 输入顺序与 def.cpp 一致（query, key, value, dOut, ...）；block_shape 已改为属性
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // 0: query
            {{{total_kv, n_kv, d}, {total_kv, n_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 1: key
            {{{total_kv, n_kv, d}, {total_kv, n_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 2: value
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // 3: dout
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // 4: attentionOut (REQUIRED)
            {{{total_q, n}, {total_q, n}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // 5: softmaxLse (REQUIRED)
            {{{batch, n, 16, 8}, {batch, n, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},       // 6: rsvdBlockIdx
            {{{batch, n, 16}, {batch, n, 16}}, ge::DT_INT32, ge::FORMAT_ND},             // 7: rsvdBlockCount
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},                             // 8: metadataOptional
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},                             // 9: attenMaskOptional
            {{{batch + 1}, {batch + 1}}, ge::DT_INT64, ge::FORMAT_ND},                   // 10: cuSeqLengthsQOptional
            {{{batch + 1}, {batch + 1}}, ge::DT_INT64, ge::FORMAT_ND},                   // 11: cuSeqLengthsKvOptional
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},                             // 12: sequsedQOptional
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}                              // 13: sequsedKvOptional
        },
        {{{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{1, 128})},
         {"isPackedGQA", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layoutKv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmaxPrecision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"winLeft", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"winRight", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {total_q, n, d},     // 0: dQ 输出
        {total_kv, n_kv, d}, // 1: dK 输出
        {total_kv, n_kv, d}  // 2: dV 输出
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
