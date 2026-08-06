/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;
constexpr size_t OUTPUT_IDX_Q = 0;
constexpr size_t OUTPUT_IDX_K_CACHE = 1;
constexpr size_t OUTPUT_IDX_V_CACHE = 2;

class UndGenQkvRmsNormRopeCacheInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheInferShape TearDown" << std::endl;
    }
};

gert::InfershapeContextPara BuildInferShapeContext(int64_t undLen, int64_t genLen, int64_t numHeadQ, int64_t numHeadK,
                                                   int64_t numHeadV, int64_t headDim = HEAD_DIM,
                                                   int64_t maxPos = 4096)
{
    const int64_t total = undLen + genLen;
    const int64_t numHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (total + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    return gert::InfershapeContextPara(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, headDim}, {undLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, headDim}, {maxPos, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, headDim}, {blockNum, BLOCK_SIZE, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, headDim}, {blockNum, BLOCK_SIZE, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, total}, {MROPE_AXIS_NUM, total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{genLen, numHead, headDim}, {genLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16, 16})},
        });
}

void CheckInferDataType()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("UndGenQkvRmsNormRopeCache")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType bf16Dtype = ge::DT_BF16;
    ge::DataType floatDtype = ge::DT_FLOAT;
    ge::DataType int64Dtype = ge::DT_INT64;
    ge::DataType qOutDtype = ge::DT_FLOAT;
    ge::DataType kCacheOutDtype = ge::DT_FLOAT;
    ge::DataType vCacheOutDtype = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(12)
                             .NodeIoNum(12, 3)
                             .NodeInputTd(0, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, floatDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, int64Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, int64Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(8, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(9, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(10, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(11, int64Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&bf16Dtype, &bf16Dtype, &bf16Dtype, &floatDtype, &bf16Dtype, &bf16Dtype,
                                              &int64Dtype, &int64Dtype, &bf16Dtype, &bf16Dtype, &bf16Dtype,
                                              &int64Dtype})
                             .OutputDataTypes({&qOutDtype, &kCacheOutDtype, &vCacheOutDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_Q), bf16Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_K_CACHE), bf16Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_V_CACHE), bf16Dtype);
}
} // namespace

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_success_h8_1_1)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    // total = 8, blockNum = 1 + 1 = 2
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
    CheckInferDataType();
}

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_success_h16_2_2)
{
    auto para = BuildInferShapeContext(80, 16, 16, 2, 2);
    // total = 96, blockNum = 1 + 1 = 2
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {96, 16, 128},
        {2, 128, 2, 128},
        {2, 128, 2, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infershape 只保留推导前提的校验（und_qkv/gen_qkv 必须是 3D），
// dtype、维度值、跨输入一致性、支持范围的校验统一在 tiling 侧，见 test_*_tiling.cpp
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_und_qkv_not_3d)
{
    auto para = BuildInferShapeContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{8, 1280}, {8, 1280}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_gen_qkv_not_3d)
{
    auto para = BuildInferShapeContext(8, 4, 8, 1, 1);
    para.inputTensorDesc_[8] = {{{4, 1280}, {4, 1280}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// q 恒为 3D [T, Hq, D]：Hq 直接来自属性，任何场景都可知；T 与 D 能推则推、推不出才置 -1。
// k_cache/v_cache 是原地别名，shape 一律按输入透传。

// und_qkv 为未知 rank（-2）：T 不可知，但 D 可从 cos_sin_cache 的 [max_pos, D] 拿到
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_unknown_rank_und_qkv)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// gen_qkv 为未知 rank：T = und_len + gen_len 不可知，D 仍从 und_qkv 尾维拿到
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_unknown_rank_gen_qkv)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[8] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// D 的取值链路：und_qkv 与 cos_sin_cache 都是 -2 时，退到 und_weights_q 的 [D]
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_head_dim_falls_back_to_weights)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    para.inputTensorDesc_[3] = {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 所有带 D 的输入都是 -2：D 才真的推不出来，此时 q 为 [-1, Hq, -1]
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_unknown_rank_all_inputs)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    for (size_t idx : {0, 1, 2, 3, 4, 5, 8, 9, 10}) {
        auto dtype = (idx == 3) ? ge::DT_FLOAT : ge::DT_BF16;
        para.inputTensorDesc_[idx] = {{{-2}, {-2}}, dtype, ge::FORMAT_ND};
    }
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, -1},
        {-2},
        {-2},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 已知 rank 但第一维为动态 shape（-1）：T 不可加，第 0 维取 -1，Hq 与 D 仍可推出
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_dynamic_dim_und_len)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-1, 10, 128}, {-1, 10, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
