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
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
using AttrList = std::vector<gert::InfershapeContextPara::OpAttr>;
using Td = gert::InfershapeContextPara::TensorDescription;

AttrList MakeAttrs(int64_t topk, int64_t cmpRatio, const char *layoutQ, const char *layoutK, int64_t sparseMode,
                   int64_t actType, bool hasWeighted, bool expand, bool append, int64_t quantMode, bool returnValue,
                   int64_t selectMode = 0, int64_t keyReduce = 0, float scoreScale = 1.0f)
{
    return {
        {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
        {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutK)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
        {"has_weighted_sum", Ops::Transformer::AnyValue::CreateFrom<bool>(hasWeighted)},
        {"expand_indices", Ops::Transformer::AnyValue::CreateFrom<bool>(expand)},
        {"append_tail", Ops::Transformer::AnyValue::CreateFrom<bool>(append)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(returnValue)},
        {"select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(selectMode)},
        {"key_reduce_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(keyReduce)},
        {"score_scale", Ops::Transformer::AnyValue::CreateFrom<float>(scoreScale)},
        {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"k_descale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
}

Td NullIn(ge::DataType dt = ge::DT_INT32)
{
    return {{{}, {}}, dt, ge::FORMAT_ND};
}

std::vector<Td> GliInputs(Td query, Td key, Td weights, Td residual = NullIn(), Td actQ = NullIn(), Td actK = NullIn(),
                          Td qDescale = NullIn(ge::DT_FLOAT), Td kDescale = NullIn(ge::DT_FLOAT),
                          Td blockTable = NullIn())
{
    return {query,    key,        weights,  actQ,     actK,     NullIn(), NullIn(),
            residual, blockTable, qDescale, kDescale, NullIn(), NullIn(), NullIn()};
}
} // namespace

class GenericLightningIndexerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GenericLightningIndexerProto SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "GenericLightningIndexerProto TearDown" << std::endl;
    }
};

TEST_F(GenericLightningIndexerProto, P_LI_BSND)
{
    gert::InfershapeContextPara para("GenericLightningIndexer",
                                     GliInputs({{{1, 32, 8, 128}, {1, 32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 32, 8}, {1, 32, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
                                     {NullIn(), NullIn(ge::DT_FLOAT)},
                                     MakeAttrs(2048, 1, "BSND", "BSND", 3, 0, true, false, false, -1, true));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 32, 2048}, {1, 32, 2048}});
}

TEST_F(GenericLightningIndexerProto, P_LIv2_cmp_ratio4)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{2, 16, 64, 128}, {2, 16, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                  {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                  {{{2, 16, 64}, {2, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND}, {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(2048, 4, "BSND", "BSND", 3, 0, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 16, 2048}, {0}});
}

TEST_F(GenericLightningIndexerProto, P_QLI_quant_int8)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                  {{{1, 512, 1, 128}, {1, 512, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(), NullIn(),
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 512, 1}, {1, 512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(1024, 1, "BSND", "BSND", 0, 0, true, false, false, 2, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 8, 1024}, {0}});
}

TEST_F(GenericLightningIndexerProto, P_QLIv2_mxfp8)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                  {{{1, 128, 1, 128}, {1, 128, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(2048, 4, "BSND", "BSND", 3, 0, true, false, false, 3, true));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 8, 2048}, {1, 8, 2048}});
}

TEST_F(GenericLightningIndexerProto, P_PKI_expand_append)
{
    const int64_t topk = 128;
    const int64_t r = 16;
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 64, 8, 128}, {1, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)},
        MakeAttrs(topk, r, "BSND", "BSND", 3, 0, true, true, true, -1, false, 0, 0, 1.0f / 11.313708f));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 64, topk * r + (r - 1)}, {0}});
}

TEST_F(GenericLightningIndexerProto, P_QWEN_sum_block_causal)
{
    const int64_t topk = 64;
    const int64_t r = 4;
    gert::InfershapeContextPara para("GenericLightningIndexer",
                                     GliInputs({{{1, 32, 8, 128}, {1, 32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               NullIn(ge::DT_FLOAT16), {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}),
                                     {NullIn(), NullIn(ge::DT_FLOAT)},
                                     MakeAttrs(topk, r, "BSND", "BSND", 4, 0, false, true, true, -1, true));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 32, topk * r + (r - 1)}, {1, 32, topk}});
}

TEST_F(GenericLightningIndexerProto, TND_LIv2)
{
    int64_t actualQ[] = {8, 16};
    int64_t actualK[] = {32, 64};
    gert::InfershapeContextPara para("GenericLightningIndexer",
                                     GliInputs({{{16, 8, 128}, {16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(),
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualQ},
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualK}),
                                     {NullIn(), NullIn(ge::DT_FLOAT)},
                                     MakeAttrs(512, 1, "TND", "TND", 0, 0, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{16, 512}, {0}});
}

TEST_F(GenericLightningIndexerProto, P_MSA_identity_maxpool)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16)),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(16, 16, "BSND", "BSND", 3, 1, false, false, false, -1, true, 1, 1));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{0}, {1, 8, 8, 16}});
}

TEST_F(GenericLightningIndexerProto, P_MSA_REJ_incomplete_combo)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16)),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(32, 1, "BSND", "BSND", 0, 1, false, false, false, -1, false, 0, 0));
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(GenericLightningIndexerProto, Reject_append_without_expand)
{
    gert::InfershapeContextPara para("GenericLightningIndexer",
                                     GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
                                     {NullIn(), NullIn(ge::DT_FLOAT)},
                                     MakeAttrs(32, 4, "BSND", "BSND", 0, 0, true, false, true, -1, false));
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(GenericLightningIndexerProto, Reject_identity_act_step1)
{
    gert::InfershapeContextPara para("GenericLightningIndexer",
                                     GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
                                     {NullIn(), NullIn(ge::DT_FLOAT)},
                                     MakeAttrs(32, 1, "BSND", "BSND", 0, 1, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(GenericLightningIndexerProto, PA_BSND_alias)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{8, 128, 1, 128}, {8, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(),
                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT), NullIn(ge::DT_FLOAT),
                  {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(128, 1, "BSND", "PA_BSND", 3, 0, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 8, 128}, {0}});
}

TEST_F(GenericLightningIndexerProto, PA_BBND_alias)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{8, 128, 1, 128}, {8, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(),
                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT), NullIn(ge::DT_FLOAT),
                  {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(128, 1, "BSND", "BBND", 3, 0, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 8, 128}, {0}});
}

TEST_F(GenericLightningIndexerProto, PA_BNBD_alias)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{8, 1, 128, 128}, {8, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(),
                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT), NullIn(ge::DT_FLOAT),
                  {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND}),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(128, 1, "BSND", "BNBD", 3, 0, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 8, 128}, {0}});
}

TEST_F(GenericLightningIndexerProto, TND_cu_seqlens_Bplus1)
{
    int64_t actualQ[] = {0, 8, 16};
    int64_t actualK[] = {0, 32, 64};
    gert::InfershapeContextPara para("GenericLightningIndexer",
                                     GliInputs({{{16, 8, 128}, {16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(),
                                               {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actualQ},
                                               {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actualK}),
                                     {NullIn(), NullIn(ge::DT_FLOAT)},
                                     MakeAttrs(512, 1, "TND", "TND", 0, 0, true, false, false, -1, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{16, 512}, {0}});
}

TEST_F(GenericLightningIndexerProto, P_MSA_TND_score_NT)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{16, 8, 128}, {16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{32, 1, 128}, {32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16)),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(16, 16, "TND", "TND", 3, 1, false, false, false, -1, true, 1, 1));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{0}, {8, 16, 16}});
}

TEST_F(GenericLightningIndexerProto, P_MSA_mixed_fp16_int8)
{
    gert::InfershapeContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16)),
        {NullIn(), NullIn(ge::DT_FLOAT)}, MakeAttrs(16, 16, "BSND", "BSND", 3, 1, false, false, false, -1, true, 1, 1));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{0}, {1, 8, 8, 16}});
}

TEST_F(GenericLightningIndexerProto, InferDataType)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("GenericLightningIndexer")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType fp16 = ge::DT_FLOAT16;
    ge::DataType i32 = ge::DT_INT32;
    ge::DataType f32 = ge::DT_FLOAT;
    std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> dtypeAttrs = {
        {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
        {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"has_weighted_sum", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"expand_indices", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"append_tail", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"key_reduce_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"score_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"k_descale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
    auto holder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(14, 2)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&fp16, &fp16, &fp16, &i32, &i32, &i32, &i32, &i32, &i32, &f32, &f32, &i32, &i32, &i32})
            .NodeAttrs(dtypeAttrs)
            .Build();
    auto ctx = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(ctx->GetOutputDataType(1), ge::DT_FLOAT);
}
