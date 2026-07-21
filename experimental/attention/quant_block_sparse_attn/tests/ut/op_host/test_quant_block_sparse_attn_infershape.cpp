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
#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"

class QuantBlockSparseAttnInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantBlockSparseAttnInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBlockSparseAttnInfershape TearDown" << std::endl;
    }
};

namespace {
using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

std::vector<TensorDesc> MakeInputs(int64_t d0, int64_t d1, int64_t d2)
{
    return {
        {{{d0, d1, d2}, {d0, d1, d2}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{d1}, {d1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, d1, 2, 4}, {1, d1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, d1, 2}, {1, d1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeOutputs()
{
    return {
        {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeAttrs(const std::string &layoutQ, bool returnLse)
{
    return {
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"paBlockStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>("B_N_Qb_Kb")},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)},
    };
}
} // namespace

TEST_F(QuantBlockSparseAttnInfershape, infershape_tnd_no_lse)
{
    gert::InfershapeContextPara para("QuantBlockSparseAttn", MakeInputs(256, 4, 128), MakeOutputs(),
                                     MakeAttrs("TND", false));

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 4, 128},
        {0},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantBlockSparseAttnInfershape, infershape_tnd_with_lse)
{
    gert::InfershapeContextPara para("QuantBlockSparseAttn", MakeInputs(256, 4, 128), MakeOutputs(),
                                     MakeAttrs("TND", true));

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 4, 128},
        {4, 256},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantBlockSparseAttnInfershape, infershape_ntd_with_lse)
{
    gert::InfershapeContextPara para("QuantBlockSparseAttn", MakeInputs(4, 256, 128), MakeOutputs(),
                                     MakeAttrs("NTD", true));

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 4, 128},
        {4, 256},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantBlockSparseAttnInfershape, infershape_2d_query_with_lse_failed)
{
    gert::InfershapeContextPara para("QuantBlockSparseAttn",
                                     {
                                         {{{256, 128}, {256, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                         {{{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                         {{{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1, 4, 2, 4}, {1, 4, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1, 4, 2}, {1, 4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     MakeOutputs(), MakeAttrs("TND", true));

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(QuantBlockSparseAttnInfershape, infershape_bsnd_4d_with_lse)
{
    gert::InfershapeContextPara para("QuantBlockSparseAttn",
                                     {
                                         {{{4, 256, 4, 128}, {4, 256, 4, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                         {{{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                         {{{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{4, 4, 2, 4}, {4, 4, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{4, 4, 2}, {4, 4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     MakeOutputs(), MakeAttrs("BSND", true));

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(QuantBlockSparseAttnInfershape, infershape_inferdatatype_non_bf16)
{
    ge::DataType outputDtype0 = ge::DT_FLOAT;
    ge::DataType outputDtype1 = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("QuantBlockSparseAttn")
                             .NodeIoNum(16, 2)
                             .OutputDataTypes({&outputDtype0, &outputDtype1})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("QuantBlockSparseAttn")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT);
}

TEST_F(QuantBlockSparseAttnInfershape, infershape_inferdatatype_already_bf16)
{
    ge::DataType outputDtype0 = ge::DT_BF16;
    ge::DataType outputDtype1 = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("QuantBlockSparseAttn")
                             .NodeIoNum(16, 2)
                             .OutputDataTypes({&outputDtype0, &outputDtype1})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("QuantBlockSparseAttn")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT);
}
