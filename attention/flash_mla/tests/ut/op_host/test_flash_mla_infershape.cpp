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

#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
gert::StorageShape StorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    for (auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::InfershapeContextPara MakeInferCase(const std::vector<int64_t> &query, const std::string &layout,
                                          int64_t heads = 8, bool lse = true)
{
    return gert::InfershapeContextPara(
        "FlashMla", {{StorageShape(query), ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(heads)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125F)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT32_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT32_MAX)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(lse)}});
}

struct InferCase {
    const char *name;
    const char *layout;
    std::vector<int64_t> query;
    std::vector<int64_t> output;
    std::vector<int64_t> lse;
};

class FlashMlaInferShape : public testing::TestWithParam<InferCase> {};

TEST_P(FlashMlaInferShape, OutputAndLseLayouts)
{
    const auto &test = GetParam();
    for (bool lse : {false, true}) {
        SCOPED_TRACE(lse);
        auto para = MakeInferCase(test.query, test.layout, 8, lse);
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, {test.output, lse ? test.lse : std::vector<int64_t>{0}});
    }
}

INSTANTIATE_TEST_SUITE_P(
    Layouts, FlashMlaInferShape,
    testing::Values(InferCase{"Bsh", "BSH", {2, 3, 4096}, {2, 3, 4096}, {2, 8, 3, 1}},
                    InferCase{"Bsnd", "BSND", {2, 3, 8, 512}, {2, 3, 8, 512}, {2, 8, 3, 1}},
                    InferCase{"Bnsd", "BNSD", {2, 8, 3, 512}, {2, 8, 3, 512}, {2, 8, 3, 1}},
                    InferCase{"Tnd", "TND", {6, 8, 512}, {6, 8, 512}, {6, 8, 1}},
                    InferCase{"BshNbsd", "BSH_NBSD", {2, 3, 4096}, {8, 2, 3, 512}, {2, 8, 3, 1}},
                    InferCase{"BsndNbsd", "BSND_NBSD", {2, 3, 8, 512}, {8, 2, 3, 512}, {2, 8, 3, 1}},
                    InferCase{"BnsdNbsd", "BNSD_NBSD", {2, 8, 3, 512}, {8, 2, 3, 512}, {2, 8, 3, 1}},
                    InferCase{"TndNtd", "TND_NTD", {6, 8, 512}, {8, 6, 512}, {6, 8, 1}}),
    [](const testing::TestParamInfo<InferCase> &info) { return info.param.name; });

struct InvalidInferCase {
    const char *name;
    const char *layout;
    std::vector<int64_t> query;
    int64_t heads = 8;
};

class FlashMlaInvalidInferShape : public testing::TestWithParam<InvalidInferCase> {};

TEST_P(FlashMlaInvalidInferShape, RejectsInvalidQueryContract)
{
    const auto &test = GetParam();
    auto para = MakeInferCase(test.query, test.layout, test.heads);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

INSTANTIATE_TEST_SUITE_P(Invalid, FlashMlaInvalidInferShape,
                         testing::Values(InvalidInferCase{"UnknownLayout", "BSHD", {2, 3, 4096}},
                                         InvalidInferCase{"RepeatedLayoutSuffix", "BSH_BSH", {2, 3, 4096}},
                                         InvalidInferCase{"DenseNtdSuffix", "BSH_NTD", {2, 3, 4096}},
                                         InvalidInferCase{"TndNbsdSuffix", "TND_NBSD", {6, 8, 512}},
                                         InvalidInferCase{"ExtraSuffix", "BNSD_NBSD_NBSD", {2, 8, 3, 512}},
                                         InvalidInferCase{"BshRank", "BSH", {2, 3, 8, 512}},
                                         InvalidInferCase{"BnsdRank", "BNSD", {2, 3, 4096}},
                                         InvalidInferCase{"BshHeadRemainder", "BSH", {2, 3, 4097}},
                                         InvalidInferCase{"HeadMismatch", "BSND", {2, 3, 4, 512}},
                                         InvalidInferCase{"ZeroHeads", "BNSD", {2, 8, 3, 512}, 0},
                                         InvalidInferCase{"NegativeHeads", "BNSD", {2, 8, 3, 512}, -1},
                                         InvalidInferCase{"DynamicDimension", "BNSD", {2, 8, -1, 512}}),
                         [](const testing::TestParamInfo<InvalidInferCase> &info) { return info.param.name; });

TEST(FlashMlaInferShapeBoundary, EmptyBatchAndTokens)
{
    auto batch = MakeInferCase({0, 8, 3, 512}, "BNSD_NBSD");
    ExecuteTestCase(batch, ge::GRAPH_SUCCESS, {{8, 0, 3, 512}, {0, 8, 3, 1}});
    auto tokens = MakeInferCase({0, 8, 512}, "TND_NTD");
    ExecuteTestCase(tokens, ge::GRAPH_SUCCESS, {{8, 0, 512}, {0, 8, 1}});
}

TEST(FlashMlaInferShapeBoundary, MissingAttributes)
{
    auto para = MakeInferCase({2, 8, 3, 512}, "BNSD");
    para.attrs_.clear();
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
    para = MakeInferCase({2, 8, 3, 512}, "BNSD");
    para.attrs_.pop_back(); // LSE flag occupies attribute slot 9.
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

class FlashMlaInferDataType : public testing::TestWithParam<ge::DataType> {};

TEST_P(FlashMlaInferDataType, PreservesFloatTypesAndConvertsInt8ToBf16)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("FlashMla");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    ge::DataType query = GetParam();
    ge::DataType output = ge::DT_UNDEFINED;
    ge::DataType lse = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("FlashMla")
                      .IrInputNum(1)
                      .NodeIoNum(1, 2)
                      .NodeInputTd(0, query, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&query})
                      .OutputDataTypes({&output, &lse})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), query == ge::DT_INT8 ? ge::DT_BF16 : query);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}

INSTANTIATE_TEST_SUITE_P(Supported, FlashMlaInferDataType, testing::Values(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8));

TEST(FlashMlaInferShapeBoundary, NullContexts)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("FlashMla");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    EXPECT_EQ(impl->infer_shape(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(impl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}
} // namespace
