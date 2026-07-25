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

#include "infer_shape_case_executor.h"

namespace {
using OpAttr = gert::InfershapeContextPara::OpAttr;

std::vector<gert::InfershapeContextPara::TensorDescription> BuildInputs()
{
    return {
        {{{20, 17, 128}, {20, 17, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256, 128}, {256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{17}, {17}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 2, 128, 128}, {8, 2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{8, 2, 128, 128}, {8, 2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{8, 2, 128, 1}, {8, 2, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<gert::InfershapeContextPara::TensorDescription> BuildOutputs()
{
    return {
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> BuildAttrs(const std::vector<int64_t> &headNums, const std::string &layoutQkv = "NTD",
                               const std::string &layoutQOut = "NTD", float epsilon = 1e-6f)
{
    return {
        {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
        {"layout_qkv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQkv)},
        {"layout_q_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQOut)},
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
    };
}

} // namespace

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums));

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128}, {16, 17}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersTndOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "TND", "TND"));

    std::vector<std::vector<int64_t>> expected = {
        {17, 16, 128}, {17, 16}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersTndInputNtdOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "TND", "NTD"));

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128}, {16, 17}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidLayoutQkv)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "tnd"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersDefaultLayoutsWhenLayoutAttrsMissing)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para(
        "QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)}});

    std::vector<std::vector<int64_t>> expected = {
        {16, 20, 128}, {16, 20}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersDefaultLayoutsWhenLayoutAttrsEmpty)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "", ""));

    std::vector<std::vector<int64_t>> expected = {
        {16, 20, 128}, {16, 20}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsQkvHeadLessThanNq)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 15, 128}, {17, 15, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "TND", "TND"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidLayoutQOut)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "NTD", "ntd"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsNtdInputTndLayoutQOut)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "NTD", "TND"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidHeadNumsAttr)
{
    std::vector<int64_t> headNums = {};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);

    headNums = {0, 2, 2};
    gert::InfershapeContextPara zeroHeadPara("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                             BuildAttrs(headNums));
    ExecuteTestCase(zeroHeadPara, ge::GRAPH_FAILED, expected);

    headNums = {-1, 2, 2};
    gert::InfershapeContextPara negativeHeadPara("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                                 BuildAttrs(headNums));
    ExecuteTestCase(negativeHeadPara, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidInputDimNum)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    std::vector<std::vector<int64_t>> expected;

    auto inputs = BuildInputs();
    inputs[0] = {{{20, 17}, {20, 17}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara qkvPara("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(), BuildAttrs(headNums));
    ExecuteTestCase(qkvPara, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsNonPositiveInputDims)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    std::vector<std::vector<int64_t>> expected;

    auto inputs = BuildInputs();
    inputs[0] = {{{20, -1, 128}, {20, -1, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara qkvPara("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(), BuildAttrs(headNums));
    ExecuteTestCase(qkvPara, ge::GRAPH_FAILED, expected);

    inputs = BuildInputs();
    inputs[0] = {{{20, 17, 0}, {20, 17, 0}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara qkvHeadDimPara("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                               BuildAttrs(headNums));
    ExecuteTestCase(qkvHeadDimPara, ge::GRAPH_FAILED, expected);
}
