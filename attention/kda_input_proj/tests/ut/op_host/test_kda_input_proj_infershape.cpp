/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_kda_input_proj_infershape.cpp
 * \brief InferShape / InferDataType UT for KdaInputProj.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
constexpr int64_t kT = 8;
constexpr int64_t kHidden = 256;
constexpr int64_t kQkv = 128;
constexpr int64_t kBeta = 16;
constexpr int64_t kGate = 64;
constexpr int64_t kG = 64;
constexpr int64_t kMxBlock = 64;
constexpr int64_t kScalePack = 2;

gert::StorageShape Shape2D(int64_t d0, int64_t d1)
{
    return {{d0, d1}, {d0, d1}};
}

gert::StorageShape Shape3D(int64_t d0, int64_t d1, int64_t d2)
{
    return {{d0, d1, d2}, {d0, d1, d2}};
}

gert::StorageShape WeightShape(int64_t nSize, int64_t kSize, bool trans)
{
    return trans ? Shape2D(nSize, kSize) : Shape2D(kSize, nSize);
}

gert::StorageShape ScaleShape(int64_t nSize, int64_t kSize, bool transQkv)
{
    const int64_t mxK = kSize > 0 ? (kSize + kMxBlock - 1) / kMxBlock : -1;
    return transQkv ? Shape3D(nSize, mxK, kScalePack) : Shape3D(mxK, nSize, kScalePack);
}

std::vector<gert::InfershapeContextPara::OpAttr> MakeAttrs(bool transQkv, bool transBeta, bool transGate, bool transG)
{
    return {
        {"trans_weight_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(transQkv)},
        {"trans_weight_beta", Ops::Transformer::AnyValue::CreateFrom<bool>(transBeta)},
        {"trans_weight_gate", Ops::Transformer::AnyValue::CreateFrom<bool>(transGate)},
        {"trans_weight_g", Ops::Transformer::AnyValue::CreateFrom<bool>(transG)},
    };
}

gert::InfershapeContextPara BuildInfer(int64_t tSize, int64_t hidden, int64_t nQkv, int64_t nBeta, int64_t nGate,
                                       int64_t nG, bool transQkv = true, bool transBeta = true, bool transGate = true,
                                       bool transG = true, bool withAttrs = true, int64_t xRank = 2,
                                       int64_t scaleDim2 = kScalePack)
{
    gert::StorageShape xShape = xRank == 3 ? Shape3D(tSize, hidden, 1) : Shape2D(tSize, hidden);
    gert::StorageShape scale = ScaleShape(nQkv, hidden, transQkv);
    if (scaleDim2 != kScalePack) {
        const int64_t mxK = hidden > 0 ? (hidden + kMxBlock - 1) / kMxBlock : -1;
        scale = transQkv ? Shape3D(nQkv, mxK, scaleDim2) : Shape3D(mxK, nQkv, scaleDim2);
    }
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = withAttrs ?
                                                                 MakeAttrs(transQkv, transBeta, transGate, transG) :
                                                                 std::vector<gert::InfershapeContextPara::OpAttr>{};
    return gert::InfershapeContextPara("KdaInputProj",
                                       {
                                           {xShape, ge::DT_BF16, ge::FORMAT_ND},
                                           {WeightShape(nQkv, hidden, transQkv), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                           {WeightShape(nBeta, hidden, transBeta), ge::DT_BF16, ge::FORMAT_ND},
                                           {WeightShape(nGate, hidden, transGate), ge::DT_BF16, ge::FORMAT_ND},
                                           {WeightShape(nG, hidden, transG), ge::DT_BF16, ge::FORMAT_ND},
                                           {scale, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                       },
                                       {
                                           {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                       },
                                       attrs);
}

void CheckInferDataTypeSuccess()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("KdaInputProj");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType bf16 = ge::DT_BF16;
    ge::DataType fp8 = ge::DT_FLOAT8_E4M3FN;
    ge::DataType e8m0 = ge::DT_FLOAT8_E8M0;
    ge::DataType qkvOut = ge::DT_FLOAT;
    ge::DataType betaOut = ge::DT_FLOAT16;
    ge::DataType gateOut = ge::DT_FLOAT;
    ge::DataType gOut = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("KdaInputProj")
                             .NodeIoNum(6, 4)
                             .NodeInputTd(0, bf16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, fp8, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, bf16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, bf16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, bf16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, e8m0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&bf16, &fp8, &bf16, &bf16, &bf16, &e8m0})
                             .OutputDataTypes({&qkvOut, &betaOut, &gateOut, &gOut})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}

void CheckInferDataTypeFail(int inputIdx, ge::DataType badDtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("KdaInputProj");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType dtypes[6] = {ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16,
                              ge::DT_BF16, ge::DT_BF16,          ge::DT_FLOAT8_E8M0};
    dtypes[inputIdx] = badDtype;
    ge::DataType qkvOut = ge::DT_BF16;
    ge::DataType betaOut = ge::DT_FLOAT;
    ge::DataType gateOut = ge::DT_BF16;
    ge::DataType gOut = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("KdaInputProj")
                             .NodeIoNum(6, 4)
                             .NodeInputTd(0, dtypes[0], ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, dtypes[1], ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, dtypes[2], ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, dtypes[3], ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, dtypes[4], ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, dtypes[5], ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&dtypes[0], &dtypes[1], &dtypes[2], &dtypes[3], &dtypes[4], &dtypes[5]})
                             .OutputDataTypes({&qkvOut, &betaOut, &gateOut, &gOut})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_FAILED);
}
} // namespace

class KdaInputProjInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KdaInputProjInferShape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "KdaInputProjInferShape TearDown" << std::endl;
    }
};

TEST_F(KdaInputProjInferShape, infershape_trans_true)
{
    auto para = BuildInfer(kT, kHidden, kQkv, kBeta, kGate, kG, true, true, true, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kT, kQkv}, {kT, kBeta}, {kT, kGate}, {kT, kG}});
}

TEST_F(KdaInputProjInferShape, infershape_trans_false)
{
    auto para = BuildInfer(kT, kHidden, kQkv, kBeta, kGate, kG, false, false, false, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kT, kQkv}, {kT, kBeta}, {kT, kGate}, {kT, kG}});
}

TEST_F(KdaInputProjInferShape, infershape_mixed_trans)
{
    auto para = BuildInfer(kT, kHidden, kQkv, kBeta, kGate, kG, true, false, true, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kT, kQkv}, {kT, kBeta}, {kT, kGate}, {kT, kG}});
}

TEST_F(KdaInputProjInferShape, infershape_default_attrs)
{
    auto para = BuildInfer(kT, kHidden, kQkv, kBeta, kGate, kG, true, true, true, true, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kT, kQkv}, {kT, kBeta}, {kT, kGate}, {kT, kG}});
}

TEST_F(KdaInputProjInferShape, infershape_unknown_t)
{
    auto para = BuildInfer(-1, kHidden, kQkv, kBeta, kGate, kG, false, false, false, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-1, kQkv}, {-1, kBeta}, {-1, kGate}, {-1, kG}});
}

TEST_F(KdaInputProjInferShape, infershape_unknown_out_features)
{
    auto para = BuildInfer(kT, kHidden, -1, -1, -1, -1, false, false, false, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kT, -1}, {kT, -1}, {kT, -1}, {kT, -1}});
}

TEST_F(KdaInputProjInferShape, infershape_fail_x_rank)
{
    auto para = BuildInfer(kT, kHidden, kQkv, kBeta, kGate, kG, true, true, true, true, true, 3);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_fail_weight_k_mismatch)
{
    // trans=true 时 weight=[N, K]，K 必须等于 x.dim1
    gert::InfershapeContextPara bad("KdaInputProj",
                                    {
                                        {Shape2D(kT, kHidden), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kQkv, kHidden, true), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                        {WeightShape(kBeta, kHidden + 1, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kGate, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kG, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {ScaleShape(kQkv, kHidden, true), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                    },
                                    MakeAttrs(true, true, true, true));
    ExecuteTestCase(bad, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_fail_scale_layout)
{
    gert::InfershapeContextPara para(
        "KdaInputProj",
        {
            {Shape2D(kT, kHidden), ge::DT_BF16, ge::FORMAT_ND},
            {WeightShape(kQkv, kHidden, false), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {WeightShape(kBeta, kHidden, false), ge::DT_BF16, ge::FORMAT_ND},
            {WeightShape(kGate, kHidden, false), ge::DT_BF16, ge::FORMAT_ND},
            {WeightShape(kG, kHidden, false), ge::DT_BF16, ge::FORMAT_ND},
            {ScaleShape(kQkv, kHidden, true), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // trans=true 排布，但 attr 是 false
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        MakeAttrs(false, false, false, false));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_fail_scale_pack)
{
    auto para = BuildInfer(kT, kHidden, kQkv, kBeta, kGate, kG, true, true, true, true, true, 2, 1);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_fail_invalid_dim)
{
    auto para = BuildInfer(0, kHidden, kQkv, kBeta, kGate, kG);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_fail_weight_rank)
{
    gert::InfershapeContextPara bad("KdaInputProj",
                                    {
                                        {Shape2D(kT, kHidden), ge::DT_BF16, ge::FORMAT_ND},
                                        {Shape3D(kQkv, kHidden, 1), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                        {WeightShape(kBeta, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kGate, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kG, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {ScaleShape(kQkv, kHidden, true), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                    },
                                    MakeAttrs(true, true, true, true));
    ExecuteTestCase(bad, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_fail_scale_rank)
{
    gert::InfershapeContextPara bad("KdaInputProj",
                                    {
                                        {Shape2D(kT, kHidden), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kQkv, kHidden, true), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                        {WeightShape(kBeta, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kGate, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {WeightShape(kG, kHidden, true), ge::DT_BF16, ge::FORMAT_ND},
                                        {Shape2D(kQkv, kHidden / kMxBlock), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                    },
                                    MakeAttrs(true, true, true, true));
    ExecuteTestCase(bad, ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjInferShape, infershape_unknown_hidden)
{
    auto para = BuildInfer(kT, -1, kQkv, kBeta, kGate, kG, true, true, true, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kT, kQkv}, {kT, kBeta}, {kT, kGate}, {kT, kG}});
}

TEST_F(KdaInputProjInferShape, infer_datatype_success)
{
    CheckInferDataTypeSuccess();
}

TEST_F(KdaInputProjInferShape, infer_datatype_fail_x_fp16)
{
    CheckInferDataTypeFail(0, ge::DT_FLOAT16);
}

TEST_F(KdaInputProjInferShape, infer_datatype_fail_weight_qkv)
{
    CheckInferDataTypeFail(1, ge::DT_BF16);
}

TEST_F(KdaInputProjInferShape, infer_datatype_fail_scale)
{
    CheckInferDataTypeFail(5, ge::DT_FLOAT);
}
