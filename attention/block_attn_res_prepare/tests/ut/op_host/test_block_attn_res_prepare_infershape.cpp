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
 * \\file test_block_attn_res_prepare_infershape.cpp
 * \\brief BlockAttnResPrepare shape and dtype inference unit tests.
 */

#include <vector>

#include "gtest/gtest.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {

using TensorDesc = gert::InfershapeContextPara::TensorDescription;

std::vector<TensorDesc> MakeInputs(const std::initializer_list<int64_t> &blockResShape,
                                   const std::initializer_list<int64_t> &pseudoQueryShape)
{
    return {
        TensorDesc(gert::StorageShape(blockResShape, blockResShape), ge::DT_FLOAT, ge::FORMAT_ND),
        TensorDesc(gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND),
        TensorDesc(gert::StorageShape(pseudoQueryShape, pseudoQueryShape), ge::DT_FLOAT,
                   ge::FORMAT_ND),
    };
}

std::vector<TensorDesc> MakeOutputs()
{
    return {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

TEST(BlockAttnResPrepareInfershapeTest, Basic)
{
    gert::InfershapeContextPara para("BlockAttnResPrepare", MakeInputs({2, 8, 128}, {3, 128}), MakeOutputs(), {});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3, 2, 128}, {3, 2}, {3, 2}});
}

TEST(BlockAttnResPrepareInfershapeTest, SupportsEmptyT)
{
    gert::InfershapeContextPara para("BlockAttnResPrepare", MakeInputs({0, 8, 128}, {3, 128}), MakeOutputs(), {});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3, 0, 128}, {3, 0}, {3, 0}});
}

TEST(BlockAttnResPrepareInfershapeTest, SupportsEmptyS)
{
    gert::InfershapeContextPara para("BlockAttnResPrepare", MakeInputs({2, 8, 128}, {0, 128}), MakeOutputs(), {});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{0, 2, 128}, {0, 2}, {0, 2}});
}

TEST(BlockAttnResPrepareInfershapeTest, RejectsInvalidBlockResRank)
{
    gert::InfershapeContextPara para("BlockAttnResPrepare", MakeInputs({2, 128}, {3, 128}), MakeOutputs(), {});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(BlockAttnResPrepareInfershapeTest, RejectsInvalidPseudoQueryRank)
{
    gert::InfershapeContextPara para("BlockAttnResPrepare", MakeInputs({2, 8, 128}, {3, 1, 128}), MakeOutputs(), {});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(BlockAttnResPrepareInfershapeTest, RejectsMismatchedHeadDim)
{
    gert::InfershapeContextPara para("BlockAttnResPrepare", MakeInputs({2, 8, 128}, {3, 64}), MakeOutputs(), {});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(BlockAttnResPrepareInfershapeTest, OutputDtypeFollowsBlockRes)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto inferDtype = spaceRegistry->GetOpImpl("BlockAttnResPrepare")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);

    ge::DataType fp32 = ge::DT_FLOAT;
    ge::DataType int32 = ge::DT_INT32;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(3)
                             .NodeIoNum(3, 3)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&fp32, &int32, &fp32})
                             .OutputDataTypes({&fp32, &fp32, &fp32})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDtype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

} // namespace
