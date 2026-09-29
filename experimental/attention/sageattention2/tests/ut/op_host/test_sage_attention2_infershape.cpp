/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
void CheckShape(gert::Shape q, gert::Shape k, gert::Shape v,
                ge::graphStatus expectedStatus, gert::Shape expectedLse, bool returnLse = true)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("SageAttention2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::Tensor qt({}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_FLOAT16);
    qt.MutableOriginShape() = q;
    qt.MutableStorageShape() = q;
    gert::Tensor kt({}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_FLOAT16);
    kt.MutableOriginShape() = k;
    kt.MutableStorageShape() = k;
    gert::Tensor vt({}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_FLOAT16);
    vt.MutableOriginShape() = v;
    vt.MutableStorageShape() = v;
    auto holder = gert::OpInferShapeContextBuilder()
        .OpType("SageAttention2").OpName("sage_infer_test").IONum(3, 2)
        .InputTensors({&qt, &kt, &vt})
        .OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .AppendAttr(0.125f).AppendAttr(true).AppendAttr(false).AppendAttr(returnLse).Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_shape(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(*context->GetOutputShape(0), q);
        EXPECT_EQ(*context->GetOutputShape(1), expectedLse);
    }
}

TEST(SageAttention2InferShape, GqaCrossAttention)
{
    CheckShape({2, 8, 33, 64}, {2, 2, 65, 64}, {2, 2, 65, 64}, ge::GRAPH_SUCCESS, {2, 8, 33});
}
TEST(SageAttention2InferShape, LseDisabledStillHasShape)
{
    CheckShape({1, 4, 128, 96}, {1, 2, 512, 96}, {1, 2, 512, 96}, ge::GRAPH_SUCCESS, {1, 4, 128}, false);
}
TEST(SageAttention2InferShape, DynamicDimensions)
{
    CheckShape({-1, 8, -1, 128}, {-1, 2, -1, 128}, {-1, 2, -1, 128}, ge::GRAPH_SUCCESS, {-1, 8, -1});
}
TEST(SageAttention2InferShape, UnknownRank)
{
    CheckShape({-2}, {-2}, {-2}, ge::GRAPH_SUCCESS, {-1, -1, -1});
}
TEST(SageAttention2InferShape, InvalidRanks)
{
    CheckShape({1, 33, 64}, {1, 2, 65, 64}, {1, 2, 65, 64}, ge::GRAPH_FAILED, {});
    CheckShape({1, 4, 33, 64}, {1, 65, 64}, {1, 2, 65, 64}, ge::GRAPH_FAILED, {});
    CheckShape({1, 4, 33, 64}, {1, 2, 65, 64}, {1, 65, 64}, ge::GRAPH_FAILED, {});
}

void CheckDtype(ge::DataType q, ge::DataType k, ge::DataType v, ge::graphStatus expectedStatus)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("SageAttention2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    auto holder = gert::OpInferDataTypeContextBuilder()
        .OpType("SageAttention2").OpName("sage_infer_test").IONum(3, 2)
        .InputTensorDesc(0, q, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensorDesc(1, k, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensorDesc(2, v, ge::FORMAT_ND, ge::FORMAT_ND)
        .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND).Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(0), q);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    }
}
TEST(SageAttention2InferShape, OutputDtypes)
{
    CheckDtype(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS);
    CheckDtype(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS);
}
TEST(SageAttention2InferShape, InvalidDtypes)
{
    CheckDtype(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::GRAPH_FAILED);
}
} // namespace
