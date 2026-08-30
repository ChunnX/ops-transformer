/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_context_faker.h"

namespace {
constexpr const char *OP_TYPE = "MoeTutelDispatch";

const gert::OpImplKernelRegistry::OpImplFunctions *GetOpImpl()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    return spaceRegistry == nullptr ? nullptr : spaceRegistry->GetOpImpl(OP_TYPE);
}

std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    std::vector<int64_t> result(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        result[i] = shape.GetDim(i);
    }
    return result;
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}
} // namespace

class MoeTutelDispatchInferDataTypeTest : public testing::Test {};

TEST_F(MoeTutelDispatchInferDataTypeTest, RegistersShapeAndDtypeInference)
{
    auto *opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
}

TEST_F(MoeTutelDispatchInferDataTypeTest, ShapeInferencePreservesCallerDescriptor)
{
    auto *opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    const std::vector<std::vector<int64_t>> cases = {
        {2, 5, 8},  // E < K
        {5, 5, 8},  // E > K
        {-1, 5, 8}, // dynamic E
    };
    for (const auto &expected : cases) {
        gert::StorageShape outputShape = MakeStorageShape(expected);
        auto contextHolder = gert::InferShapeContextFaker()
                                 .SetOpType(OP_TYPE)
                                 .NodeIoNum(4, 1)
                                 .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .Build();
        auto *context = contextHolder.GetContext();
        ASSERT_NE(context, nullptr);
        // InferShapeContextFaker intentionally starts output slots empty. GE
        // populates this slot from the graph's output descriptor before the
        // callback; seed it explicitly to model that caller-provided state.
        *context->GetOutputShape(0) = outputShape.GetStorageShape();
        EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ToVector(*context->GetOutputShape(0)), expected);
    }
}

TEST_F(MoeTutelDispatchInferDataTypeTest, NullShapeContextRejected)
{
    auto *opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    EXPECT_EQ(opImpl->infer_shape(nullptr), ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchInferDataTypeTest, OutputFollowsX)
{
    auto *opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    const ge::DataType cases[] = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    for (const ge::DataType dtype : cases) {
        ge::DataType xType = dtype;
        ge::DataType gateType = dtype;
        ge::DataType indexType = ge::DT_INT32;
        ge::DataType locationType = ge::DT_INT32;
        ge::DataType outputType = ge::DT_UNDEFINED;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .SetOpType(OP_TYPE)
                                 .IrInputNum(4)
                                 .NodeIoNum(4, 1)
                                 .NodeInputTd(0, xType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, gateType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, indexType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(3, locationType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&xType, &gateType, &indexType, &locationType})
                                 .OutputDataTypes({&outputType})
                                 .Build();
        auto *context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
    }
}

TEST_F(MoeTutelDispatchInferDataTypeTest, NullContextRejected)
{
    auto *opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}
