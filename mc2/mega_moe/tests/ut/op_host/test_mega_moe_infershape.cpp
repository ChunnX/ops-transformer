/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mega_moe_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;

namespace {
class MegaMoeInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeInfershape TearDown" << std::endl;
    }
};

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

TEST_F(MegaMoeInfershape, BasicShape_H4096) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape expertTokenNumsShape = {{}, {}};
    
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .Build();
    
    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    
    auto outputShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), 128);
    EXPECT_EQ(outputShape->GetDim(1), 4096);
    
    auto expertTokenNumsOutShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    EXPECT_EQ(expertTokenNumsOutShape->GetDimNum(), 1);
    EXPECT_EQ(expertTokenNumsOutShape->GetDim(0), 4);
}

TEST_F(MegaMoeInfershape, BasicShape_H5120) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    
    gert::StorageShape xShape = {{256, 5120}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape expertTokenNumsShape = {{}, {}};
    
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 8, 8, 8, 8, 1, 1})
                     .InputShapes({&xShape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(64)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .Build();
    
    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    
    auto outputShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), 256);
    EXPECT_EQ(outputShape->GetDim(1), 5120);
    
    auto expertTokenNumsOutShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    EXPECT_EQ(expertTokenNumsOutShape->GetDimNum(), 1);
    EXPECT_EQ(expertTokenNumsOutShape->GetDim(0), 8);
}

TEST_F(MegaMoeInfershape, BasicShape_H7168) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    
    gert::StorageShape xShape = {{512, 7168}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape expertTokenNumsShape = {{}, {}};
    
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 2, 2, 2, 2, 1, 1})
                     .InputShapes({&xShape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .Build();
    
    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    
    auto outputShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(outputShape->GetDimNum(), 2);
    EXPECT_EQ(outputShape->GetDim(0), 512);
    EXPECT_EQ(outputShape->GetDim(1), 7168);
    
    auto expertTokenNumsOutShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    EXPECT_EQ(expertTokenNumsOutShape->GetDimNum(), 1);
    EXPECT_EQ(expertTokenNumsOutShape->GetDim(0), 2);
}

TEST_F(MegaMoeInfershape, InvalidH) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    
    gert::StorageShape xShape = {{128, 2048}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape expertTokenNumsShape = {{}, {}};
    
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .Build();
    
    EXPECT_NE(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeInfershape, InvalidEpWorldSize) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape expertTokenNumsShape = {{}, {}};
    
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .Build();
    
    EXPECT_NE(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeInfershape, MoeExpertNumMismatch) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape expertTokenNumsShape = {{}, {}};
    
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(15)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .Build();
    
    EXPECT_NE(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeInfershape, InferDataType_BF16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    
    auto holder = gert::InferDataTypeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputDataType({ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16,
                                     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                                     ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                                     ge::DT_INT8, ge::DT_FLOAT})
                     .OutputDataType({ge::DT_BF16, ge::DT_INT32})
                     .Build();
    
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    
    auto outputDtype = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0);
    EXPECT_EQ(outputDtype, ge::DT_BF16);
}

TEST_F(MegaMoeInfershape, InferDataType_FP16) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    
    auto holder = gert::InferDataTypeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputDataType({ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT,
                                     ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                                     ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                                     ge::DT_INT8, ge::DT_FLOAT})
                     .OutputDataType({ge::DT_FLOAT16, ge::DT_INT32})
                     .Build();
    
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    
    auto outputDtype = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0);
    EXPECT_EQ(outputDtype, ge::DT_FLOAT16);
}

TEST_F(MegaMoeInfershape, InferDataType_InvalidX) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    
    auto holder = gert::InferDataTypeContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputDataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16,
                                     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                                     ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                                     ge::DT_INT8, ge::DT_FLOAT})
                     .OutputDataType({ge::DT_BF16, ge::DT_INT32})
                     .Build();
    
    EXPECT_NE(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

}