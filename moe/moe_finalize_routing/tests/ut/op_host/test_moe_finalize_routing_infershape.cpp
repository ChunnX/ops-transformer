/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeFinalizeRoutingInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingProto TearDown" << std::endl;
    }
};

struct MoeFinalizeRoutingInfo {
    gert::StorageShape &expandedXShape;
    gert::StorageShape &x1Shape;
    gert::StorageShape &x2Shape;
    gert::StorageShape &biasShape;
    gert::StorageShape &scalesShape;
    gert::StorageShape &expandedRowIdxShape;
    gert::StorageShape &expandedExpertIdxShape;
    std::vector<int64_t> expectOutShape;

    ge::DataType expandedXDtype;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType biasDtype;
    ge::DataType scalesDtype;
    ge::DataType expandedRowIdxDtype;
    ge::DataType expandedExpertIdxDtype;
    ge::DataType yDtype;

    int64_t dropPadMode = 0;
};

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const MoeFinalizeRoutingInfo ioInfo, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    gert::StorageShape yStorageShape = {};
    std::vector<gert::StorageShape *> ouputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("MoeFinalizeRouting")
                             .NodeIoNum(7, 1)
                             .NodeInputTd(0, ioInfo.expandedXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ioInfo.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ioInfo.x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ioInfo.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ioInfo.scalesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ioInfo.expandedRowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, ioInfo.expandedExpertIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ioInfo.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)

                             .InputTensors({(gert::Tensor *)&ioInfo.expandedXShape})
                             .InputTensors({(gert::Tensor *)&ioInfo.x1Shape})
                             .InputTensors({(gert::Tensor *)&ioInfo.x2Shape})
                             .InputTensors({(gert::Tensor *)&ioInfo.biasShape})
                             .InputTensors({(gert::Tensor *)&ioInfo.scalesShape})
                             .InputTensors({(gert::Tensor *)&ioInfo.expandedRowIdxShape})
                             .InputTensors({(gert::Tensor *)&ioInfo.expandedExpertIdxShape})

                             .OutputShapes(ouputShapes)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MoeFinalizeRouting")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(yStorageShape.GetOriginShape()), ioInfo.expectOutShape);
}

TEST_F(MoeFinalizeRoutingInfershape, moe_finalize_routing_infershape_0)
{
    gert::StorageShape expandedXShape = {{16, 16}, {16, 16}};
    gert::StorageShape x1Shape = {{16, 16}, {16, 16}};
    gert::StorageShape x2Shape = {{16, 16}, {16, 16}};
    gert::StorageShape biasShape = {{16, 16}, {16, 16}};
    gert::StorageShape scalesShape = {{16, 1}, {16, 1}};
    gert::StorageShape expandedRowIdxShape = {{16}, {16}};
    gert::StorageShape expandedExpertIdxShape = {{16, 1}, {16, 1}};
    std::vector<int64_t> expectOutShape = {16, 16}; // scale第0维，expandedX第一维
    MoeFinalizeRoutingInfo ioInfoT = {expandedXShape,
                                      x1Shape,
                                      x2Shape,
                                      biasShape,
                                      scalesShape,
                                      expandedRowIdxShape,
                                      expandedExpertIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT,
                                      ge::DT_FLOAT,
                                      ge::DT_FLOAT,
                                      ge::DT_FLOAT,
                                      ge::DT_FLOAT,
                                      ge::DT_INT32,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT
                                      };
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}