/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_init_routing_quant_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
    struct MoeInitRoutingQuantInfo 
    {
        gert::StorageShape& xShape;
        gert::StorageShape& RowIdxShape;
        gert::StorageShape& expertIdxShape;
        std::vector<int64_t> expandedXShape;
        std::vector<int64_t> expandedRowIdxShape;
        std::vector<int64_t> expandedExpertShape;

        ge::DataType XDtype;
        ge::DataType RowIdxDtype;
        ge::DataType expertIdxDtype;
        ge::DataType expandedXDtype;
        ge::DataType expandedRowIdxDtype;
        ge::DataType expandedExpertDtype;
        int64_t activeNum = 0;
    };

    static std::vector<int64_t> ToVector(const gert::Shape& shape) {
        size_t shapeSize = shape.GetDimNum();
        std::vector<int64_t> shapeVec(shapeSize, 0);

        for (size_t i = 0; i < shapeSize; i++) {
            shapeVec[i] = shape.GetDim(i);
        }
        return shapeVec;
    }

    static void ExeTestCase(const MoeInitRoutingQuantInfo ioInfo,
        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
    {
        /* make infershape context */
        gert::StorageShape expandedXStorageShape = {};
        gert::StorageShape expandedRowIdxStorageShape = {};
        gert::StorageShape expandedExpertStorageShape = {};

        std::vector<gert::StorageShape*> ouputShapes = {&expandedXStorageShape, &expandedRowIdxStorageShape, &expandedExpertStorageShape};
        auto contextHolder = gert::InferShapeContextFaker()
            .SetOpType("MoeInitRoutingQuant")
            .NodeIoNum(3, 3)
            .NodeInputTd(0, ioInfo.XDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ioInfo.RowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ioInfo.expertIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ioInfo.expandedXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ioInfo.expandedRowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ioInfo.expandedExpertDtype, ge::FORMAT_ND, ge::FORMAT_ND)

            .InputTensors({(gert::Tensor *)&ioInfo.xShape})
            .InputTensors({(gert::Tensor *)&ioInfo.RowIdxShape})
            .InputTensors({(gert::Tensor *)&ioInfo.expertIdxShape})

            .OutputShapes(ouputShapes)
            .Attr("active_num", int64_t(ioInfo.activeNum))
            .Build();

        /* get infershape func */
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        auto inferShapeFunc = spaceRegistry->GetOpImpl("MoeInitRoutingQuant")->infer_shape;
        ASSERT_NE(inferShapeFunc, nullptr);

        /* do infershape */
        EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
        EXPECT_EQ(ToVector(expandedXStorageShape.GetOriginShape()), ioInfo.expandedXShape);
        EXPECT_EQ(ToVector(expandedRowIdxStorageShape.GetOriginShape()), ioInfo.expandedRowIdxShape);
        EXPECT_EQ(ToVector(expandedExpertStorageShape.GetOriginShape()), ioInfo.expandedExpertShape);
    }
}

class MoeInitRoutingQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingQuant SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingQuant TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_01)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShap = {8, 3};
    std::vector<int64_t> expandedRowIdxShape = {8};
    std::vector<int64_t> expandedExpertShape = {8};
    MoeInitRoutingQuantInfo ioInfoT = {xShape, RowIdxShape, expertIdxShape, expandedXShap, expandedRowIdxShape, expandedExpertShape,
    ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT8, ge::DT_INT32, ge::DT_INT32, 8};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}