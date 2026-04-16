/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_matmul_allto_all_infershape.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MatmulAlltoAllInferShapeUt {

class MatmulAlltoAllInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAlltoAllInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAlltoAllInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(MatmulAlltoAllInferShapeTest, BasicFp16Rank2)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, BasicBf16Rank4)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_BF16, ge::FORMAT_ND}, {x2Shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, BasicRank8)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, BasicRank16)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 16}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, TransX2Rank2)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{2048, 1024}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, DynamicShapeMNegativeOne)
{
    gert::StorageShape x1Shape = {{-1, 1024}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, IllegalRankSize)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 5}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, IllegalKExceedLimit)
{
    gert::StorageShape x1Shape = {{8192, 70000}, {}};
    gert::StorageShape x2Shape = {{70000, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, IllegalKMismatch)
{
    gert::StorageShape x1Shape = {{8192, 1024}, {}};
    gert::StorageShape x2Shape = {{2048, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, IllegalX1Transpose)
{
    gert::StorageShape x1Shape = {{1024, 8192}, {}};
    gert::StorageShape x2Shape = {{1024, 2048}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulAlltoAll", {{x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferDataTypeNonQuant)
{
    ge::DataType x1Type = ge::DT_FLOAT16;
    ge::DataType x2Type = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 1)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                 {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
                 {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, x1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, x2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&x1Type, &x2Type})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferDataTypeBf16)
{
    ge::DataType x1Type = ge::DT_BF16;
    ge::DataType x2Type = ge::DT_BF16;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 1)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                        {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_BF16))},
                        {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, x1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, x2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&x1Type, &x2Type})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferDataTypeKcQuant)
{
    ge::DataType x1Type = ge::DT_FLOAT16;
    ge::DataType x2Type = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 1)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                 {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT))},
                 {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                 {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, x1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, x2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&x1Type, &x2Type})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferDataTypeInvalidQuantMode)
{
    ge::DataType x1Type = ge::DT_FLOAT16;
    ge::DataType x2Type = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 1)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                 {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
                 {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, x1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, x2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&x1Type, &x2Type})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferDataTypeYDtypeUndefined)
{
    ge::DataType x1Type = ge::DT_FLOAT16;
    ge::DataType x2Type = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(2, 1)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                 {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
                 {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, x1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, x2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&x1Type, &x2Type})
            .OutputDataTypes({&outputType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

} // namespace MatmulAlltoAllInferShapeUt
