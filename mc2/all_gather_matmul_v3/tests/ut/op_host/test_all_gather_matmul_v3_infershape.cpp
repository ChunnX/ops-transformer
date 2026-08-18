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
 * \file test_all_gather_matmul_v3_infershape.cpp
 * \brief infershape ut
 */

#include <gtest/gtest.h>
#include <iostream>
#include "mc2_infer_shape_case_executor.h"

class AllGatherMatmulV3InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV3InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV3InferShapeTest TearDown" << std::endl;
    }
};

static std::vector<gert::InfershapeContextPara::OpAttr> BuildAttrs(int64_t rankSize, bool isTransB = true)
{
    return {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(isTransB)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rankSize)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(
                                static_cast<int64_t>((1ULL << 32) | (1ULL << 16) | 32))},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_BF16))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("urma")}};
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_E4M3_Rank4)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_E4M3_Rank2)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(2));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 2560}, {4096, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_E4M3_Rank8)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(8));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{16384, 2560}, {16384, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_E4M3_Rank16)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(16));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 16}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{32768, 2560}, {32768, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_E5M2_Rank4)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_FP4_Rank4)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, Basic2D_MixedE4M3E5M2_Rank4)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, DynamicM)
{
    gert::StorageShape x1Shape = {{-1, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{-1, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, DynamicK)
{
    gert::StorageShape x1Shape = {{2048, -1}, {}};
    gert::StorageShape x2Shape = {{2560, -1}, {}};
    gert::StorageShape x1ScaleShape = {{2048, -1, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, -1, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {-1, -1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, DynamicN)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{-1, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{-1, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, InvalidX1Dtype)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, InvalidX2Dtype)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_INT8, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, FP4Mismatch)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, KMismatch)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4032}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 63, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, KNotDivisible64)
{
    gert::StorageShape x1Shape = {{2048, 4033}, {}};
    gert::StorageShape x2Shape = {{2560, 4033}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 63, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 63, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, X1OneDim)
{
    gert::StorageShape x1Shape = {{4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, X2OneDim)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV3InferShapeTest, X1ScaleWrongShape)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{1024, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, X2ScaleWrongShape)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{1280, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, X1ScaleWrongDimCount)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{2560, 4096}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV3InferShapeTest, TransBFalse)
{
    gert::StorageShape x1Shape = {{2048, 4096}, {}};
    gert::StorageShape x2Shape = {{4096, 2560}, {}};
    gert::StorageShape x1ScaleShape = {{2048, 64, 2}, {}};
    gert::StorageShape x2ScaleShape = {{2560, 64, 2}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV3",
        {{x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}},
        BuildAttrs(4, false));
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 2560}, {8192, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}
