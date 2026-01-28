/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {

// inferShape用例 ======================================================================================================
class AllGatherMatmulV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllGatherMatmulV2InferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AllGatherMatmulV2InferShapeTest TearDown" << std::endl; }
};

TEST_F(AllGatherMatmulV2InferShapeTest, basic)
{
    gert::StorageShape x1_shape = {{8192, 12288}, {}};
    gert::StorageShape x2_shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV2InferShapeTest, empty_tensor_fail_test)
{
    gert::StorageShape x1_shape = {{8192, 0}, {}};
    gert::StorageShape x2_shape = {{0, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV2InferShapeTest, pertensor)
{
    gert::StorageShape x1_shape = {{8192, 12288}, {}};
    gert::StorageShape x2_shape = {{12288, 3904}, {}};
    gert::StorageShape x1_scale_shape = {{1}, {}};
    gert::StorageShape x2_scale_shape = {{1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1_shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x1_scale_shape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x2_scale_shape}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulV2InferShapeTest, perblock)
{
    gert::StorageShape x1_shape = {{8192, 12288}, {}};
    gert::StorageShape x2_shape = {{12288, 3904}, {}};
    gert::StorageShape x1_scale_shape = {{64, 96}, {}};
    gert::StorageShape x2_scale_shape = {{96, 30}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AllGatherMatmulV2",
        {
            {x1_shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x1_scale_shape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{x2_scale_shape}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// inferDtype用例 ======================================================================================================
class AllGatherMatmulV2InferDTypeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllGatherMatmulV2InferDTypeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AllGatherMatmulV2InferDTypeTest TearDown" << std::endl; }
};

TEST_F(AllGatherMatmulV2InferDTypeTest, y_dtype_equal_x1_dtype_fp16)
{
    ge::DataType x1_type = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(1, 2)
        .InputDataTypes({&x1_type})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1_type);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1_type);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, y_dtype_equal_x1_dtype_bf16)
{
    ge::DataType x1_type = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(1, 2)
        .InputDataTypes({&x1_type})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1_type);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1_type);
}

TEST_F(AllGatherMatmulV2InferDTypeTest, attr_y_dtype)
{
    ge::DataType x1_type = ge::DT_FLOAT8_E5M2;
    ge::DataType attr_y_dtype = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(1, 2)
        .InputDataTypes({&x1_type})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(attr_y_dtype))}
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), attr_y_dtype);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), x1_type);
}

} // namespace