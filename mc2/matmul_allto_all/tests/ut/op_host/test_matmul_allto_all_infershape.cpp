/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "mc2_infer_shape_case_executor.h"

namespace MatmulAlltoAllInferShapeUT{

const std::string OP_NAME = "MatmulAlltoAll";

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

TEST_F(MatmulAlltoAllInferShapeTest, BaseCaseTest)
{
    gert::StorageShape x1Shape = {{28543, 3072}, {}};
    gert::StorageShape x2Shape = {{3072, 9216}, {}};
    gert::StorageShape biasShape = {{9216}, {}};
    gert::StorageShape x1ScaleShape = {{28543, 48, 2}, {}};
    gert::StorageShape x2ScaleShape = {{9216, 48, 2}, {}};
    // 保留的参数
    gert::StorageShape commScaleShape = {{}, {}};
    gert::StorageShape x1OffsetShape = {{}, {}};
    gert::StorageShape x2OffsetShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {commScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x1OffsetShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x2OffsetShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -2})},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_UNDEFINED))},
            {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 2}
    };
    std::vector<std::vector<int64_t>> expectOutputShape = {{57086, 4608}};  // {m*2, n/2}
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatmulAlltoAllInferShapeTest, ExceptionCaseTest)
{
    gert::StorageShape x1Shape = {{28543, 3072, 1024}, {}};
    gert::StorageShape x2Shape = {{3072, 9216}, {}};
    gert::StorageShape biasShape = {{9216}, {}};
    gert::StorageShape x1ScaleShape = {{28543, 48, 2}, {}};
    gert::StorageShape x2ScaleShape = {{9216, 48, 2}, {}};
    // 保留的参数
    gert::StorageShape commScaleShape = {{}, {}};
    gert::StorageShape x1OffsetShape = {{}, {}};
    gert::StorageShape x2OffsetShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {commScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x1OffsetShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x2OffsetShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -2})},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_UNDEFINED))},
            {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 2}
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
}

} // namespace MatmulAlltoAllInferShapeUT
