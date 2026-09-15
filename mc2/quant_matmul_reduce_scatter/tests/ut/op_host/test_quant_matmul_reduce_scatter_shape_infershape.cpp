/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_matmul_reduce_scatter_shape_infershape.cpp
 * \brief infershape ut
 */

#include <gtest/gtest.h>
#include "test_quant_matmul_reduce_scatter_host_ut_param.h"
#include "mc2_infer_shape_case_executor.h"

namespace QuantMatmulReduceScatterUT {

class QuantMatmulReduceScatterInferShapeTest
    : public testing::TestWithParam<QuantMatmulReduceScatterInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantMatmulReduceScatterInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantMatmulReduceScatterInferShapeTest TearDown" << std::endl;
    }
};

TEST_P(QuantMatmulReduceScatterInferShapeTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    // def 输入顺序: context(0) x1(1) x2(2) x1_scale(3) x2_scale(4)，仅放入已实例化的输入
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    inputTensorDesc.emplace_back(param.context);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.x2);
    if (param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.x1Scale);
    if (param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.x2Scale);
    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc({param.y});

    gert::InfershapeContextPara infershapeContextPara(
        "QuantMatmulReduceScatter", inputTensorDesc, outputTensorDesc,
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransA)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransB)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankSize)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)}},
        param.inputInstance, param.outputInstance);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    QuantMatmulReduceScatterInferShapeUT, QuantMatmulReduceScatterInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantMatmulReduceScatterInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantMatmulReduceScatterInferShapeUtParam>);

} // namespace QuantMatmulReduceScatterUT
