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
 * \file test_quant_matmul_reduce_scatter_dtype_infershape.cpp
 * \brief inferdtype ut
 */

#include <gtest/gtest.h>
#include "test_quant_matmul_reduce_scatter_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace QuantMatmulReduceScatterUT {

class QuantMatmulReduceScatterInferDataTypeTest
    : public testing::TestWithParam<QuantMatmulReduceScatterInferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantMatmulReduceScatterInferDataTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantMatmulReduceScatterInferDataTypeTest TearDown" << std::endl;
    }
};

TEST_P(QuantMatmulReduceScatterInferDataTypeTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    // def 输入顺序: context(0) x1(1) x2(2) x1_scale(3) x2_scale(4)，仅放入已实例化的输入
    ge::DataType contextDtype = ge::DT_INT32;
    std::vector<void *> inputDataTypes;
    inputDataTypes.emplace_back(&contextDtype);
    inputDataTypes.emplace_back(&param.x1);
    inputDataTypes.emplace_back(&param.x2);
    inputDataTypes.emplace_back(&param.x1Scale);
    inputDataTypes.emplace_back(&param.x2Scale);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("QuantMatmulReduceScatter")
            .IrInstanceNum(param.inputInstance, param.outputInstance)
            .InputDataTypes(inputDataTypes)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
                        {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
                        {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransA)},
                        {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransB)},
                        {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankSize)},
                        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
                        {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("QuantMatmulReduceScatter")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.expectYDtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantMatmulReduceScatterInferDataTypeUT, QuantMatmulReduceScatterInferDataTypeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantMatmulReduceScatterInferDataTypeUtParam>(
        ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantMatmulReduceScatterInferDataTypeUtParam>);

} // namespace QuantMatmulReduceScatterUT
