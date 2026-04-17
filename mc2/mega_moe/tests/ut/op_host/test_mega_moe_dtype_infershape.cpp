/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MegaMoeUT {

class InferDataTypeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MegaMoe InferDataTypeTest SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MegaMoe InferDataTypeTest TearDown" << std::endl;
    }
};

TEST_F(InferDataTypeTest, mega_moe_infer_dtype_bf16) {
    ge::DataType x1 = ge::DT_BF16;
    ge::DataType y = ge::DT_BF16;
    std::vector<void*> inputDataTypes = {&x1};
    std::vector<uint32_t> inputInstance = {1};
    std::vector<uint32_t> outputInstance = {1};

    auto contextHolder = gert::InferDataTypeContextFaker()
        .SetOpType("MegaMoe")
        .IrInstanceNum(inputInstance, outputInstance)
        .InputDataTypes(inputDataTypes)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    EXPECT_EQ(inferDtypeFunc(contextHolder.template GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.template GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), y);
}

TEST_F(InferDataTypeTest, mega_moe_infer_dtype_fp16) {
    ge::DataType x1 = ge::DT_FLOAT16;
    ge::DataType y = ge::DT_FLOAT16;
    std::vector<void*> inputDataTypes = {&x1};
    std::vector<uint32_t> inputInstance = {1};
    std::vector<uint32_t> outputInstance = {1};

    auto contextHolder = gert::InferDataTypeContextFaker()
        .SetOpType("MegaMoe")
        .IrInstanceNum(inputInstance, outputInstance)
        .InputDataTypes(inputDataTypes)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    EXPECT_EQ(inferDtypeFunc(contextHolder.template GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.template GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), y);
}

} // namespace MegaMoeUT