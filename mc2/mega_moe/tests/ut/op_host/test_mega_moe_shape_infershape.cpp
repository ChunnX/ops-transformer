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
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

namespace MegaMoeUT {

class InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MegaMoe InferShapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MegaMoe InferShapeTest TearDown" << std::endl;
    }
};

TEST_F(InferShapeTest, mega_moe_infer_shape_basic) {
    gert::StorageShape xShape = {{128, 512}, {128, 512}};
    gert::StorageShape yShape = {{128, 512}, {128, 512}};
    
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_BF16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_BF16, ge::FORMAT_ND);
    
    gert::InfershapeContextPara inferShapeContextPara(
        "MegaMoe",
        {x},
        {y},
        {},
        {1},
        {1}
    );
    
    ExecuteTestCase(inferShapeContextPara, ge::GRAPH_SUCCESS, {{128, 512}});
}

} // namespace MegaMoeUT