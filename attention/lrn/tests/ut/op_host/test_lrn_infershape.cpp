/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

using namespace std;

// LRN InferShape: y copies the 4-D NCHW shape of x dim-by-dim (and InferDataType
// propagates the x dtype). The rank/dtype gates live in the tiling func; the
// infershape func itself is a pure shape copy.

class LRNInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- LRN InferShape UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- LRN InferShape UT TearDown ---" << std::endl;
    }
};

// fp16 4-D NCHW: y shape = x shape
TEST_F(LRNInferShapeTest, fp16_4d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LRN",
        {
            {{{2, 64, 8, 16}, {2, 64, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 64, 8, 16},
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// fp32 4-D NCHW: y shape = x shape
TEST_F(LRNInferShapeTest, fp32_4d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LRN",
        {
            {{{1, 3, 5, 7}, {1, 3, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 3, 5, 7},
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// larger C (multiple c_chunks at tiling time; infershape is rank-agnostic)
TEST_F(LRNInferShapeTest, fp16_large_c)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LRN",
        {
            {{{4, 384, 16, 16}, {4, 384, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 384, 16, 16},
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
