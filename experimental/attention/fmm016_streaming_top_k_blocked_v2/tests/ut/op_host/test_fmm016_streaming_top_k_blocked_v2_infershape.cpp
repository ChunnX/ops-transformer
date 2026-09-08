/**
 * Copyright (c) 2026 fk2323
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

TEST(Fmm016StreamingTopKBlockedV2InfershapeTest, InfersTopKOutputShape)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs;
    attrs.emplace_back("top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4));
    gert::InfershapeContextPara context("Fmm016StreamingTopKBlockedV2",
                                        {
                                            {{{2, 3, 9, 2, 8}, {2, 3, 9, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{2, 3, 8, 2048}, {2, 3, 8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{2, 3, 9, 2}, {2, 3, 9, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                        },
                                        attrs);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 9, 4}, {2, 3, 9, 4}});
}

TEST(Fmm016StreamingTopKBlockedV2InfershapeTest, InfersDefaultTopKAndDynamicBatch)
{
    gert::InfershapeContextPara context("Fmm016StreamingTopKBlockedV2",
                                        {
                                            {{{-1, 1, 17, 1, 4}, {-1, 1, 17, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{-1, 1, 4, 1024}, {-1, 1, 4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{-1, 1, 17, 1}, {-1, 1, 17, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 1, 17, 1}, {-1, 1, 17, 1}});
}
