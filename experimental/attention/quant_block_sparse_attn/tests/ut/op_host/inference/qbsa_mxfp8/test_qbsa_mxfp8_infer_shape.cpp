/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "../../test_utils/qbsa_inference_case_executor.h"

namespace QuantBlockSparseAttnUT {

class QbsaMxfp8InferShapeTest : public testing::TestWithParam<QuantBlockSparseAttnInferShapeUtParam> {};

TEST_P(QbsaMxfp8InferShapeTest, InferShape)
{
    const auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    ASSERT_EQ(param.quant_mode, optiling::QBSA_QUANT_MODE_MXFP8_FULL_QUANT);
    ExecuteQbsaInferShapeCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    QbsaMxfp8InferShape, QbsaMxfp8InferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnInferShapeUtParam>);

} // namespace QuantBlockSparseAttnUT
