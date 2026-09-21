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
#include <cstdint>
#include <iostream>

#include "../test_utils/qbsa_tiling_case_executor.h"

namespace QuantBlockSparseAttnUT {

class QbsaTilingDispatchTest : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {};

TEST_P(QbsaTilingDispatchTest, RejectUnsupportedQuantMode)
{
    const auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteQbsaTilingCase(param, tilingInfo));
}

INSTANTIATE_TEST_SUITE_P(
    QbsaTilingDispatch, QbsaTilingDispatchTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

TEST(QbsaTilingDispatchContractTest, UnsupportedModeIsNotRecognized)
{
    for (uint32_t mode : {0U, 3U, 4U, UINT32_MAX}) {
        EXPECT_FALSE(optiling::IsQbsaQuantModeSupported(mode));
    }
}

} // namespace QuantBlockSparseAttnUT
