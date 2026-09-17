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

#include "../../test_utils/qbsa_tiling_case_executor.h"

namespace QuantBlockSparseAttnUT {

class QbsaFp8Arch35TilingTest : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {};

TEST_P(QbsaFp8Arch35TilingTest, Tiling)
{
    const auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    ASSERT_EQ(param.quant_mode, optiling::QBSA_QUANT_MODE_FP8);

    TilingInfo tilingInfo;
    const bool ok = ExecuteQbsaFp8TilingCase(param, tilingInfo);
    if (param.expectResult != ge::GRAPH_SUCCESS) {
        EXPECT_FALSE(ok);
        return;
    }

    ASSERT_TRUE(ok);
    EXPECT_GT(tilingInfo.blockNum, 0U);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(optiling::BuildQbsaFp8TilingKey(
                                        param.layout_q == "NTD" ? QBSA_LAYOUT_NTD : QBSA_LAYOUT_TND, param.mask_mode,
                                        param.return_softmax_lse)));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::QuantBlockSparseAttnTilingData));
    const auto data = DecodeQbsaTilingData(tilingInfo);
    ExpectQbsaCommonTilingData(param, tilingInfo, data);
    EXPECT_EQ(data.fp8Params.needInit, 1U);
}

INSTANTIATE_TEST_SUITE_P(
    QbsaFp8Arch35Tiling, QbsaFp8Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

} // namespace QuantBlockSparseAttnUT
