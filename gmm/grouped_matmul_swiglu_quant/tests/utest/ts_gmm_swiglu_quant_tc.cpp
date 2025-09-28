/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file ts_gmm_swiglu_quant_tc.cpp
 * \brief GroupedMatmulSwigluQuant用例.
 */

#include "ts_gmm_swiglu_quant.h"
class Ts_GmmSwigluQuant_Ascend910B2_Case : public Ts_GmmSwigluQuant_WithParam_Ascend910B2 {};

TEST_P(Ts_GmmSwigluQuant_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GmmSwigluQuant_General_Case = ::testing::Values(
    GmmSwigluQuantCase("case_001", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(1, 4, 768, 128, ge::DataType::DT_FLOAT)),
    GmmSwigluQuantCase("case_002", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(1, 4, 768, 128, ge::DataType::DT_FLOAT16)),
    GmmSwigluQuantCase("case_003", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(1, 4, 768, 128, ge::DataType::DT_BF16)),
    GmmSwigluQuantCase("case_004", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(128, 256, 2048, 1536, ge::DataType::DT_FLOAT)),
    GmmSwigluQuantCase("case_005", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(128, 256, 2048, 1536, ge::DataType::DT_FLOAT16)),
    GmmSwigluQuantCase("case_006", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(128, 256, 2048, 1536, ge::DataType::DT_BF16)),
    GmmSwigluQuantCase("case_007", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(128, 256, 4096, 3072, ge::DataType::DT_FLOAT)),
    GmmSwigluQuantCase("case_008", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(128, 256, 4096, 3072, ge::DataType::DT_FLOAT16)),
    GmmSwigluQuantCase("case_009", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            GmmSwigluQuantCase::Param(128, 256, 4096, 3072, ge::DataType::DT_BF16)));

INSTANTIATE_TEST_SUITE_P(GroupedMatmulSwigluQuant, Ts_GmmSwigluQuant_Ascend910B2_Case, Tc_GmmSwigluQuant_General_Case);