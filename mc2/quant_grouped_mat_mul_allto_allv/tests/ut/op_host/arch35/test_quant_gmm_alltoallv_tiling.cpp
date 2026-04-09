/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_gmm_alltoallv_tiling.cpp
 * \brief tiling ut
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include "../quant_gmm_alltoallv_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#include "../../../../op_host/op_tiling/arch35/tt_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../../op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.h"

static std::string GetCsvPath(const char* file)
{
    const char* envPath = std::getenv("CSV_CASE_PATH");
    if (envPath != nullptr && strlen(envPath) > 0) {
        return std::string(envPath);
    }
    return ReplaceFileExtension2Csv(file);
}

namespace QuantGmmAlltoallvUT {

static const std::string OP_NAME = "QuantGroupedMatMulAlltoAllv";

struct QuantGmmAlltoallvCompileInfo {} compileInfo;

class QuantGmmAlltoallvArch35TilingTest : public testing::TestWithParam<QuantGmmAlltoallvTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantGmmAlltoallvArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantGmmAlltoallvArch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(QuantGmmAlltoallvArch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.gmmX, param.gmmWeight,
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         param.mmX, param.mmWeight,
         param.gmmXScale, param.gmmWeightScale,
         param.mmXScale, param.mmWeightScale,
         {{}, ge::DT_INT64, ge::FORMAT_ND}});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {param.gmmY, param.mmY});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.sendCounts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.recvCounts)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transGmmWeight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transMmWeight)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmXQuantMode)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmWeightQuantMode)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmXQuantMode)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmWeightQuantMode)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"gmm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"mm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}});

    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    size_t tilingDataSize = sizeof(QuantGmmA2avTilingData);

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_,
                                              &compileInfo, "Ascend950", coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(
    QuantGmmAlltoallvTilingUT,
    QuantGmmAlltoallvArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<QuantGmmAlltoallvTilingUtParam>(GetCsvPath(__FILE__))),
    PrintCaseInfoString<QuantGmmAlltoallvTilingUtParam>
);

} // namespace QuantGmmAlltoallvUT
