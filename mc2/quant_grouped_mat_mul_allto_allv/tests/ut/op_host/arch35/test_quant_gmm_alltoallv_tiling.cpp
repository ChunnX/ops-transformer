/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include "../quant_gmm_alltoallv_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#include "../../../../op_host/op_tiling/arch35/tt_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../../op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.h"

using namespace QuantGmmAlltoAllvUT;

static std::string GetCsvPath(const char* file)
{
    const char* envPath = std::getenv("CSV_CASE_PATH");
    if (envPath != nullptr && strlen(envPath) > 0) {
        return std::string(envPath);
    }
    return ReplaceFileExtension2Csv(file);
}

static const std::string OP_NAME = "QuantGroupedMatMulAlltoAllv";

struct QuantGmmAlltoAllvCompileInfo {} compileInfo;

class QuantGmmAlltoAllvArch35TilingTest
    : public testing::TestWithParam<QuantGmmAlltoAllvTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantGmmAlltoAllvArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantGmmAlltoAllvArch35TilingTest TearDown." << std::endl;
    }
};

static gert::StorageShape BuildStorageShape(const std::vector<int64_t>& shape)
{
    if (shape.empty() || shape[0] <= 0) {
        return {};
    }
    switch (shape.size()) {
        case 1:
            return {{shape[0]}, {shape[0]}};
        case 2:
            return {{shape[0], shape[1]}, {shape[0], shape[1]}};
        case 3:
            return {{shape[0], shape[1], shape[2]}, {shape[0], shape[1], shape[2]}};
        case 4:
            return {{shape[0], shape[1], shape[2], shape[3]},
                    {shape[0], shape[1], shape[2], shape[3]}};
        default:
            return {};
    }
}

static gert::StorageShape BuildScaleStorageShape(const std::vector<int64_t>& shape, size_t mxDim)
{
    if (shape.empty() || shape[0] <= 0) {
        return {};
    }
    if (shape.size() == mxDim) {
        return BuildStorageShape(shape);
    }
    return {{shape[0]}, {shape[0]}};
}

TEST_P(QuantGmmAlltoAllvArch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    const uint64_t gmmXScaleDimMx = 3;
    const uint64_t gmmWeightScaleDimMx = 4;
    const uint64_t mmXScaleDimMx = 3;
    const uint64_t mmWeightScaleDimMx = 3;
    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    size_t tilingDataSize = sizeof(QuantGmmA2avTilingData);

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_ = {
        {BuildStorageShape(param.gmmXShape), param.gmmXDtype, param.gmmXFormat},
        {BuildStorageShape(param.gmmWeightShape), param.gmmWeightDtype, param.gmmWeightFormat},
        {{}, ge::DT_INT64, ge::FORMAT_ND},
        {{}, ge::DT_INT64, ge::FORMAT_ND},
        {BuildStorageShape(param.mmXShape), param.mmXDtype, param.mmXFormat},
        {BuildStorageShape(param.mmWeightShape), param.mmWeightDtype, param.mmWeightFormat},
        {BuildScaleStorageShape(param.gmmXScaleShape, gmmXScaleDimMx), param.gmmXScaleDtype,
         param.gmmXScaleFormat},
        {BuildScaleStorageShape(param.gmmWeightScaleShape, gmmWeightScaleDimMx),
         param.gmmWeightScaleDtype, param.gmmWeightScaleFormat},
        {BuildScaleStorageShape(param.mmXScaleShape, mmXScaleDimMx), param.mmXScaleDtype,
         param.mmXScaleFormat},
        {BuildScaleStorageShape(param.mmWeightScaleShape, mmWeightScaleDimMx),
         param.mmWeightScaleDtype, param.mmWeightScaleFormat},
        {{}, ge::DT_INT64, ge::FORMAT_ND},
    };

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_ = {
        {BuildStorageShape(param.gmmYShape), param.gmmYDtype, param.gmmYFormat},
        {BuildStorageShape(param.mmYShape), param.mmYDtype, param.mmYFormat},
    };

    std::vector<gert::TilingContextPara::OpAttr> attrs_ = {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
        {"send_counts",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.sendCounts)},
        {"recv_counts",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.recvCounts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transGmmWeight)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transMmWeight)},
        {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmXQuantMode)},
        {"gmm_weight_quant_mode",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmWeightQuantMode)},
        {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmXQuantMode)},
        {"mm_weight_quant_mode",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmWeightQuantMode)},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
        {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"gmm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"mm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
    };

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_,
                                              &compileInfo, "Ascend950", coreNum, ubSize,
                                              tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectStatus,
                       param.expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(
    QuantGmmAlltoAllvTilingUT,
    QuantGmmAlltoAllvArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<QuantGmmAlltoAllvTilingUtParam>(GetCsvPath(__FILE__))),
    PrintCaseInfoString<QuantGmmAlltoAllvTilingUtParam>
);
