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
#include "../allto_all_matmul_v2_host_ut_param.h"
#include "tiling_case_executor.h"

namespace AlltoAllMatmulV2UT {

static const std::string OP_NAME = "AlltoAllMatmulV2";

struct AlltoAllMatmulV2CompileInfo {
} compileInfo;

class AlltoAllMatmulV2Arch35TilingTest : public testing::TestWithParam<AlltoAllMatmulV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulV2Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulV2Arch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AlltoAllMatmulV2Arch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.context, param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtype)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.precisionMode)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc, param.coreNum);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, "", {});
}

INSTANTIATE_TEST_SUITE_P(AlltoAllMatmulV2TilingUT, AlltoAllMatmulV2Arch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<AlltoAllMatmulV2TilingUtParam>);

} // namespace AlltoAllMatmulV2UT
