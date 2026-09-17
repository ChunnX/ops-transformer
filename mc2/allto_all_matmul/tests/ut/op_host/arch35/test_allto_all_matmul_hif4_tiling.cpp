/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.openhisilicon.com
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_allto_all_matmul_hif4_tiling.cpp
 * \brief AlltoAllMatmul HiF4 全量化端到端 tiling UT(CSV 参数化)
 *
 * 用例来源: mc2/allto_all_matmul/docs/hif4/HiF4_UT_Case_Design.md §2
 * 用例数据: test_allto_all_matmul_hif4_tiling.csv(socVersion=9201, 31 条)
 *
 * HiF4 仅在 A6(ascend960dt/DAV_9201) 支持, 且依赖 A6 devkit 提供的
 * ge::DT_HIFLOAT4_SCALE(C_DT_HIFLOAT4_SCALE=43) 枚举。当前 A5 包
 * ge 枚举止于 DT_HIFLOAT4(42)/DT_MAX(43), op_host 侧 HiF4 代码无法编译,
 * 因此本用例在参数层做 dtype 桥接, 并以 DT_MAX>43 作为运行门控:
 *   - A5 包: 整体 GTEST_SKIP(不阻塞存量用例)
 *   - A6 devkit(框架 GE_DTYPE 补齐 HIFLOAT4/HIFLOAT4_SCALE 前亦可): 桥接值生效
 * 已知问题 R1(ND 布局 tiling 误拒)以 expectStatus=FAILED 钉住, 修复后翻转。
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include "../allto_all_matmul_host_ut_param.h"
#include "tiling_case_executor.h"

namespace AlltoAllMatmulHif4UT {

static const std::string OP_NAME = "AlltoAllMatmul";

struct AlltoAllMatmulCompileInfo {
} compileInfo;

// A6 devkit 落位 C_DT_HIFLOAT4_SCALE=43 后 DT_MAX>=44; A5 包 DT_MAX=43
static bool Hif4ScaleDtypeReady()
{
    return static_cast<int32_t>(ge::DT_MAX) > 43;
}

// HiF4 dtype 桥接: 框架 GE_DTYPE 映射表补齐 HIFLOAT4/HIFLOAT4_SCALE 前的本地解析
// (42=DT_HIFLOAT4 为 A5 已有枚举; 43=C_DT_HIFLOAT4_SCALE 计划值, devkit 落位后核对)
static void FixupHif4Dtype(const std::string &dtypeStr, ge::DataType &dtype)
{
    if (dtypeStr == "HIFLOAT4") {
        dtype = ge::DT_HIFLOAT4;
    } else if (dtypeStr == "HIFLOAT4_SCALE") {
        dtype = static_cast<ge::DataType>(43);
    }
}

struct AlltoAllMatmulHif4TilingUtParam : public AlltoAllMatmulUT::AlltoAllMatmulTilingUtParam {
    explicit AlltoAllMatmulHif4TilingUtParam(const csv_map &csvMap)
        : AlltoAllMatmulUT::AlltoAllMatmulTilingUtParam(csvMap)
    {
        // 框架解析结果为 DT_UNDEFINED 时按本地映射修复 HiF4 dtype
        FixupHif4Dtype(ReadMap(csvMap, "x1Dtype"), this->x1.dtype_);
        FixupHif4Dtype(ReadMap(csvMap, "x2Dtype"), this->x2.dtype_);
        FixupHif4Dtype(ReadMap(csvMap, "x1ScaleDtype"), this->x1Scale.dtype_);
        FixupHif4Dtype(ReadMap(csvMap, "x2ScaleDtype"), this->x2Scale.dtype_);
        FixupHif4Dtype(ReadMap(csvMap, "outputDtype"), this->all2allOut.dtype_);
    }
};

class AlltoAllMatmulHif4Arch35TilingTest : public testing::TestWithParam<AlltoAllMatmulHif4TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulHif4Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulHif4Arch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AlltoAllMatmulHif4Arch35TilingTest, param)
{
    if (!Hif4ScaleDtypeReady()) {
        GTEST_SKIP() << "HiF4 requires A6 devkit (ge::DT_HIFLOAT4_SCALE), skip on current package";
    }
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale, param.commScale, param.x1Offset,
         param.x2Offset});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.all2allAxes))},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtypeAttr))},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantMode)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtype)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
         {"alltoallout_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.alltoalloutFlag)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

INSTANTIATE_TEST_SUITE_P(AlltoAllMatmulHif4TilingUT, AlltoAllMatmulHif4Arch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulHif4TilingUtParam>(
                             ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<AlltoAllMatmulHif4TilingUtParam>);

} // namespace AlltoAllMatmulHif4UT
