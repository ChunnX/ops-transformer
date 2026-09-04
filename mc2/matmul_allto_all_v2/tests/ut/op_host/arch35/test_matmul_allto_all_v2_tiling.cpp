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
#include "../matmul_allto_all_v2_host_ut_param.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MatmulAlltoAllV2UT {

static const std::string OP_NAME = "MatmulAlltoAllV2";

struct MatmulAlltoAllV2CompileInfo {
} compileInfo;

class MatmulAlltoAllV2Arch35TilingTest : public testing::TestWithParam<MatmulAlltoAllV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAlltoAllV2Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAlltoAllV2Arch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(MatmulAlltoAllV2Arch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.context, param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc, param.coreNum);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, "", {});
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAlltoAllV2TilingUT, MatmulAlltoAllV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAlltoAllV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAlltoAllV2TilingUtParam>);

// Cover the rawTilingData capacity check (L591-600 in matmul_allto_all_v2_tiling_base.h)
// by creating TilingContextPara with artificially small/zero tilingDataSize.
TEST(MatmulAlltoAllV2SmallTilingDataTest, CapacityTooSmall)
{
    MatmulAlltoAllV2CompileInfo compileInfo;
    gert::StorageShape ctxShape = GetStorageShape("1 2052");
    gert::StorageShape x1Shape = GetStorageShape("2048 1024");
    gert::StorageShape x2Shape = GetStorageShape("4096 1024");
    gert::StorageShape x1ScaleShape = GetStorageShape("2048 16 2");
    gert::StorageShape x2ScaleShape = GetStorageShape("4096 16 2");
    gert::StorageShape yShape = GetStorageShape("4096 2048");

    gert::TilingContextPara::TensorDescription context(ctxShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1Scale(x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2Scale(x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription y(yShape, ge::DT_BF16, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc(
        {context, x1, x2, TD_DEFAULT, x1Scale, x2Scale});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc({y});

    std::vector<gert::TilingContextPara::OpAttr> attrs(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1073741824)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(27)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("urma")}});

    // Case 1: tilingDataSize=1 → capacity too small
    {
        gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                                  "3510", 64);
        tilingContextPara.tilingDataSize_ = 1;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {});
    }
    // Case 2: tilingDataSize=0 → CreateCap(0) may return null
    {
        gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                                  "3510", 64);
        tilingContextPara.tilingDataSize_ = 0;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {});
    }
}

// Cover TilingParseForMatmulAlltoAllV2 (L48/L50/L51 in matmul_allto_all_v2_tiling.cpp) which
// is registered but never called by the tiling UT.
TEST(MatmulAlltoAllV2TilingParseTest, CallTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    // TilingParseForMatmulAlltoAllV2 ignores the context parameter ((void)context),
    // so passing nullptr is safe and covers the 3-line function body.
    auto ret = opImpl->tiling_parse(nullptr);
    EXPECT_EQ(ge::GRAPH_SUCCESS, static_cast<ge::graphStatus>(ret));
}

} // namespace MatmulAlltoAllV2UT
