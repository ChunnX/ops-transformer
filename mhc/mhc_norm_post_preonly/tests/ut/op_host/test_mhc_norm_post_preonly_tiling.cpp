/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/arch35/mhc_norm_post_preonly_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint32_t T = 4U;
constexpr uint32_t N = 4U;
constexpr uint32_t D = 2560U;

gert::TilingContextPara MakeContext(bool useRmsMid = false, bool returnHInF32 = false)
{
    static optiling::MhcNormPostPreonlyCompileInfo compileInfo = {32U, 0U};
    const gert::StorageShape emptyShape({}, {});
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{N, N * D}, {N, N * D}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {emptyShape, ge::DT_FLOAT, ge::FORMAT_ND}, // pre_mix：当前必须为空
        {{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {useRmsMid ? gert::StorageShape({N * D}, {N * D}) : emptyShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
        {returnHInF32 ? gert::StorageShape({T, D}, {T, D}) : emptyShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {emptyShape, ge::DT_FLOAT, ge::FORMAT_ND}, // h_pre：当前必须为空
    };
    return gert::TilingContextPara(
        "MhcNormPostPreonly", inputs, outputs,
        {{"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
         {"mhc_pre_norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
         {"return_h_in_f32", Ops::Transformer::AnyValue::CreateFrom<bool>(returnHInF32)}},
        &compileInfo, "Ascend950", 32U, 262144U, 4096U);
}
} // namespace

TEST(MhcNormPostPreonlyTiling, SupportsNamedTilingKeyCombinations)
{
    ExecuteTestCase(MakeContext(false, false), ge::GRAPH_SUCCESS, 4);
    ExecuteTestCase(MakeContext(true, false), ge::GRAPH_SUCCESS, 5);
    ExecuteTestCase(MakeContext(false, true), ge::GRAPH_SUCCESS, 6);
    ExecuteTestCase(MakeContext(true, true), ge::GRAPH_SUCCESS, 7);
}

TEST(MhcNormPostPreonlyTiling, RejectsMismatchedResidualShape)
{
    auto context = MakeContext();
    context.inputTensorDesc_[1].shape_ = {{T, N, D + 1U}, {T, N, D + 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(MhcNormPostPreonlyTiling, RejectsMismatchedTokenShapes)
{
    auto context = MakeContext();
    context.inputTensorDesc_[2].shape_ = {{T + 1U, N}, {T + 1U, N}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(MhcNormPostPreonlyTiling, RejectsInvalidParameterShapes)
{
    auto context = MakeContext(true, false);
    context.inputTensorDesc_[5].shape_ = {{2}, {2}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext(true, false);
    context.inputTensorDesc_[10].shape_ = {{N * D - 1U}, {N * D - 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(MhcNormPostPreonlyTiling, RejectsInvalidOutputShapeAndDtype)
{
    auto context = MakeContext();
    context.outputTensorDesc_[1].shape_ = {{T, N, D + 1U}, {T, N, D + 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[4].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
