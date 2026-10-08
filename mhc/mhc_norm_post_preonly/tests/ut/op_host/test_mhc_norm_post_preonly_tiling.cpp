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

gert::TilingContextPara MakeContext(bool useRmsMid = false, bool returnHInF32 = false, uint32_t tokens = T,
                                    uint32_t embedDim = D)
{
    static optiling::MhcNormPostPreonlyCompileInfo compileInfo = {32U, 0U};
    const gert::StorageShape emptyShape({}, {});
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {{{tokens, embedDim}, {tokens, embedDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{tokens, N, embedDim}, {tokens, N, embedDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{tokens, N}, {tokens, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{tokens, N, N}, {tokens, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{N, N * embedDim}, {N, N * embedDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {emptyShape, ge::DT_FLOAT, ge::FORMAT_ND}, // pre_mix：当前必须为空
        {{{embedDim}, {embedDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{embedDim}, {embedDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {useRmsMid ? gert::StorageShape({N * embedDim}, {N * embedDim}) : emptyShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {{{tokens, embedDim}, {tokens, embedDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{tokens, N, embedDim}, {tokens, N, embedDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {returnHInF32 ? gert::StorageShape({tokens, embedDim}, {tokens, embedDim}) : emptyShape, ge::DT_FLOAT,
         ge::FORMAT_ND},
        {emptyShape, ge::DT_FLOAT, ge::FORMAT_ND}, // h_pre：当前必须为空
    };
    return gert::TilingContextPara("MhcNormPostPreonly", inputs, outputs,
                                   {{"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                    {"mhc_pre_norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                    {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                    {"return_h_in_f32", Ops::Transformer::AnyValue::CreateFrom<bool>(returnHInF32)}},
                                   &compileInfo, "Ascend950", 32U, 262144U, 4096U);
}
} // namespace

TEST(MhcNormPostPreonlyTiling, SupportsNamedTilingKeyCombinations)
{
    constexpr uint32_t singleAivTokens = 17U;
    ExecuteTestCase(MakeContext(false, false, singleAivTokens), ge::GRAPH_SUCCESS, 0);
    ExecuteTestCase(MakeContext(true, false, singleAivTokens), ge::GRAPH_SUCCESS, 1);
    ExecuteTestCase(MakeContext(false, true, singleAivTokens), ge::GRAPH_SUCCESS, 2);
    ExecuteTestCase(MakeContext(true, true, singleAivTokens), ge::GRAPH_SUCCESS, 3);
    ExecuteTestCase(MakeContext(false, false), ge::GRAPH_SUCCESS, 4);
    ExecuteTestCase(MakeContext(true, false), ge::GRAPH_SUCCESS, 5);
    ExecuteTestCase(MakeContext(false, true), ge::GRAPH_SUCCESS, 6);
    ExecuteTestCase(MakeContext(true, true), ge::GRAPH_SUCCESS, 7);
}

TEST(MhcNormPostPreonlyTiling, SupportsAllDeclaredEmbeddingDimensions)
{
    for (uint32_t embedDim : {2560U, 5120U, 6144U, 7168U}) {
        ExecuteTestCase(MakeContext(false, false, T, embedDim), ge::GRAPH_SUCCESS, 4);
    }
}

TEST(MhcNormPostPreonlyTiling, SupportsBsndShapes)
{
    auto context = MakeContext();
    context.inputTensorDesc_[0].shape_ = {{1U, T, D}, {1U, T, D}};
    context.inputTensorDesc_[1].shape_ = {{1U, T, N, D}, {1U, T, N, D}};
    context.inputTensorDesc_[2].shape_ = {{1U, T, N}, {1U, T, N}};
    context.inputTensorDesc_[3].shape_ = {{1U, T, N, N}, {1U, T, N, N}};
    context.outputTensorDesc_[0].shape_ = {{1U, T, D}, {1U, T, D}};
    context.outputTensorDesc_[1].shape_ = {{1U, T, N, D}, {1U, T, N, D}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 4);
}

TEST(MhcNormPostPreonlyTiling, RejectsUnsupportedEmbeddingDimension)
{
    ExecuteTestCase(MakeContext(false, false, T, 4096U), ge::GRAPH_FAILED);
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

    context = MakeContext();
    context.inputTensorDesc_[3].shape_ = {{T + 1U, N, N}, {T + 1U, N, N}};
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

    context = MakeContext();
    context.inputTensorDesc_[4].shape_ = {{N, N * D - 1U}, {N, N * D - 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[6].shape_ = {{N + 1U}, {N + 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[8].shape_ = {{D - 1U}, {D - 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[9].shape_ = {{D - 1U}, {D - 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(MhcNormPostPreonlyTiling, RejectsUnsupportedReservedInterfaceShapes)
{
    const gert::StorageShape emptyShape({}, {});
    auto context = MakeContext();
    context.inputTensorDesc_[7].shape_ = {{T, N}, {T, N}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[3].shape_ = emptyShape;
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[8].shape_ = emptyShape;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(MhcNormPostPreonlyTiling, RejectsInvalidOutputShapeAndDtype)
{
    auto context = MakeContext();
    context.outputTensorDesc_[1].shape_ = {{T, N, D + 1U}, {T, N, D + 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.outputTensorDesc_[0].shape_ = {{T, D + 1U}, {T, D + 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext(false, true);
    context.outputTensorDesc_[2].shape_ = {{T, D + 1U}, {T, D + 1U}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[4].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.inputTensorDesc_[1].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(context, ge::GRAPH_FAILED);

    context = MakeContext();
    context.outputTensorDesc_[0].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
