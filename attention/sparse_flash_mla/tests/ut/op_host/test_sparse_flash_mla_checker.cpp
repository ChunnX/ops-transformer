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
#include "../../../op_host/checkers/mask_checker.h"
#include "../../../op_host/checkers/softmax_lse_checker.h"
#include "../../../op_host/checkers/sparse_compression_checker.h"

namespace {
using optiling::sparse_mla_checker::CheckContext;
using optiling::sparse_mla_checker::MaskChecker;
using optiling::sparse_mla_checker::OperatorVariant;
using optiling::sparse_mla_checker::SoftmaxLseChecker;
using optiling::sparse_mla_checker::SparseCompressionChecker;

CheckContext MakeSparseContext()
{
    CheckContext context;
    context.opName = "SparseFlashMla";
    context.variant = OperatorVariant::SPARSE;
    context.cmpRatio = 1;
    context.oriMaskMode = 0;
    context.cmpMaskMode = 0;
    context.oriWinLeft = -1;
    context.oriWinRight = -1;
    return context;
}
} // namespace

TEST(SparseFlashMlaChecker, AcceptsOriSparseIndicesWithMatchingTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.oriSparseIndices.present = true;
    context.oriTopkLength.present = true;

    SparseCompressionChecker compressionChecker;
    MaskChecker maskChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(maskChecker.CheckFeature(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, RejectsOriSparseIndicesWithoutTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.oriSparseIndices.present = true;

    SparseCompressionChecker compressionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
}

TEST(SparseFlashMlaChecker, AcceptsAllSparseTopkLengthPairs)
{
    CheckContext context = MakeSparseContext();
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;

    SparseCompressionChecker compressionChecker;
    MaskChecker maskChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(maskChecker.CheckFeature(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, SkipsSoftmaxLseChecksWhenReturnIsDisabled)
{
    CheckContext context = MakeSparseContext();
    context.returnSoftmaxLse = false;
    context.softmaxLse.present = true;

    SoftmaxLseChecker checker;
    EXPECT_EQ(checker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiPara(context), ge::GRAPH_SUCCESS);
}
