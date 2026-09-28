/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_turbo_quant_attention_score_infershape.cpp
 * \brief InferShape unit test for TurboQuantAttentionScore.
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class TurboQuantAttentionScoreInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TurboQuantAttentionScoreInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TurboQuantAttentionScoreInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(TurboQuantAttentionScoreInfershapeTest, DecodeBasicShape)
{
    constexpr int64_t numTokens = 4;
    constexpr int64_t numQHeads = 32;
    constexpr int64_t numKvHeads = 8;
    constexpr int64_t headDim = 128;
    constexpr int64_t pagedBlockSize = 128;
    constexpr int64_t numBlocks = 16;
    constexpr int64_t maxBlocksPerSeq = 8;
    constexpr int64_t mseBits = 3;
    constexpr int64_t idxPackedBytes = (headDim * mseBits + 7) / 8;
    constexpr int64_t qjlPackedBytes = (headDim + 7) / 8;
    constexpr int64_t maxKvLen = maxBlocksPerSeq * pagedBlockSize;

    gert::StorageShape queryShape = {{numTokens, numQHeads, headDim}, {numTokens, numQHeads, headDim}};
    gert::StorageShape queryQjlShape = {{numTokens, numQHeads, headDim}, {numTokens, numQHeads, headDim}};
    gert::StorageShape idxShape = {{numBlocks, pagedBlockSize, numKvHeads, idxPackedBytes},
                                   {numBlocks, pagedBlockSize, numKvHeads, idxPackedBytes}};
    gert::StorageShape qjlShape = {{numBlocks, pagedBlockSize, numKvHeads, qjlPackedBytes},
                                   {numBlocks, pagedBlockSize, numKvHeads, qjlPackedBytes}};
    gert::StorageShape normShape = {{numBlocks, pagedBlockSize, numKvHeads}, {numBlocks, pagedBlockSize, numKvHeads}};
    gert::StorageShape gammaShape = {{numBlocks, pagedBlockSize, numKvHeads}, {numBlocks, pagedBlockSize, numKvHeads}};
    gert::StorageShape blockTableShape = {{numTokens, maxBlocksPerSeq}, {numTokens, maxBlocksPerSeq}};
    gert::StorageShape seqLensShape = {{numTokens}, {numTokens}};

    gert::InfershapeContextPara infershapeContextPara(
        "TurboQuantAttentionScore",
        {
            {queryShape, ge::DT_BF16, ge::FORMAT_ND},
            {queryQjlShape, ge::DT_BF16, ge::FORMAT_ND},
            {idxShape, ge::DT_UINT8, ge::FORMAT_ND},
            {qjlShape, ge::DT_UINT8, ge::FORMAT_ND},
            {normShape, ge::DT_BF16, ge::FORMAT_ND},
            {gammaShape, ge::DT_BF16, ge::FORMAT_ND},
            {blockTableShape, ge::DT_INT32, ge::FORMAT_ND},
            {seqLensShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mseBits)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{numTokens, numQHeads, maxKvLen}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
