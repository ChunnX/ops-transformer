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
#include <iostream>
#include <string>
#include <vector>

#include "quant_block_sparse_attn_host_ut_param.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

// Compile the device address helpers in the existing host UT translation unit.
#ifndef __aicore__
#define __aicore__
#define QBSA_UT_DEFINED_AICORE
#endif
#include "../../../op_kernel/arch35/common/util_regbase_mx.h"
#ifdef QBSA_UT_DEFINED_AICORE
#undef __aicore__
#undef QBSA_UT_DEFINED_AICORE
#endif

namespace QuantBlockSparseAttnUT {

static const std::string OP_NAME = "QuantBlockSparseAttn";

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

class QuantBlockSparseAttnInferShapeTest : public testing::TestWithParam<QuantBlockSparseAttnInferShapeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "QuantBlockSparseAttnInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantBlockSparseAttnInferShapeTest TearDown" << std::endl; }
};

TEST_P(QuantBlockSparseAttnInferShapeTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<TensorDesc> inputTensorDesc({
        param.query,
        param.key,
        param.value,
        param.qDescale,
        param.kDescale,
        param.vDescale,
        param.pScale,
        param.cuSeqlensQ,
        param.cuSeqlensKv,
        param.sequsedQ,
        param.sequsedKv,
        param.sparseIndices,
        param.sparseSeqLen,
        param.blockTable,
        param.attenMask,
        param.metadata,
    });

    std::vector<TensorDesc> outputTensorDesc({
        param.attentionOut,
        param.softmaxLse,
    });

    std::vector<OpAttr> attrs({
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_q_block_size)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_kv_block_size)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_sparse_indices)},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_mode)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
    });

    gert::InfershapeContextPara para(OP_NAME, inputTensorDesc, outputTensorDesc, attrs);

    ExecuteTestCase(para, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnInferShape, QuantBlockSparseAttnInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnInferShapeUtParam>);

namespace {
template <QBSALayout layout>
void CheckPaddedLayout()
{
    constexpr uint32_t batches = 3, heads = 8, sq = 129;
    std::vector<uint64_t> slots;
    // Independent storage order; each cell records its logical B,N,S identity.
    for (uint32_t b = 0; b < batches; ++b) {
        const uint32_t outer = layout == QBSALayout::BSND ? sq : heads;
        const uint32_t inner = layout == QBSALayout::BSND ? heads : sq;
        for (uint32_t i = 0; i < outer; ++i) {
            for (uint32_t j = 0; j < inner; ++j) {
                const uint32_t n = layout == QBSALayout::BSND ? j : i;
                const uint32_t s = layout == QBSALayout::BSND ? i : j;
                slots.push_back((b * heads + n) * sq + s);
            }
        }
    }
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t n = 0; n < heads; ++n) {
            for (uint32_t s = 0; s < sq; ++s) {
                const uint64_t expected = (b * heads + n) * sq + s;
                const auto slot = regbasemx::MxQuerySlot<layout>(b * sq, s, n, heads, sq);
                ASSERT_LT(slot, slots.size());
                EXPECT_EQ(slots[slot], expected);
                EXPECT_EQ(regbasemx::MxLseSlot<layout>(b * sq, s, n, heads, sq), expected);
            }
        }
    }
}

TEST(QbsaMxQueryLayout, BSND) { CheckPaddedLayout<QBSALayout::BSND>(); }
TEST(QbsaMxQueryLayout, BNSD) { CheckPaddedLayout<QBSALayout::BNSD>(); }
TEST(QbsaMxQueryLayout, PackedTndAndWideOffsets)
{
    const uint64_t base = uint64_t{1} << 30;
    EXPECT_EQ(regbasemx::MxQuerySlot<QBSALayout::TND>(base, 7, 3, 8, 0), (base + 7) * 8 + 3);
    EXPECT_EQ(regbasemx::MxLseSlot<QBSALayout::TND>(base, 7, 3, 8, 0), (base + 7) * 8 + 3);
    EXPECT_EQ(regbasemx::MxQuerySlot<QBSALayout::BNSD>(base, 7, 3, 8, 129), base * 8 + 3 * 129 + 7);
}
} // namespace

} // namespace QuantBlockSparseAttnUT
