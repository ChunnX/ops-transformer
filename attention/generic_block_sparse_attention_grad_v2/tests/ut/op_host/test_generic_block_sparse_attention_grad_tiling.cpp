/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <iostream>
#include <gtest/gtest.h>
#include <string>

// 引入自动生成的 Tiling 头文件
#include "../../../op_host/generic_block_sparse_attention_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class GenericBlockSparseAttentionGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- GenericBlockSparseAttentionGradTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- GenericBlockSparseAttentionGradTiling UT TearDown ---" << std::endl;
    }
};

TEST_F(GenericBlockSparseAttentionGradTilingTest, tiling_bnsd_case0)
{
    optiling::GenericBlockSparseAttentionGradCompileInfo compileInfo;

    int64_t b = 1, n = 4, n_kv = 2, s = 128, s_kv = 128, d = 128;
    int64_t blockX = 64, blockY = 64;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t actualSeqData[1] = {s};
    int64_t actualSeqKvData[1] = {s_kv};

    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttentionGrad",
        {// --- Input Info ---
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n, s}, {b, n, s}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, n, ceilKv, ceilQ}, {b, n, ceilKv, ceilQ}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{b, n, ceilKv}, {b, n, ceilKv}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT32, ge::FORMAT_ND, (void *)actualSeqData},
         {{{b}, {b}}, ge::DT_INT32, ge::FORMAT_ND, (void *)actualSeqKvData}},
        {// --- Output Info ---
         {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// --- Attr Info ---

         {"block_shape",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{blockX, blockY})},
         {"isPackedGQA", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"layoutKv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmaxPrecision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"winLeft", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"winRight", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}},
        &compileInfo);

    uint64_t expectTilingKey = 101UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
