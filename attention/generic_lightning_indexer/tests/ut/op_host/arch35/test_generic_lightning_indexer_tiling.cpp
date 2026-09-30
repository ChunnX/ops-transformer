/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/tilingdata_base.h"
#include "../../../../op_host/generic_lightning_indexer_tiling.h"

using namespace std;

namespace {
constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;

using AttrList = std::vector<gert::TilingContextPara::OpAttr>;
using Td = gert::TilingContextPara::TensorDescription;

AttrList MakeAttrs(int64_t topk, int64_t cmpRatio, const char *layoutQ, const char *layoutK, int64_t sparseMode,
                   bool hasWeighted, bool expand, bool append, int64_t quantMode, bool returnValue,
                   int64_t selectMode = 0, int64_t keyReduce = 0, int64_t actType = 0, float scoreScale = 1.0f)
{
    return {
        {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
        {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutK)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
        {"has_weighted_sum", Ops::Transformer::AnyValue::CreateFrom<bool>(hasWeighted)},
        {"expand_indices", Ops::Transformer::AnyValue::CreateFrom<bool>(expand)},
        {"append_tail", Ops::Transformer::AnyValue::CreateFrom<bool>(append)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(returnValue)},
        {"select_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(selectMode)},
        {"key_reduce_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(keyReduce)},
        {"score_scale", Ops::Transformer::AnyValue::CreateFrom<float>(scoreScale)},
        {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"k_descale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
}

Td NullIn(ge::DataType dt = ge::DT_INT32)
{
    return {{{}, {}}, dt, ge::FORMAT_ND};
}

std::vector<Td> GliInputs(Td query, Td key, Td weights, Td residual = NullIn(), Td actQ = NullIn(), Td actK = NullIn(),
                          Td qDescale = NullIn(ge::DT_FLOAT), Td kDescale = NullIn(ge::DT_FLOAT),
                          Td blockTable = NullIn())
{
    return {query,    key,        weights,  actQ,     actK,     NullIn(), NullIn(),
            residual, blockTable, qDescale, kDescale, NullIn(), NullIn(), NullIn()};
}
} // namespace

class GenericLightningIndexerTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GenericLightningIndexerTilingArch35 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "GenericLightningIndexerTilingArch35 TearDown" << std::endl;
    }
};

TEST_F(GenericLightningIndexerTilingArch35, P_LI_success)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{2, 16, 8, 128}, {2, 16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{2, 256, 1, 128}, {2, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
                                 {
                                     {{{2, 16, 2048}, {2, 16, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2048, 1, "BSND", "BSND", 3, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_LIv2_cmp_ratio_needs_residual)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t residual[] = {1, 2};
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{2, 8, 16, 128}, {2, 8, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{2, 8, 16}, {2, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, residual}),
                                 {
                                     {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2048, 4, "BSND", "BSND", 3, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_LIv2_missing_residual_failed)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{2, 8, 16, 128}, {2, 8, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{2, 8, 16}, {2, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND}),
                                 {
                                     {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2048, 4, "BSND", "BSND", 3, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GenericLightningIndexerTilingArch35, P_PKI_expand_append)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t residual[] = {3};
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 32, 8, 128}, {1, 32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 8}, {1, 32, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, residual}),
        {
            {{{1, 32, 128 * 16 + 15}, {1, 32, 128 * 16 + 15}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(128, 16, "BSND", "BSND", 3, true, true, true, -1, false, 0, 0, 0, 0.088388347f), &compileInfo,
        "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QWEN_sum_no_weights)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t residual[] = {1};
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 16, 8, 128}, {1, 16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16),
                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, residual}),
        {
            {{{1, 16, 64 * 4 + 3}, {1, 16, 64 * 4 + 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 64}, {1, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(64, 4, "BSND", "BSND", 4, false, true, true, -1, true), &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QWEN_weights_forbidden)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{1, 16, 8, 128}, {1, 16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{1, 16, 8}, {1, 16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
                                 {
                                     {{{1, 16, 64}, {1, 16, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(64, 1, "BSND", "BSND", 4, false, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GenericLightningIndexerTilingArch35, P_MSA_REJ_key_reduce)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16)),
        {
            {{{1, 8, 32}, {1, 8, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(32, 1, "BSND", "BSND", 0, false, false, false, -1, false, 0, 1), &compileInfo, "Ascend950", 56,
        262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GenericLightningIndexerTilingArch35, P_MSA_identity_maxpool)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16)),
        {
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 8, 8, 16}, {1, 8, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(16, 16, "BSND", "BSND", 3, false, false, false, -1, true, 1, 1, 1), &compileInfo, "Ascend950", 56,
        262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QLI_fp8_pth)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                           {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                           {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(),
                                           NullIn(), NullIn(), {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{1, 256, 1}, {1, 256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}),
                                 {
                                     {{{1, 8, 1024}, {1, 8, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(1024, 1, "BSND", "BSND", 3, true, false, false, 0, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QLIv2_mxfp8)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t residual[] = {1};
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                  {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, residual}, NullIn(), NullIn(),
                  {{{1, 8, 16, 4}, {1, 8, 16, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                  {{{1, 64, 1, 4}, {1, 64, 1, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}),
        {
            {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(2048, 4, "BSND", "BSND", 3, true, false, false, 3, true), &compileInfo, "Ascend950", 56, 262144,
        16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_LI_PA_BBND)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t actK[] = {256};
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{8, 128, 1, 128}, {8, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(),
                                           {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actK}, NullIn(ge::DT_FLOAT),
                                           NullIn(ge::DT_FLOAT), {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND}),
                                 {
                                     {{{1, 8, 128}, {1, 8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(128, 1, "BSND", "PA_BBND", 3, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QLI_int8)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                  {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(), NullIn(),
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 256, 1}, {1, 256, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
        {
            {{{1, 8, 1024}, {1, 8, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1024, 1, "BSND", "BSND", 3, true, false, false, 2, false), &compileInfo, "Ascend950", 56, 262144,
        16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, TilingDataClassRegistered)
{
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("GenericLightningIndexer"), nullptr);
}

TEST_F(GenericLightningIndexerTilingArch35, TND_cu_seqlens_Bplus1)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int32_t actQ[] = {0, 8, 16};
    int32_t actK[] = {0, 32, 64};
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{16, 8, 128}, {16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(),
                                           {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actQ},
                                           {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actK}),
                                 {
                                     {{{16, 512}, {16, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(512, 1, "TND", "TND", 0, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, TND_exclusive_prefix_B)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int32_t actQ[] = {8, 16};
    int32_t actK[] = {32, 64};
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{16, 8, 128}, {16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(),
                                           {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actQ},
                                           {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actK}),
                                 {
                                     {{{16, 512}, {16, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(512, 1, "TND", "TND", 0, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_PKI_int64_residual)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t residual[] = {3};
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 32, 8, 128}, {1, 32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 8}, {1, 32, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, residual}),
        {
            {{{1, 32, 128 * 16 + 15}, {1, 32, 128 * 16 + 15}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(128, 16, "BSND", "BSND", 3, true, true, true, -1, false, 0, 0, 0, 0.088388347f), &compileInfo,
        "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_LI_BNBD)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    int64_t actK[] = {256};
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{8, 1, 128, 128}, {8, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(),
                                           {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actK}, NullIn(ge::DT_FLOAT),
                                           NullIn(ge::DT_FLOAT), {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND}),
                                 {
                                     {{{1, 8, 128}, {1, 8, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(128, 1, "BSND", "BNBD", 3, true, false, false, -1, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QLI_fp8_fp16_descale)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                  {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, NullIn(), NullIn(), NullIn(),
                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 256, 1}, {1, 256, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}),
        {
            {{{1, 8, 1024}, {1, 8, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1024, 1, "BSND", "BSND", 3, true, false, false, 0, false), &compileInfo, "Ascend950", 56, 262144,
        16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_MSA_mixed_fp16_int8)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para(
        "GenericLightningIndexer",
        GliInputs({{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                  {{{1, 32, 1, 128}, {1, 32, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, NullIn(ge::DT_FLOAT16), NullIn(),
                  NullIn(), NullIn(), NullIn(ge::DT_FLOAT), {{{1, 1, 32}, {1, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}),
        {
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 8, 8, 16}, {1, 8, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(16, 16, "BSND", "BSND", 3, false, false, false, -1, true, 1, 1, 1), &compileInfo, "Ascend950", 56,
        262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QWEN_fp8_key)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{1, 16, 8, 128}, {1, 16, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                           NullIn(ge::DT_FLOAT16), NullIn(), NullIn(), NullIn(), NullIn(ge::DT_FLOAT),
                                           {{{1, 8, 1}, {1, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}),
                                 {
                                     {{{1, 16, 8 * 4 + 3}, {1, 16, 8 * 4 + 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{1, 16, 8}, {1, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(8, 4, "BSND", "BSND", 4, false, true, true, -1, true), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

TEST_F(GenericLightningIndexerTilingArch35, P_QWEN_fp8_key_bf16)
{
    optiling::GenericLightningIndexerCompileInfo compileInfo;
    gert::TilingContextPara para("GenericLightningIndexer",
                                 GliInputs({{{1, 16, 8, 128}, {1, 16, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                           {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                           NullIn(ge::DT_BF16), NullIn(), NullIn(), NullIn(), NullIn(ge::DT_FLOAT),
                                           {{{1, 8, 1}, {1, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}),
                                 {
                                     {{{1, 16, 8 * 4 + 3}, {1, 16, 8 * 4 + 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{1, 16, 8}, {1, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(8, 4, "BSND", "BSND", 4, false, true, true, 0, false), &compileInfo,
                                 "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}
