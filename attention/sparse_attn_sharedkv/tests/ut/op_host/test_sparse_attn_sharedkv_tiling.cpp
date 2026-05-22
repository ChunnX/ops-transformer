/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sparse_attn_sharedkv_tiling.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

struct SASCompileInfo {
    int64_t core_num;
};

class SparseAttnSharedkvTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseAttnSharedkvTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseAttnSharedkvTiling TearDown" << std::endl;
    }
};

TEST_F(SparseAttnSharedkvTiling, test_tiling_swa_only_ori_kv_fp16_tnd_pa_nd)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_swa_only_ori_kv_bf16_tnd_pa_nd)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_cfa_ori_and_cmp_kv_fp16_tnd_pa_nd)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_scfa_with_sparse_indices_fp16_tnd_pa_nd)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_scfa_with_sparse_indices_bf16_tnd_pa_nd)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_n1_not_64_failed)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 32, 512}, {512, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 32, 512}, {512, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 32, 1}, {512, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_ori_kv_null_failed)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_cmp_sparse_indices_without_cmp_kv_failed)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SparseAttnSharedkvTiling, test_tiling_unsupported_dtype_failed)
{
    SASCompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[1024] = {0};
    gert::TilingContextPara tilingContextPara(
        "SparseAttnSharedkv",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 64, 1}, {512, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_kv_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
