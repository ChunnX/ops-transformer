/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/mla_prolog_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MlaPrologV3 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MlaPrologV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MlaPrologV3 TearDown" << std::endl;
    }
};

//全量化kvcache pertensor量化
TEST_F(MlaPrologV3, MlaProlog_tiling_test0) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND},//token_x
        {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq
        {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr
        {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk
        {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv
        {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin
        {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos
        {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index
        {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq
        {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr
        {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha
    },
    {
        {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},//query
        {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},//todo!
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo);
    int64_t expectTilingKey = 1574177;
    string expectTilingData = "34359738376 0 34359738400 4294967297 6597069773824 17592186044928 274877908992 4294967328 274877907072 549755813904 38654705688 137438953504 274877906952 824633720896 2199023255553 549755813889 0 4191350054637797376 4251398049163101612 925353388 4575657222473777152 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(MlaPrologV3, MlaProlog_tiling_test1) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{8, 1, 6144}, {8, 1, 6144}}, ge::DT_INT8, ge::FORMAT_ND},//token_x
        {{{6144, 1536}, {6144, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq
        {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr
        {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk
        {{{6144, 512 + 64}, {6144, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv
        {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin
        {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos
        {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index
        {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq
        {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr
        {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha
    },
    {
        {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},//query
        {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},//todo!
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo);
    int64_t expectTilingKey = 1574177;
    string expectTilingData = "34359738376 0 34359738400 4294967297 6597069772800 17592186044928 274877908992 4294967328 274877907072 549755813904 38654705688 137438953504 274877906952 824633720896 2199023255553 549755813889 0 4191350054637797376 4251398049163101612 925353388 4575657222473777152 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

//全量化kvcache pertensor量化 NZ
TEST_F(MlaPrologV3, MlaProlog_tiling_test2) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND},//token_x
        {{{1536/32, 7168/16, 16, 32}, {1536/32, 7168/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq
        {{{128 + 64, 1536/16, 16, 32}, {128 + 64, 1536/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr
        {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk
        {{{(512 + 64)/32, 7168/16, 16, 32}, {(512 + 64)/32, 7168/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv
        {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin
        {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos
        {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index
        {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq
        {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr
        {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha
    },
    {
        {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},//query
        {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{8 * 1, 32, 1}, {8 * 1, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},//todo!
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910 B", 48, 196608, 16384);
    int64_t expectTilingKey = 1574180;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {17393920};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 非PA半量化 BSND qnorm_flag==true /mlaprolog_L0_1_128_1_11776_11776_128_BSND_1_000035
TEST_F(MlaPrologV3, MlaProlog_tiling_test4) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{1,11776,7168}, {1,11776,7168}}, ge::DT_BF16, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{1,11776,64}, {1,11776,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{1,11776,64}, {1,11776,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{1,11776,1,512}, {1,11776,1,512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache 9
        {{{1,11776,1,64}, {1,11776,1,64}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache 10
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{1,512}, {1,512}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{1,64}, {1,64}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{1,1536}, {1,1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{1,11776,128,512}, {1,11776,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{1,11776,128,64}, {1,11776,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{1,11776,1,512}, {1,11776,1,512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{1,11776,1,64}, {1,11776,1,64}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{1,11776,1536}, {1,11776,1536}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{11776,1}, {11776,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.000875660550762735)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.000607527341571764)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1574048;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {34492416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 非PA全量化 BSND qnorm_flag==false /mlaprolog_L0_1_128_1_1024_1024_128_BSND_2_000037
TEST_F(MlaPrologV3, MlaProlog_tiling_test5) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{1,1024,7168}, {1,1024,7168}}, ge::DT_INT8, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{1,1024,64}, {1,1024,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{1,1024,64}, {1,1024,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{1,1024,1,512}, {1,1024,1,512}}, ge::DT_BF16, ge::FORMAT_ND},//kv_cache 9
        {{{1,1024,1,64}, {1,1024,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache 10
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{1024,1}, {1024,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{1,1536}, {1,1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{1,576}, {1,576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{1,1024,128,512}, {1,1024,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{1,1024,128,64}, {1,1024,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{1,1024,1,512}, {1,1024,1,512}}, ge::DT_BF16, ge::FORMAT_ND},//kv_cache
        {{{1,1024,1,64}, {1,1024,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00263855234125857)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00306797597892537)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1574112;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {35033088};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
  
// 非PA全量化 TND qnorm_flag==false /mlaprolog_L0_4_128_1_2_2_128_TND_2_000016
TEST_F(MlaPrologV3, MlaProlog_tiling_test6) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{8,7168}, {8,7168}}, ge::DT_INT8, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{8,64}, {8,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{8,64}, {8,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{8,1,656}, {8,1,656}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache 9
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache 10
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{8,1}, {8,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{1,1536}, {1,1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{1,576}, {1,576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{8,128,512}, {8,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{8,128,64}, {8,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{8,1,656}, {8,1,656}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0022971812167027)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00235037235057241)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1574304;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {17918208};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 非PA全量化 BSND qnorm_flag==TRUE /mlaprolog_L0_1_128_1_6016_6016_128_BSND_2_000020
TEST_F(MlaPrologV3, MlaProlog_tiling_test7) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{1,6016,7168}, {1,6016,7168}}, ge::DT_INT8, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{1,6016,64}, {1,6016,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{1,6016,64}, {1,6016,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{1,6016,1,512}, {1,6016,1,512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache 9
        {{{1,6016,1,64}, {1,6016,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache 10
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{6016,1}, {6016,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{1,1536}, {1,1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{1,576}, {1,576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{1,6016,128,512}, {1,6016,128,512}}, ge::DT_INT8, ge::FORMAT_ND},//query
        {{{1,6016,128,64}, {1,6016,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{1,6016,1,512}, {1,6016,1,512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{1,6016,1,64}, {1,6016,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{6016,128,1}, {6016,128,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{1,6016,1536}, {1,6016,1536}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{6016,1}, {6016,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.000971848152426739)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00141311948888896)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1574176;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {51810304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 非PA全量化 TND qnorm_flag==false /mlaprolog_L0_1_128_1_512_512_128_TND_1_000015
TEST_F(MlaPrologV3, MlaProlog_tiling_test8) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{512,7168}, {512,7168}}, ge::DT_BF16, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{512,64}, {512,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{512,64}, {512,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{512,1,656}, {512,1,656}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache 9
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache 10
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{512,128,512}, {512,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{512,128,64}, {512,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{512,1,656}, {512,1,656}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{512,1536}, {512,1536}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{512, 1}, {512, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.000651222723526155)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00268072038395418)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1574240;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {34492416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// mlaprolog_L0_2_128_1_4825_65536_128_TND_PA_BLK_NZ_1_2_000028 
TEST_F(MlaPrologV3, MlaProlog_tiling_test9) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{9650,6144}, {9650,6144}}, ge::DT_BF16, ge::FORMAT_ND},//token_x 0
        {{{6144,1536}, {6144,1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536, 24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{6144,576}, {6144,576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{9650,64}, {9650,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{9650,64}, {9650,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{512,256,1,512}, {512,256,1,512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache 9
        {{{512,256,1,64}, {512,256,1,64}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache 10
        {{{38}, {38}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{1,512}, {1,512}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{1,64}, {1,64}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{1,1536}, {1,1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{9650,128,512}, {9650,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{9650,128,64}, {9650,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{512,256,1,512}, {512,256,1,512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{512,256,1,64}, {512,256,1,64}}, ge::DT_INT8, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00324166008356131)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00218671504998371)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1590436;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {34492416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// mlaprolog_L0_16_128_1_2_2_128_BSND_1_000000
TEST_F(MlaPrologV3, MlaProlog_tiling_test10) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{16,2,7168}, {16,2,7168}}, ge::DT_BF16, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168,1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536,24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168,576}, {7168,576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{16,2,64}, {16,2,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{16,2,64}, {16,2,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{16,2,1,512}, {16,2,1,512}}, ge::DT_BF16, ge::FORMAT_ND},//kv_cache 9
        {{{16,2,1,64}, {16,2,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache 10
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{1,1536}, {1,1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{16,2,128,512}, {16,2,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{16,2,128,64}, {16,2,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{16,2,1,512}, {16,2,1,512}}, ge::DT_BF16, ge::FORMAT_ND},//kv_cache
        {{{16,2,1,64}, {16,2,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{16,2,1536}, {16,2,1536}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{32,1}, {32,1}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.00457361635559103)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.00484664050635162)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1573984;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21206016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// mlaprolog_L0_4_128_1_2317_65536_128_BSND_PA_BLK_NZ_1_0_000011
TEST_F(MlaPrologV3, MlaProlog_tiling_test11) {
    optiling::MlaPrologCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara("MlaPrologV3",
    {
        {{{4,2317,7168}, {4,2317,7168}}, ge::DT_BF16, ge::FORMAT_ND},//token_x 0
        {{{7168,1536}, {7168,1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dq 1
        {{{1536,24576}, {1536,24576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr 2
        {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk 3
        {{{7168,576}, {7168,576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr 4
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq 5
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv 6
        {{{4,2317,64}, {4,2317,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin 7
        {{{4,2317,64}, {4,2317,64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos 8
        {{{2048,128,1,512}, {2048,128,1,512}}, ge::DT_BF16, ge::FORMAT_ND},//kv_cache 9
        {{{2048,128,1,64}, {2048,128,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache 10
        {{{4,19}, {4,19}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index 11
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x 12
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq 13
        {{{1,24576}, {1,24576}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr 14
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr 15
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv 16
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr 17
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq 18
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len 19
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//k_nope_clip_alpha 20
    },
    {
        {{{4,2317,128,512}, {4,2317,128,512}}, ge::DT_BF16, ge::FORMAT_ND},//query
        {{{4,2317,128,64}, {4,2317,128,64}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{2048,128,1,512}, {2048,128,1,512}}, ge::DT_BF16, ge::FORMAT_ND},//kv_cache
        {{{2048,128,1,64}, {2048,128,1,64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_norm
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(0.0033201761781808)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(0.0034101718259947)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
        {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}, // 128 : set value of tile size
        {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
    },
    &compileInfo, "Ascend910B", 48, 196608, 16384);
    int64_t expectTilingKey = 1573988;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {34492416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}