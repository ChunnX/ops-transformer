/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MlaPrologV3Proto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MlaPrologV3Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MlaPrologV3Proto TearDown" << std::endl;
    }
};

TEST_F(MlaPrologV3Proto, mla_prolog_v3_infershape_0)
{
    // char* cacheMode = "PA_BSND";
    gert::InfershapeContextPara infershapeContextPara("MlaPrologV3",
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
        {{{}, {}},ge::DT_FLOAT , ge::FORMAT_ND},//quant_scale_ckr
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq
        {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},//actual_seq_len
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}//k_nope_clip_alpha
    },
    {
        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},//query
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}//dequant_scale_q_norm
    },
    {
        // {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
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
        {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 1, 32, 512}, {8, 1, 32, 64}, {16, 128, 1, 512}, {16, 128, 1, 64}, {8 * 1, 32, 1}, {1}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MlaPrologV3Proto, mla_prolog_v3_infershape_1)
{
    // char* cacheMode = "PA_BSND";
    gert::InfershapeContextPara infershapeContextPara("MlaPrologV3",
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
        {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},//query_norm
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}//dequant_scale_q_norm
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
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 1, 32, 512}, {8, 1, 32, 64}, {16, 128, 1, 512}, {16, 128, 1, 64}, {8 * 1, 32, 1}, {1}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}