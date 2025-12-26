/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MlaPrologV2Proto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MlaPrologV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MlaPrologV2Proto TearDown" << std::endl;
    }
};

TEST_F(MlaPrologV2Proto, mla_prolog_v2_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MlaPrologV2",
    {
        {{{48, 10, 7168}, {48, 10, 7168}}, ge::DT_BF16, ge::FORMAT_ND},//token_x
        {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dq
        {{{1536, 768}, {1536, 768}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_uq_qr
        {{{4, 128, 512}, {4, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},//weight_uk
        {{{7168, 576}, {7168, 576}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},//weight_dkv_kr
        {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_cq
        {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},//rmsnorm_gamma_ckv
        {{{48, 10, 64}, {48, 10, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_sin
        {{{48, 10, 64}, {48, 10, 64}}, ge::DT_BF16, ge::FORMAT_ND},//rope_cos
        {{{49488, 16, 1, 512}, {49488, 16, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},//kv_cache
        {{{49488, 16, 1, 64}, {49488, 16, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},//kr_cache
        {{{48, 10}, {48, 10}}, ge::DT_INT64, ge::FORMAT_ND},//cache_index
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_x
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dq
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_uq_qr
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_w_dkv_kr
        {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckv
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//quant_scale_ckr
        {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},//smooth_scales_cq
    },
    {
        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},//query
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},//query_rope
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//kv_cache
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//kr_cache  
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},//dequant_scale_q_nope
    },
    {
        {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{48, 10, 4, 512}, {48, 10, 4, 64}, {49488, 16, 1, 64}, {48, 10}, {480, 4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}