/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdint>
#include <climits>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
using namespace at_npu::native;

const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;

std::tuple<at::Tensor, at::Tensor> npu_ai_infra_mhc_pre_split_post_res(const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias, double norm_eps)
{

    c10::SymInt T, N, B, S;
    at::Tensor h_post_out, h_res_out;

    if (x.dim() == DIM_3) {
        T = x.sym_size(DIM_0);
        N = x.sym_size(DIM_1);
        
        h_post_out = at::empty_symint(
            {T, N},
            phi.options().dtype(at::kFloat)
        );

        h_res_out = at::empty_symint(
            {T, N, N},
            phi.options().dtype(at::kFloat)
        );
    } else {
        B = x.sym_size(DIM_0);
        S = x.sym_size(DIM_1);
        N = x.sym_size(DIM_2);

        h_post_out = at::empty_symint(
            {B, S, N},
            phi.options().dtype(at::kFloat)
        );

        h_res_out = at::empty_symint(
            {B, S, N, N},
            phi.options().dtype(at::kFloat)
        );
    }

    EXEC_NPU_CMD_V1(aclnnAiInfraMhcPreSplitPostRes,
        x, phi, alpha, bias, norm_eps,
        h_post_out, h_res_out
    );

    return std::make_tuple(h_post_out, h_res_out);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_ai_infra_mhc_pre_split_post_res_meta(const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias, double norm_eps)
{
    c10::SymInt T, N, B, S;
    at::Tensor h_post_out, h_res_out;

    if (x.dim() == DIM_3) {
        T = x.sym_size(DIM_0);
        N = x.sym_size(DIM_1);
        
        h_post_out = at::empty_symint(
            {T, N},
            phi.options().dtype(at::kFloat)
        );

        h_res_out = at::empty_symint(
            {T, N, N},
            phi.options().dtype(at::kFloat)
        );
    } else {
        B = x.sym_size(DIM_0);
        S = x.sym_size(DIM_1);
        N = x.sym_size(DIM_2);

        h_post_out = at::empty_symint(
            {B, S, N},
            phi.options().dtype(at::kFloat)
        );

        h_res_out = at::empty_symint(
            {B, S, N, N},
            phi.options().dtype(at::kFloat)
        );
    }

    return std::make_tuple(h_post_out, h_res_out);
}
}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_ai_infra_mhc_pre_split_post_res", &custom::npu_ai_infra_mhc_pre_split_post_res);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_ai_infra_mhc_pre_split_post_res", &custom::npu_ai_infra_mhc_pre_split_post_res_meta);
}