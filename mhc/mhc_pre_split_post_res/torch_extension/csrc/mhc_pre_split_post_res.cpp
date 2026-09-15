/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_split_post_res.cpp
 * \brief ACLNN Wrapper for aclnnMhcPreSplitPostRes
 */

#include <tuple>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t X_RANK_TND = 3;
constexpr int64_t X_RANK_BSND = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;

// x 支持 [T, n, D]（TND）与 [B, S, n, D]（BSND）两种布局：
//   TND  -> h_post [T, n],       h_res [T, n, n]
//   BSND -> h_post [B, S, n],    h_res [B, S, n, n]
// 输出固定 FP32（与 phi dtype 一致），layout 跟随 x。
std::tuple<at::Tensor, at::Tensor> CreateOutputTensors(const at::Tensor &x, const at::Tensor &phi)
{
    auto outOpt = phi.options().dtype(at::kFloat);
    if (x.dim() == X_RANK_TND) {
        auto T = x.size(DIM_0);
        auto n = x.size(DIM_1);
        return std::make_tuple(at::empty({T, n}, outOpt), at::empty({T, n, n}, outOpt));
    }
    auto B = x.size(DIM_0);
    auto S = x.size(DIM_1);
    auto n = x.size(DIM_2);
    return std::make_tuple(at::empty({B, S, n}, outOpt), at::empty({B, S, n, n}, outOpt));
}

} // namespace

std::tuple<at::Tensor, at::Tensor> MhcPreSplitPostRes(
    const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias, double normEps)
{
    auto [hPostOut, hResOut] = CreateOutputTensors(x, phi);

    ACLNN_CMD(aclnnMhcPreSplitPostRes, x, phi, alpha, bias, normEps, hPostOut, hResOut);

    return std::make_tuple(hPostOut, hResOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_split_post_res", &MhcPreSplitPostRes, "mhc_pre_split_post_res");
}

} // namespace op_api
