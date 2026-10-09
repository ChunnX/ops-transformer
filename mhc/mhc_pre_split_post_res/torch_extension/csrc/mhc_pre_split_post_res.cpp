/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_split_post_res.cpp
 * \brief ACLNN Wrapper for aclnnMhcPreSplitPostRes
 */

#include <cmath>

#include <torch/extension.h>
#include "aclnn_common.h"

// N 固定规格：与 op_host/mhc_pre_split_post_res_const.h 的 MHC_N_VALUE(=4) 同源互锚
// （torch 扩展构建隔离无法 include host 头，改动必须两侧同步）
constexpr int64_t kMhcNValue = 4;

namespace op_api {

std::tuple<at::Tensor, at::Tensor> MhcPreSplitPostRes(
    const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias,
    double normEps)
{
    TORCH_CHECK(std::isfinite(normEps) && normEps > 0.0, "norm_eps must be finite and greater than zero, but got ",
                normEps);

    // h_res 为可选输出：phi 行数为 N^2+N（完整形态）时分配真 shape；
    // 仅 post 形态（phi 行数==N）时第二个返回 0 元素 tensor（aclnn 层按 phi shape 忽略）
    const int64_t n = x.size(x.dim() - 2);
    TORCH_CHECK(n == kMhcNValue, "n must be ", kMhcNValue, ", but got ", n);
    const bool hasRes = phi.size(0) == n * n + n;
    std::vector<int64_t> outSizes = x.sizes().vec();
    outSizes.pop_back();  // [T, n] / [B, S, n]
    at::Tensor hPostOut = at::empty(outSizes, phi.options());
    at::Tensor hResOut;
    if (hasRes) {
        outSizes.push_back(n);  // [T, n, n] / [B, S, n, n]
        hResOut = at::empty(outSizes, phi.options());
    } else {
        hResOut = at::empty({0}, phi.options());
    }

    // torch schema 的 float 在 C++ 侧为 64 位 double；aclnn 及其后各层为 32 位 float，
    // 在本层显式收窄，转换点唯一化。ACLNN_CMD 内部以非 const 左值引用转发参数，
    // 故先落具名局部变量（左值）再传入。
    float normEpsF = static_cast<float>(normEps);
    ACLNN_CMD(aclnnMhcPreSplitPostRes, x, phi, alpha, bias, normEpsF, hPostOut, hResOut);

    return std::tuple<at::Tensor, at::Tensor>(hPostOut, hResOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_split_post_res", &MhcPreSplitPostRes, "mhc_pre_split_post_res");
}

} // namespace op_api
