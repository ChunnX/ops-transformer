/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor> MakeOutputs(const at::Tensor &hOut, const at::Tensor &residual)
{
    if (hOut.dim() == DIM_TWO) {
        return {
            at::empty_symint({hOut.sym_size(0), hOut.sym_size(1)}, hOut.options()),
            at::empty_symint({hOut.sym_size(0), residual.sym_size(1), hOut.sym_size(1)}, hOut.options()),
        };
    }
    return {
        at::empty_symint({hOut.sym_size(0), hOut.sym_size(1), hOut.sym_size(2)}, hOut.options()),
        at::empty_symint({hOut.sym_size(0), hOut.sym_size(1), residual.sym_size(2), hOut.sym_size(2)}, hOut.options()),
    };
}

// h_pre 为接口预留的可选输出，形状与 h_post 一致：2D [T, N]，3D [B, S, N]。
at::Tensor MakeHPreOutput(const at::Tensor &hOut, const at::Tensor &residual)
{
    if (hOut.dim() == DIM_TWO) {
        return at::empty_symint({hOut.sym_size(0), residual.sym_size(1)}, hOut.options().dtype(at::kFloat));
    }
    return at::empty_symint({hOut.sym_size(0), hOut.sym_size(1), residual.sym_size(2)},
                            hOut.options().dtype(at::kFloat));
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcNormPostPreonly(
    const at::Tensor &hOut, const at::Tensor &residual, const at::Tensor &hPost, const c10::optional<at::Tensor> &hResl,
    const at::Tensor &phi, const at::Tensor &alpha0, const at::Tensor &bias, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &preMix, const c10::optional<at::Tensor> &gamma0,
    const c10::optional<at::Tensor> &gamma2, double normEps, double mhcPreNormEps, double hcEps, bool returnHInF32)
{
    auto [hInPrime, x2Out] = MakeOutputs(hOut, residual);
    at::Tensor hInF32;
    if (returnHInF32) {
        hInF32 = at::empty_symint(hInPrime.sym_sizes(), hOut.options().dtype(at::kFloat));
    }
    // h_pre 为接口预留的可选输出，当前暂不支持计算，返回带预期形状的空 Tensor 占位；
    // 传入 ACLNN 的可选输出使用未定义 Tensor，由 ACLNN 侧按 nullptr 处理。
    at::Tensor hPre = MakeHPreOutput(hOut, residual);
    at::Tensor hPreForAclnn;
    const float normEpsFloat = static_cast<float>(normEps);
    const float mhcPreNormEpsFloat = static_cast<float>(mhcPreNormEps);
    const float hcEpsFloat = static_cast<float>(hcEps);

    ACLNN_CMD(aclnnMhcNormPostPreonly, hOut, residual, hPost, hResl, phi, alpha0, bias, gamma1, preMix, gamma0,
              gamma2, normEpsFloat, mhcPreNormEpsFloat, hcEpsFloat, returnHInF32, hInPrime, x2Out, hInF32,
              hPreForAclnn);
    return {hInPrime, x2Out, hInF32, hPre};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_norm_post_preonly", &MhcNormPostPreonly);
}

} // namespace op_api
