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
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> MhcSandwichNormPostPreonly(
    const at::Tensor &hOut, const at::Tensor &residual, const at::Tensor &hPost,
    const c10::optional<at::Tensor> &hResOptional, const at::Tensor &phi, const at::Tensor &alpha0,
    const at::Tensor &bias, const at::Tensor &gamma0, const at::Tensor &gamma1, const c10::optional<at::Tensor> &gamma2,
    double normEps, double mhcPreNormEps, double hcEps, bool returnHInF32)
{
    auto [hInPrime, x2Out] = MakeOutputs(hOut, residual);
    at::Tensor hInF32;
    if (returnHInF32) {
        hInF32 = at::empty_symint(hInPrime.sym_sizes(), hOut.options().dtype(at::kFloat));
    }
    const float normEpsFloat = static_cast<float>(normEps);
    const float mhcPreNormEpsFloat = static_cast<float>(mhcPreNormEps);
    const float hcEpsFloat = static_cast<float>(hcEps);
    ACLNN_CMD(aclnnMhcSandwichNormPostPreonly, hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma1, gamma0,
              gamma2, normEpsFloat, mhcPreNormEpsFloat, hcEpsFloat, returnHInF32, hInPrime, x2Out, hInF32);
    return {hInPrime, x2Out, hInF32};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_sandwich_norm_post_preonly", &MhcSandwichNormPostPreonly);
}

} // namespace op_api
