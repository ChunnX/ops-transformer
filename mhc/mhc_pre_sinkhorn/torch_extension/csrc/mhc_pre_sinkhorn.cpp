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
 * \file mhc_pre_sinkhorn.cpp
 * \brief ACLNN Wrapper composed from aclnnMhcPre and aclnnMhcSinkhorn
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

namespace {
constexpr int64_t SINKHORN_ALIGN_N = 8;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
MhcPreSinkhorn(const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias,
               int64_t hcMult, int64_t numIters, double hcEps, double normEps, bool outFlag)
{
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "x dim must be 3 (t, n, d) or 4 (b, s, n, d), but got ", x.dim());
    bool isTnd = (x.dim() == 3);

    at::Tensor hin;
    at::Tensor hPost;
    at::Tensor hResBeforeSinkhorn;
    at::Tensor hRes;
    at::Tensor hPre;
    at::Tensor hcBeforeNorm;
    at::Tensor invRms;
    at::Tensor sumOut;
    at::Tensor normOut;

    int64_t skIterCount = numIters;
    int64_t n = hcMult;

    if (isTnd) {
        int64_t t = x.size(0);
        int64_t c = x.size(2);
        hin = at::empty({t, c}, x.options());
        hPost = at::empty({t, n}, phi.options());
        hRes = at::empty({t, n * n}, phi.options());
        if (outFlag) {
            hPre = at::empty({t, n}, phi.options());
            hcBeforeNorm = at::empty({t, n * n + 2 * n}, phi.options());
            invRms = at::empty({t}, phi.options());
            sumOut = at::empty({2 * skIterCount * t * SINKHORN_ALIGN_N}, phi.options());
            normOut = at::empty({2 * skIterCount * t * n * SINKHORN_ALIGN_N}, phi.options());
        } else {
            hPre = at::empty({0}, phi.options());
            hcBeforeNorm = at::empty({0}, phi.options());
            invRms = at::empty({0}, phi.options());
            sumOut = at::empty({0}, phi.options());
            normOut = at::empty({0}, phi.options());
        }
    } else {
        int64_t b = x.size(0);
        int64_t s = x.size(1);
        int64_t c = x.size(3);
        hin = at::empty({b, s, c}, x.options());
        hPost = at::empty({b, s, n}, phi.options());
        hRes = at::empty({b, s, n * n}, phi.options());
        if (outFlag) {
            hPre = at::empty({b, s, n}, phi.options());
            hcBeforeNorm = at::empty({b, s, n * n + 2 * n}, phi.options());
            invRms = at::empty({b, s}, phi.options());
            sumOut = at::empty({2 * skIterCount * b * s * SINKHORN_ALIGN_N}, phi.options());
            normOut = at::empty({2 * skIterCount * b * s * n * SINKHORN_ALIGN_N}, phi.options());
        } else {
            hPre = at::empty({0}, phi.options());
            hcBeforeNorm = at::empty({0}, phi.options());
            invRms = at::empty({0}, phi.options());
            sumOut = at::empty({0}, phi.options());
            normOut = at::empty({0}, phi.options());
        }
    }

    c10::SmallVector<int64_t, 5> hResMatrixShape(hPost.sizes().begin(), hPost.sizes().end());
    hResMatrixShape.emplace_back(n);
    hResBeforeSinkhorn = at::empty(hResMatrixShape, phi.options());
    at::Tensor hResMatrix = at::empty_like(hResBeforeSinkhorn);

    const c10::optional<at::Tensor> gamma = c10::nullopt;
    c10::optional<at::Tensor> invRmsOptional = outFlag ? c10::optional<at::Tensor>(invRms) : c10::nullopt;
    c10::optional<at::Tensor> hcBeforeNormOptional = outFlag ? c10::optional<at::Tensor>(hcBeforeNorm) : c10::nullopt;
    c10::optional<at::Tensor> hPreOptional = outFlag ? c10::optional<at::Tensor>(hPre) : c10::nullopt;
    ACLNN_CMD(aclnnMhcPre, x, phi, alpha, bias, gamma, normEps, hcEps, hin, hPost, hResBeforeSinkhorn, invRmsOptional,
              hcBeforeNormOptional, hPreOptional);

    if (!outFlag) {
        sumOut = at::empty({1}, phi.options());
        normOut = at::empty({1}, phi.options());
    }
    ACLNN_CMD(aclnnMhcSinkhorn, hResBeforeSinkhorn, hcEps, numIters, hResMatrix, normOut, sumOut);
    hRes = hResMatrix.flatten(-2);

    if (outFlag) {
        invRms = invRms.unsqueeze(-1);
    } else {
        sumOut = at::empty({0}, phi.options());
        normOut = at::empty({0}, phi.options());
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_sinkhorn", &MhcPreSinkhorn, "mhc_pre_sinkhorn");
}

} // namespace op_api
