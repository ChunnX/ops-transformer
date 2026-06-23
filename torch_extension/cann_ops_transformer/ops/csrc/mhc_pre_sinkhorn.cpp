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
 * \brief ACLNN Wrapper for aclnnMhcPreSinkhorn
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mhc_pre_sinkhorn(const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha,
    const at::Tensor &bias, int64_t hcMult, int64_t numIters,
    double hcEps, double normEps, bool outFlag)
{
    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t N = x.size(2);
    int64_t C = x.size(3);

    at::Tensor hin = at::empty({B, S, C}, x.options());
    at::Tensor hPost = at::empty({B, S, N}, phi.options());
    at::Tensor hRes = at::empty({B, S, N * N}, phi.options());

    at::Tensor hPre;
    at::Tensor hcBeforeNorm;
    at::Tensor invRms;
    at::Tensor sumOut;
    at::Tensor normOut;

    int64_t skIterCount = numIters;

    if (outFlag) {
        hPre = at::empty({B, S, N}, phi.options());
        hcBeforeNorm = at::empty({B, S, N * N + 2 * N}, phi.options());
        invRms = at::empty({B, S, 1}, phi.options());
        sumOut = at::empty({2 * skIterCount, B, S, N}, phi.options());
        normOut = at::empty({2 * skIterCount, B, S, N, N}, phi.options());
    } else {
        hPre = at::empty({0}, phi.options());
        hcBeforeNorm = at::empty({0}, phi.options());
        invRms = at::empty({0}, phi.options());
        sumOut = at::empty({0}, phi.options());
        normOut = at::empty({0}, phi.options());
    }

    ACLNN_CMD(aclnnMhcPreSinkhorn, x, phi, alpha, bias, hcMult, numIters,
              hcEps, normEps, outFlag, hin, hPost, hRes, hPre,
              hcBeforeNorm, invRms, sumOut, normOut);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_sinkhorn", &mhc_pre_sinkhorn, "mhc_pre_sinkhorn");
}

} // namespace op_api