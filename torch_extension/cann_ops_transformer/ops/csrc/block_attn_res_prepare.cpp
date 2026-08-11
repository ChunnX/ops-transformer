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
 * \file block_attn_res_prepare.cpp
 * \brief PyTorch binding for BlockAttnResPrepare.
 */

#include <tuple>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor> block_attn_res_prepare(const at::Tensor &blockResidual,
                                                                        const at::Tensor &effectiveQuery,
                                                                        const at::Tensor &validBlocks, double epsilon)
{
    TORCH_CHECK(blockResidual.dim() == 3, "block_residual must be a 3D tensor, but got ", blockResidual.dim(), "D");
    TORCH_CHECK(effectiveQuery.dim() == 2, "effective_query must be a 2D tensor, but got ", effectiveQuery.dim(), "D");
    TORCH_CHECK(validBlocks.dim() == 1 && validBlocks.size(0) == 1,
                "valid_blocks must have shape [1], but got ", validBlocks.sizes());
    TORCH_CHECK(blockResidual.size(2) == effectiveQuery.size(1),
                "block_residual.size(2) must equal effective_query.size(1)");

    const int64_t totalT = blockResidual.size(0);
    const int64_t totalS = effectiveQuery.size(0);
    const int64_t totalD = blockResidual.size(2);
    const auto outputOptions = blockResidual.options().dtype(at::kFloat);
    at::Tensor softmaxMax = at::empty({totalS, totalT}, outputOptions);
    at::Tensor weightedOutput = at::empty({totalS, totalT, totalD}, outputOptions);
    at::Tensor softmaxSum = at::empty({totalS, totalT}, outputOptions);

    const float epsilonValue = static_cast<float>(epsilon);
    ACLNN_CMD(aclnnBlockAttnResPrepare, blockResidual, effectiveQuery, validBlocks, epsilonValue, softmaxMax,
              weightedOutput, softmaxSum);
    return {softmaxMax, weightedOutput, softmaxSum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attn_res_prepare", &block_attn_res_prepare, "block_attn_res_prepare");
}

} // namespace op_api
