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

std::tuple<at::Tensor, at::Tensor, at::Tensor> block_attn_res_prepare(const at::Tensor &blockRes,
                                                                      const at::Tensor &validBlocks,
                                                                      const at::Tensor &pseudoQuery, double eps)
{
    TORCH_CHECK(blockRes.dim() == 3, "block_res must be a 3D tensor, but got ", blockRes.dim(), "D");
    TORCH_CHECK(validBlocks.dim() == 1 && validBlocks.size(0) == 1,
                "valid_blocks must have shape [1], but got ", validBlocks.sizes());
    TORCH_CHECK(pseudoQuery.dim() == 2, "pseudo_query must be a 2D tensor, but got ", pseudoQuery.dim(), "D");
    TORCH_CHECK(blockRes.size(1) >= 1 && blockRes.size(1) <= 64,
                "block_res.size(1) must be in [1, 64], but got ", blockRes.size(1));
    TORCH_CHECK(blockRes.size(2) >= 1 && blockRes.size(2) <= 8192,
                "block_res.size(2) must be in [1, 8192], but got ", blockRes.size(2));
    TORCH_CHECK(blockRes.size(2) == pseudoQuery.size(1),
                "block_res.size(2) must equal pseudo_query.size(1)");

    const int64_t totalT = blockRes.size(0);
    const int64_t totalS = pseudoQuery.size(0);
    const int64_t totalD = blockRes.size(2);
    const auto outputOptions = blockRes.options().dtype(at::kFloat);
    at::Tensor numerator = at::empty({totalS, totalT, totalD}, outputOptions);
    at::Tensor logitMax = at::empty({totalS, totalT}, outputOptions);
    at::Tensor expSum = at::empty({totalS, totalT}, outputOptions);

    const float epsValue = static_cast<float>(eps);
    ACLNN_CMD(aclnnBlockAttnResPrepare, blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, epsValue);
    return {numerator, logitMax, expSum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attn_res_prepare", &block_attn_res_prepare, "block_attn_res_prepare");
}

} // namespace op_api
