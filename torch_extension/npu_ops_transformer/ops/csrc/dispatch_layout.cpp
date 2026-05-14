// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dispatch_layout(
    const at::Tensor &topkIdx,
    int64_t num_tokens,
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_topk,
    int64_t local_ranksize)
{
    TORCH_CHECK((num_ranks > 0), "The num_ranks should be greater than 0, current is: ", num_ranks);
    TORCH_CHECK((num_experts > 0), "The num_experts should be greater than 0, current is: ", num_experts);
    TORCH_CHECK((num_topk > 0), "The num_topk should be greater than 0, current is: ", num_topk);
    TORCH_CHECK((topkIdx.dim() == DIM_TWO), "The topkIdx should be 2D");
    TORCH_CHECK((topkIdx.scalar_type() == at::kLong),
                "dtype of topkIdx should be int64.");

    at::Tensor numTokensPerRank = at::empty({num_ranks}, topkIdx.options().dtype(at::kInt));
    at::Tensor numTokensPerExpert = at::empty({num_experts}, topkIdx.options().dtype(at::kInt));
    at::Tensor isTokenInRank = at::empty({num_tokens, num_ranks}, topkIdx.options().dtype(at::kInt));
    at::Tensor notifySendData = at::empty({num_tokens, num_experts}, topkIdx.options().dtype(at::kInt));
    at::Tensor sendTokenIdxSmall = at::empty({num_tokens, num_topk}, topkIdx.options().dtype(at::kInt));

    ACLNN_CMD(aclnnDispatchLayout, topkIdx, num_tokens, num_ranks, num_experts,
              num_topk, local_ranksize, numTokensPerRank, numTokensPerExpert,
              isTokenInRank, notifySendData, sendTokenIdxSmall);

    return std::tie(numTokensPerRank, numTokensPerExpert, isTokenInRank,
                    notifySendData, sendTokenIdxSmall);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_dispatch_layout", &npu_dispatch_layout, "npu_dispatch_layout");
}

} // namespace op_api