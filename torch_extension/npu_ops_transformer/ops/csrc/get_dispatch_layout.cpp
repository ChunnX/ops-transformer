// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include "aclnn_common.h"
#include "torch_bind_exception.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;

const uint32_t DIM_TWO = 2;
const uint32_t ZERO = 0;
const uint32_t FIRST = 1;
const uint32_t LOCAL_RANK_SIZE = 8;
const int MAX_BATCH_SIZE = 4096;
const int EXPERT_DATA_SIZE = 1 + MAX_BATCH_SIZE;

std::tuple<at::Tensor, at::Tensor> npu_get_dispatch_layout(const at::Tensor &topkIdx, int64_t numExperts,
                                                           int64_t numRanks)
{
    at::Tensor topkIdxInt64 = topkIdx.scalar_type() == at::kLong ? topkIdx : topkIdx.to(at::kLong);

    TORCH_BIND_ASSERT(topkIdxInt64.dim() == DIM_TWO);
    TORCH_BIND_ASSERT(topkIdxInt64.is_contiguous());
    TORCH_BIND_ASSERT(numExperts > 0);

    const int numTokens = topkIdxInt64.size(0);
    const int numTopk = topkIdxInt64.size(1);
    const int localRanksize = LOCAL_RANK_SIZE;
    auto serverNum = numRanks / localRanksize;

    auto device = topkIdxInt64.device();
    auto numTokensPerExpert = at::zeros({numExperts}, at::dtype(at::kInt).device(device));
    auto numTokensPerRank = at::zeros({numRanks}, at::dtype(at::kInt).device(device));
    auto isTokenInRank = at::zeros({numTokens, numRanks}, at::dtype(at::kInt).device(device));
    const int notifySendDataSize =
        numExperts * EXPERT_DATA_SIZE + serverNum + MAX_BATCH_SIZE * (1 + 2 * serverNum + numExperts);
    auto sendTokenIdxSmall = at::zeros({numTokens, numTopk}, at::dtype(at::kInt).device(device));
    auto notifySendData = at::zeros({notifySendDataSize}, at::dtype(at::kInt).device(device));

    ACLNN_CMD(aclnnDispatchLayout, topkIdxInt64, numTokens, numRanks, numExperts, numTopk, localRanksize,
              numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall);

    return std::make_tuple(numTokensPerExpert, sendTokenIdxSmall);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_get_dispatch_layout", &npu_get_dispatch_layout, "npu_get_dispatch_layout");
}

} // namespace op_api