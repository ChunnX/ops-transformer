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
const uint32_t NO_SCALES = 0;
const uint32_t DYNAMIC_SCALES = 2;
const uint32_t EXPAND_IDX_COUNT_PER_GROUP = 3;
const uint32_t ZERO = 0;
const uint32_t FIRST = 1;
const uint32_t SECOND = 2;
const uint32_t THIRD = 3;
const uint32_t FOURTH = 4;
const uint32_t NUM_RANKS_MIN = 2;
const uint32_t NUM_RANKS_MAX = 384;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_moe_dispatch_prefill(const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &topkWeights,
                         const at::Tensor &numTokensPerExpert, const at::Tensor &sendTokenIdxSmall,
                         const std::string &groupEp, int64_t rank, int64_t numRanks, bool useQuant)
{
    const char *groupEpPtr = groupEp.c_str();

    at::Tensor topkIdxInt32 = topkIdx.scalar_type() == at::kInt ? topkIdx : topkIdx.to(at::kInt);
    at::Tensor expertIds = topkIdxInt32;
    int64_t tpSize = 1;
    int64_t tpRank = 0;
    int64_t quantMode = useQuant ? DYNAMIC_SCALES : NO_SCALES;

    TORCH_BIND_ASSERT(numTokensPerExpert.scalar_type() == at::kInt);
    TORCH_BIND_ASSERT(numRanks >= NUM_RANKS_MIN && numRanks <= NUM_RANKS_MAX);

    TORCH_BIND_ASSERT(x.dim() == DIM_TWO and x.is_contiguous());
    TORCH_BIND_ASSERT(numTokensPerExpert.dim() == 1 and numTokensPerExpert.is_contiguous());
    TORCH_BIND_ASSERT(numTokensPerExpert.size(0) % numRanks == 0);

    auto numTokens = static_cast<int>(x.size(0));
    auto hidden = static_cast<int>(x.size(1));
    auto numExperts = static_cast<int64_t>(numTokensPerExpert.size(0));
    auto numLocalExperts = static_cast<int>(numExperts / numRanks);

    int numTopk = static_cast<int>(topkIdxInt32.size(1));
    TORCH_BIND_ASSERT(numExperts > 0);
    TORCH_BIND_ASSERT(topkIdxInt32.dim() == DIM_TWO and topkIdxInt32.is_contiguous());
    TORCH_BIND_ASSERT(topkWeights.dim() == DIM_TWO and topkWeights.is_contiguous());
    TORCH_BIND_ASSERT(numTokens == topkIdxInt32.size(0));
    TORCH_BIND_ASSERT(numTokens == sendTokenIdxSmall.size(0));
    TORCH_BIND_ASSERT(numTopk == topkWeights.size(1));
    TORCH_BIND_ASSERT(numTopk == sendTokenIdxSmall.size(1));

    int sendPerGroup = EXPAND_IDX_COUNT_PER_GROUP;

    auto sendData = torch::empty({numExperts * sendPerGroup}, at::dtype(at::kInt).device(x.device()));
    int64_t sendCount = sendPerGroup * numLocalExperts * numRanks;

    auto sendDataOffset = torch::empty({numExperts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recvData = torch::empty({numExperts * sendPerGroup}, at::dtype(at::kInt).device(x.device()));
    at::Tensor totalRecvToken = torch::empty({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recvCount = torch::empty({numExperts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recvOffset = torch::empty({numExperts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor maxBs = torch::empty({1}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recvTokensPerExpert = torch::empty({numLocalExperts}, at::dtype(at::kLong).device(x.device()));

    int64_t localRankSize = numRanks;
    int64_t localRankId = rank % localRankSize;

    at::Tensor dispatchWaitRecvCostStatsOut;

    ACLNN_CMD(aclnnNotifyDispatch, sendData, numTokensPerExpert, sendCount, numTokens, groupEpPtr, numRanks, rank,
              localRankSize, localRankId, sendDataOffset, recvData, totalRecvToken, recvCount, recvOffset, maxBs,
              recvTokensPerExpert);

    int64_t gBs = maxBs.item<int>() * numRanks;
    int64_t trt = totalRecvToken.item<int>();
    int numRecvTokens = (trt == 0) ? 1 : trt;
    auto expandxOut = useQuant ? torch::empty({numRecvTokens, hidden}, at::dtype(at::kChar).device(x.device())) :
                                 torch::empty({numRecvTokens, hidden}, x.options());
    auto dynamicScalesOut = torch::empty({numRecvTokens}, at::dtype(at::kFloat).device(x.device()));
    auto expandIdxOut = torch::empty({numRecvTokens * 3}, at::dtype(at::kInt).device(x.device()));

    ACLNN_CMD(aclnnMoeDispatchNormal, x, expertIds, sendDataOffset, sendTokenIdxSmall, recvOffset, recvCount,
              groupEpPtr, numRanks, rank, groupEpPtr, tpSize, tpRank, numExperts, quantMode, gBs, expandxOut,
              dynamicScalesOut, expandIdxOut, dispatchWaitRecvCostStatsOut);

    return {expandxOut, dynamicScalesOut, expandIdxOut, recvCount, recvTokensPerExpert};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_dispatch_prefill", &npu_moe_dispatch_prefill, "npu_moe_dispatch_prefill");
}

} // namespace op_api