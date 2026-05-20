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

at::Tensor npu_moe_combine_prefill(const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &topkWeights,
                                   const at::Tensor &srcIdx, const at::Tensor &sendHead, const std::string &groupEp,
                                   int64_t rank, int64_t numRanks)
{
    const char *groupEpPtr = groupEp.c_str();

    TORCH_BIND_ASSERT(x.dim() == DIM_TWO and x.is_contiguous());

    at::Tensor topkIdxInt32 = topkIdx.scalar_type() == at::kInt ? topkIdx : topkIdx.to(at::kInt);
    at::Tensor tokenSrcInfo = srcIdx;
    at::Tensor epSendCounts = sendHead;
    auto device = x.device();

    at::Tensor expertScales = topkWeights.scalar_type() == at::kFloat ? topkWeights : topkWeights.to(at::kFloat);

    int64_t hidden = static_cast<int>(x.size(1));
    at::Tensor tpSendCounts = at::empty({1}, at::dtype(at::kInt).device(device));
    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t moeExpertNumber = sendHead.size(0);
    int64_t globalBs = topkIdxInt32.size(0) * numRanks;

    at::Tensor combineSendCostStatsOut;

    auto combinedX = torch::empty({expertScales.size(0), hidden}, x.options());

    ACLNN_CMD(aclnnMoeCombineNormal, x, tokenSrcInfo, epSendCounts, expertScales, tpSendCounts, groupEpPtr, numRanks,
              rank, groupEpPtr, tpWorldSize, tpRankId, moeExpertNumber, globalBs, combinedX, combineSendCostStatsOut);

    return combinedX;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_combine_prefill", &npu_moe_combine_prefill, "npu_moe_combine_prefill");
}

} // namespace op_api