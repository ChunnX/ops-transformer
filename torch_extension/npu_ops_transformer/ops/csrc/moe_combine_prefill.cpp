/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: add moe_combine_prefill pybind extention file
 * Create: 2026-01-08
 * Note:
 * History: 2026-01-08 create moe_combine_prefill pybind extention file
 */

#include <torch/extension.h>
#include "aclnn_common.h"
#include "torch_bind_exception.h"
#include <hccl/hccl.h>
#include <iostream>

using namespace at;
using namespace std;

namespace {
const uint32_t DIM_TWO = 2;
} // namespace

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;

at::Tensor npu_moe_combine_prefill(const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &topkWeights,
                                   const at::Tensor &srcIdx, const at::Tensor &sendHead, const std::string &groupEp,
                                   int64_t rank, int64_t numRanks)
{
    const char *groupEpPtr = groupEp.c_str();

    TORCH_BIND_ASSERT(x.dim() == DIM_TWO and x.is_contiguous());

    // Convert topkIdx to int32 if necessary
    at::Tensor topkIdxInt32 = topkIdx.scalar_type() == at::kInt ? topkIdx : topkIdx.to(at::kInt);
    at::Tensor tokenSrcInfo = srcIdx;
    at::Tensor epSendCounts = sendHead;
    auto device = x.device();

    // Convert topkWeights to float if necessary
    at::Tensor expertScales = topkWeights.scalar_type() == at::kFloat ? topkWeights : topkWeights.to(at::kFloat);

    int64_t hidden = static_cast<int>(x.size(1));
    at::Tensor tpSendCounts = at::empty({1}, at::dtype(at::kInt).device(device));
    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t moeExpertNumber = sendHead.size(0);
    int64_t globalBs = topkIdxInt32.size(0) * numRanks;

    // Create combineSendCostStatsOut tensor (optional output for performance monitoring)
    at::Tensor combineSendCostStatsOut;

    // Combine data
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