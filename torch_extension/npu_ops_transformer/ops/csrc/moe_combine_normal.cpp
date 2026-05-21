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
const int DIM_ONE = 1;
const int DIM_TWO = 2;

std::tuple<at::Tensor, c10::optional<at::Tensor>> npu_moe_combine_normal(
    const at::Tensor &recvX,
    const at::Tensor &tokenSrcInfo,
    const at::Tensor &epRecvCount,
    const at::Tensor &topkWeights,
    const c10::optional<at::Tensor> &XOut,
    const c10::optional<at::Tensor> &sendCostStatsOut)
{
    TORCH_CHECK((recvX.dim() == DIM_TWO), "recvX should be 2D tensor");
    TORCH_CHECK((tokenSrcInfo.dim() == DIM_TWO), "tokenSrcInfo should be 2D tensor");
    TORCH_CHECK((epRecvCount.dim() == DIM_ONE), "epRecvCount should be 1D tensor");
    TORCH_CHECK((topkWeights.dim() == DIM_TWO), "topkWeights should be 2D tensor");
    
    TORCH_CHECK((recvX.scalar_type() == at::kBFloat16) || (recvX.scalar_type() == at::kHalf),
                "dtype of recvX should be bfloat16 or float16, but got ",
                std::string(c10::toString(recvX.scalar_type())));
    TORCH_CHECK((tokenSrcInfo.scalar_type() == at::kInt),
                "dtype of tokenSrcInfo should be int32, but got ",
                std::string(c10::toString(tokenSrcInfo.scalar_type())));
    TORCH_CHECK((epRecvCount.scalar_type() == at::kInt),
                "dtype of epRecvCount should be int32, but got ",
                std::string(c10::toString(epRecvCount.scalar_type())));
    TORCH_CHECK((topkWeights.scalar_type() == at::kFloat) || (topkWeights.scalar_type() == at::kHalf) || 
                (topkWeights.scalar_type() == at::kBFloat16),
                "dtype of topkWeights should be float32, float16 or bfloat16, but got ",
                std::string(c10::toString(topkWeights.scalar_type())));

    auto recvX_size = recvX.sizes();
    auto topkWeights_size = topkWeights.sizes();
    int64_t total_tokens = recvX_size[0];
    int64_t h = recvX_size[1];
    int64_t bs = topkWeights_size[0];
    int64_t topk = topkWeights_size[1];

    TORCH_CHECK((total_tokens == bs * topk),
                "recvX first dimension should equal to bs * topk, got recvX.shape[0]=",
                total_tokens, ", bs=", bs, ", topk=", topk);

    at::Tensor x_out;
    if (XOut.has_value() && XOut.value().defined()) {
        x_out = XOut.value();
        TORCH_CHECK((x_out.size(0) == bs && x_out.size(1) == h),
                    "XOut shape should be [bs, h], got [", x_out.size(0), ", ", x_out.size(1), "]");
    } else {
        x_out = at::empty({bs, h}, recvX.options().dtype(recvX.scalar_type()));
    }

    c10::optional<at::Tensor> cost_stats_out = c10::nullopt;
    if (sendCostStatsOut.has_value() && sendCostStatsOut.value().defined()) {
        cost_stats_out = sendCostStatsOut.value();
    }

    ACLNN_CMD(aclnnMoeCombineNormal, recvX, tokenSrcInfo, epRecvCount, topkWeights, x_out, cost_stats_out);

    return std::make_tuple(x_out, cost_stats_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_combine_normal", &npu_moe_combine_normal, "npu_moe_combine_normal");
}

} // namespace op_api