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

std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>> npu_moe_dispatch_normal(
    const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &sendOffset, const at::Tensor &sendTokenIdx,
    const at::Tensor &recvOffset, const at::Tensor &recvCount, const std::string &groupEp, int64_t epWorldSize,
    int64_t epRankId, const c10::optional<std::string> &groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t moeExpertNum, int64_t quantMode, int64_t globalBs, const c10::optional<at::Tensor> &recvX,
    const c10::optional<at::Tensor> &recvXScales, const c10::optional<at::Tensor> &assistInfoForCombine,
    const c10::optional<at::Tensor> &waitRecvCostStats)
{
    TORCH_CHECK((x.dim() == DIM_TWO), "x should be 2D tensor");
    TORCH_CHECK((topkIdx.dim() == DIM_TWO), "topkIdx should be 2D tensor");
    TORCH_CHECK((sendOffset.dim() == DIM_ONE), "sendOffset should be 1D tensor");
    TORCH_CHECK((sendTokenIdx.dim() == DIM_ONE), "sendTokenIdx should be 1D tensor");
    TORCH_CHECK((recvOffset.dim() == DIM_ONE), "recvOffset should be 1D tensor");
    TORCH_CHECK((recvCount.dim() == DIM_ONE), "recvCount should be 1D tensor");

    TORCH_CHECK((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf),
                "dtype of x should be bfloat16 or float16, but got ", std::string(c10::toString(x.scalar_type())));
    TORCH_CHECK((topkIdx.scalar_type() == at::kInt), "dtype of topkIdx should be int32, but got ",
                std::string(c10::toString(topkIdx.scalar_type())));
    TORCH_CHECK((sendOffset.scalar_type() == at::kInt), "dtype of sendOffset should be int32, but got ",
                std::string(c10::toString(sendOffset.scalar_type())));
    TORCH_CHECK((sendTokenIdx.scalar_type() == at::kInt), "dtype of sendTokenIdx should be int32, but got ",
                std::string(c10::toString(sendTokenIdx.scalar_type())));
    TORCH_CHECK((recvOffset.scalar_type() == at::kInt), "dtype of recvOffset should be int32, but got ",
                std::string(c10::toString(recvOffset.scalar_type())));
    TORCH_CHECK((recvCount.scalar_type() == at::kInt), "dtype of recvCount should be int32, but got ",
                std::string(c10::toString(recvCount.scalar_type())));

    auto x_size = x.sizes();
    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t total_recv_tokens = recvOffset.size(0);

    at::Tensor recv_x;
    if (recvX.has_value() && recvX.value().defined()) {
        recv_x = recvX.value();
        TORCH_CHECK((recv_x.size(0) == total_recv_tokens && recv_x.size(1) == h),
                    "recvX shape should be [total_recv_tokens, h], got [", recv_x.size(0), ", ", recv_x.size(1), "]");
    } else {
        recv_x = at::empty({total_recv_tokens, h}, x.options().dtype(x.scalar_type()));
    }

    at::Tensor recv_x_scales;
    if (recvXScales.has_value() && recvXScales.value().defined()) {
        recv_x_scales = recvXScales.value();
    } else {
        recv_x_scales = at::empty({total_recv_tokens}, x.options().dtype(at::kFloat));
    }

    at::Tensor assist_info;
    if (assistInfoForCombine.has_value() && assistInfoForCombine.value().defined()) {
        assist_info = assistInfoForCombine.value();
    } else {
        assist_info = at::empty({total_recv_tokens, 3}, x.options().dtype(at::kInt));
    }

    c10::optional<at::Tensor> cost_stats = c10::nullopt;
    if (waitRecvCostStats.has_value() && waitRecvCostStats.value().defined()) {
        cost_stats = waitRecvCostStats.value();
    }

    std::string group_tp_str = "";
    if (groupTp.has_value()) {
        group_tp_str = groupTp.value();
    }

    ACLNN_CMD(aclnnMoeDispatchNormal, x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount, groupEp.c_str(),
              epWorldSize, epRankId, group_tp_str.c_str(), tpWorldSize, tpRankId, moeExpertNum, quantMode, globalBs,
              recv_x, recv_x_scales, assist_info, cost_stats);

    return std::make_tuple(recv_x, recv_x_scales, assist_info, cost_stats);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_dispatch_normal", &npu_moe_dispatch_normal, "npu_moe_dispatch_normal");
}

} // namespace op_api