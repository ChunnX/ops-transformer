// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

/*!
 * \file moe_init_routing_grad.cpp
 * \brief ACLNN Wrapper for aclnnMoeInitRoutingV2Grad
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DROP_PAD_MODE_DROPLESS = 0;
constexpr int64_t DROP_PAD_MODE_DROP_PAD = 1;

// NPU device implementation
at::Tensor moe_init_routing_grad_npu(const at::Tensor &gradExpandedX, const at::Tensor &expandedRowIdx,
                                     int64_t topK, int64_t dropPadMode, int64_t activeNum)
{
    TORCH_CHECK(expandedRowIdx.dim() == 1, "expanded_row_idx must be a 1D tensor, but got ", expandedRowIdx.dim(),
                " dimensions.");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_DROP_PAD,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");
    TORCH_CHECK(topK > 0, "top_k must be greater than 0, but got ", topK, ".");

    int64_t gradXDim0 = expandedRowIdx.numel() / topK;
    int64_t gradXDim1 = (dropPadMode == DROP_PAD_MODE_DROP_PAD) ? gradExpandedX.size(2) : gradExpandedX.size(1);

    at::Tensor out;
    {
        auto localDevice = c10::Device(gradExpandedX.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        out = at::empty({gradXDim0, gradXDim1}, gradExpandedX.options());
    }

    ACLNN_CMD(aclnnMoeInitRoutingV2Grad, gradExpandedX, expandedRowIdx, topK, dropPadMode, activeNum, out);

    return out;
}

// Meta device implementation (shape inference)
at::Tensor moe_init_routing_grad_meta(const at::Tensor &gradExpandedX, const at::Tensor &expandedRowIdx,
                                      int64_t topK, int64_t dropPadMode, int64_t activeNum)
{
    TORCH_CHECK(expandedRowIdx.dim() == 1, "expanded_row_idx must be a 1D tensor, but got ", expandedRowIdx.dim(),
                " dimensions.");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_DROP_PAD,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");
    TORCH_CHECK(topK > 0, "top_k must be greater than 0, but got ", topK, ".");
    int64_t gradXDim0 = expandedRowIdx.numel() / topK;
    int64_t gradXDim1 = (dropPadMode == DROP_PAD_MODE_DROP_PAD) ? gradExpandedX.size(2) : gradExpandedX.size(1);
    return gradExpandedX.new_empty({gradXDim0, gradXDim1});
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("moe_init_routing_grad", &moe_init_routing_grad_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("moe_init_routing_grad", &moe_init_routing_grad_meta);
}

} // namespace op_api
