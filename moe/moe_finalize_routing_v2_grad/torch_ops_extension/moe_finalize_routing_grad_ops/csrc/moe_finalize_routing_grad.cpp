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
 * \file moe_finalize_routing_grad.cpp
 * \brief ACLNN Wrapper for aclnnMoeFinalizeRoutingV2Grad
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DROP_PAD_MODE_DROPLESS = 0;
constexpr int64_t DROP_PAD_MODE_EXPERT_PACKED = 1;

static bool HasValue(const c10::optional<at::Tensor> &opt)
{
    return opt.has_value() && opt.value().defined() && opt.value().numel() > 0;
}

// NPU device implementation
std::tuple<at::Tensor, at::Tensor> moe_finalize_routing_grad_npu(
    const at::Tensor &gradY, const at::Tensor &expandedRowIdx,
    const c10::optional<at::Tensor> &expandedXOptional, const c10::optional<at::Tensor> &scalesOptional,
    const c10::optional<at::Tensor> &expertIdxOptional, const c10::optional<at::Tensor> &biasOptional,
    int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity)
{
    TORCH_CHECK(gradY.dim() == DIM_TWO, "grad_y must be a 2D tensor, but got ", gradY.dim(), " dimensions.");
    TORCH_CHECK(expandedRowIdx.dim() == DIM_ONE, "expanded_row_idx must be a 1D tensor, but got ", expandedRowIdx.dim(),
                " dimensions.");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_EXPERT_PACKED,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");

    int64_t hidden = gradY.size(1);

    at::Tensor gradExpandedXOut;
    at::Tensor gradScalesOut;
    {
        auto localDevice = c10::Device(gradY.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        if (dropPadMode == DROP_PAD_MODE_EXPERT_PACKED) {
            gradExpandedXOut = at::empty({expertNum, expertCapacity, hidden}, gradY.options());
        } else {
            int64_t dim0 = expandedRowIdx.numel();
            if (dropPadMode == DROP_PAD_MODE_DROPLESS && activeNum > 0 && activeNum < dim0) {
                dim0 = activeNum;
            }
            gradExpandedXOut = at::empty({dim0, hidden}, gradY.options());
        }

        int64_t scalesDim1 = 1;
        at::TensorOptions scalesOpts = gradY.options();
        if (HasValue(scalesOptional)) {
            scalesDim1 = scalesOptional.value().size(1);
            scalesOpts = scalesOptional.value().options();
        }
        gradScalesOut = at::empty({gradY.size(0), scalesDim1}, scalesOpts);
    }

    ACLNN_CMD(aclnnMoeFinalizeRoutingV2Grad, gradY, expandedRowIdx, expandedXOptional, scalesOptional,
              expertIdxOptional, biasOptional, dropPadMode, activeNum, expertNum, expertCapacity, gradExpandedXOut,
              gradScalesOut);

    return std::make_tuple(std::move(gradExpandedXOut), std::move(gradScalesOut));
}

// Meta device implementation (shape inference)
std::tuple<at::Tensor, at::Tensor> moe_finalize_routing_grad_meta(
    const at::Tensor &gradY, const at::Tensor &expandedRowIdx,
    const c10::optional<at::Tensor> &expandedXOptional, const c10::optional<at::Tensor> &scalesOptional,
    const c10::optional<at::Tensor> &expertIdxOptional, const c10::optional<at::Tensor> &biasOptional,
    int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity)
{
    TORCH_CHECK(gradY.dim() == DIM_TWO, "grad_y must be a 2D tensor, but got ", gradY.dim(), " dimensions.");
    TORCH_CHECK(expandedRowIdx.dim() == DIM_ONE, "expanded_row_idx must be a 1D tensor, but got ",
                expandedRowIdx.dim(), " dimensions.");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_EXPERT_PACKED,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");

    int64_t hidden = gradY.size(1);
    c10::SmallVector<int64_t, 3> gradExpandedXShape;
    if (dropPadMode == DROP_PAD_MODE_EXPERT_PACKED) {
        gradExpandedXShape = {expertNum, expertCapacity, hidden};
    } else {
        int64_t dim0 = expandedRowIdx.numel();
        if (dropPadMode == DROP_PAD_MODE_DROPLESS && activeNum > 0 && activeNum < dim0) {
            dim0 = activeNum;
        }
        gradExpandedXShape = {dim0, hidden};
    }

    int64_t scalesDim1 = 1;
    at::ScalarType scalesDtype = gradY.scalar_type();
    if (HasValue(scalesOptional)) {
        scalesDim1 = scalesOptional.value().size(1);
        scalesDtype = scalesOptional.value().scalar_type();
    }

    at::Tensor gradExpandedX = gradY.new_empty(gradExpandedXShape);
    at::Tensor gradScales = gradY.new_empty({gradY.size(0), scalesDim1}, scalesDtype);
    return std::make_tuple(gradExpandedX, gradScales);
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("moe_finalize_routing_grad", &moe_finalize_routing_grad_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("moe_finalize_routing_grad", &moe_finalize_routing_grad_meta);
}

} // namespace op_api
