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
 * \file moe_finalize_routing.cpp
 * \brief ACLNN Wrapper for aclnnMoeFinalizeRoutingV2
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DROP_PAD_MODE_MIN = 0;
constexpr int64_t DROP_PAD_MODE_MAX = 3;

static bool HasValue(const c10::optional<at::Tensor> &opt)
{
    return opt.has_value() && opt.value().defined() && opt.value().numel() > 0;
}

at::Tensor MoeFinalizeRouting(const at::Tensor &expandedX, const at::Tensor &expandedRowIdx,
                              const c10::optional<at::Tensor> &x1Optional,
                              const c10::optional<at::Tensor> &x2Optional,
                              const c10::optional<at::Tensor> &biasOptional,
                              const c10::optional<at::Tensor> &scalesOptional,
                              const c10::optional<at::Tensor> &expertIdxOptional, int64_t dropPadMode)
{
    TORCH_CHECK(expandedX.dim() >= DIM_ONE, "expanded_x must be at least 1D, but got ", expandedX.dim(), " dimensions.");
    TORCH_CHECK(expandedRowIdx.dim() == DIM_ONE, "expanded_row_idx must be a 1D tensor, but got ", expandedRowIdx.dim(),
                " dimensions.");
    TORCH_CHECK(dropPadMode >= DROP_PAD_MODE_MIN && dropPadMode <= DROP_PAD_MODE_MAX,
                "drop_pad_mode must be in [0, 3], but got ", dropPadMode, ".");

    // Infer output rows: scales.shape[0] when scales provided, else expandedRowIdx.numel() (k=1).
    int64_t numRows = 0;
    if (HasValue(scalesOptional)) {
        numRows = scalesOptional.value().size(0);
    } else {
        numRows = expandedRowIdx.numel();
    }
    int64_t hidden = expandedX.size(-1);

    at::Tensor out;
    {
        auto localDevice = c10::Device(expandedX.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        out = at::empty({numRows, hidden}, expandedX.options());
    }

    ACLNN_CMD(aclnnMoeFinalizeRoutingV2, expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
              scalesOptional, expertIdxOptional, dropPadMode, out);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_finalize_routing", &MoeFinalizeRouting, "moe_finalize_routing");
}
} // namespace op_api
