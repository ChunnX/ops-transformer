/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr size_t INDEX_ONE = 1;
constexpr size_t INDEX_TWO = 2;
constexpr size_t DIM_THREE = 3;

at::Tensor MoeFinalizeRouting(const at::Tensor &expandedX, const at::Tensor &expandedRowIdx,
                              const c10::optional<at::Tensor> &x1Optional, const c10::optional<at::Tensor> &x2Optional,
                              const c10::optional<at::Tensor> &biasOptional,
                              const c10::optional<at::Tensor> &scalesOptional,
                              const c10::optional<at::Tensor> &expertIdxOptional,
                              c10::optional<int64_t> dropPadMode)
{
    TORCH_CHECK(expandedX.dim() == 2 || expandedX.dim() == 3,
                "expanded_x should be 2D or 3D, but got ", expandedX.dim(), "-D.");
    TORCH_CHECK(
        expandedX.scalar_type() == at::kHalf || expandedX.scalar_type() == at::kFloat ||
            expandedX.scalar_type() == at::kBFloat16,
        "expanded_x dtype should be float16, float32 or bfloat16, but got ", expandedX.scalar_type(), ".");
    TORCH_CHECK(expandedRowIdx.dim() == 1, "expanded_row_idx should be 1D, but got ", expandedRowIdx.dim(), "-D.");
    TORCH_CHECK(expandedRowIdx.scalar_type() == at::kInt,
                "expanded_row_idx dtype should be int32, but got ", expandedRowIdx.scalar_type(), ".");
    TORCH_CHECK(!x2Optional.has_value() || x1Optional.has_value(),
                "x1 must be provided when x2 is provided.");
    TORCH_CHECK(!biasOptional.has_value() || expertIdxOptional.has_value(),
                "expert_idx must be provided when bias is provided.");
    int64_t mode = c10::value_or_else(dropPadMode, [] { return 0; });
    TORCH_CHECK(mode >= 0 && mode <= 3, "drop_pad_mode must be in [0, 3], but got ", mode, ".");

    int64_t dim0 = expandedRowIdx.size(0);
    if (scalesOptional.has_value()) {
        dim0 = scalesOptional.value().size(0);
    }
    size_t dim1Index = INDEX_ONE;
    if (expandedX.dim() == DIM_THREE) {
        dim1Index = INDEX_TWO;
    }
    at::Tensor result = at::empty({dim0, expandedX.size(dim1Index)}, expandedX.options());

    ACLNN_CMD(aclnnMoeFinalizeRoutingV2, expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
              scalesOptional, expertIdxOptional, mode, result);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_finalize_routing", &MoeFinalizeRouting, "moe_finalize_routing");
}

} // namespace op_api
