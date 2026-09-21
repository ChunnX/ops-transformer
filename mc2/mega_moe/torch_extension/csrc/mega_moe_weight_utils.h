// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#ifndef MEGA_MOE_WEIGHT_UTILS_H
#define MEGA_MOE_WEIGHT_UTILS_H

#include "aclnn_common.h"

namespace op_api {
inline void CheckMegaMoeInt4Weights(at::TensorList weights, aclDataType dtype, const char *name)
{
    if (dtype != ACL_INT4) {
        return;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto &weight = weights[i];
        TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kInt, name, "[", i,
                    "] must be torch.int32 carrying packed INT4 data.");
        TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous(), name, "[", i,
                    "] must be a contiguous, non-transposed 2D packed INT4 weight, got shape ", weight.sizes(),
                    " and strides ", weight.strides(), ".");
        TORCH_CHECK(!IsOpInputBaseFormat(weight) && at_npu::native::get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ,
                    name, "[", i, "] must use FRACTAL_NZ for arch22 INT4 weights.");
    }
}

inline aclDataType ResolveMegaMoeWeightDtype(at::TensorList weights, c10::optional<int64_t> dtype, bool isArch22,
                                             const char *name)
{
    TORCH_CHECK(!weights.empty(), name, " must not be empty.");
    TORCH_CHECK(weights[0].defined(), name, "[0] must be defined.");
    auto aclDtype = dtype.has_value() ? GetAclDataType(dtype.value()) : ConvertToAclDataType(weights[0].scalar_type());
    if (isArch22) {
        // Preserve the legacy INT32 carrier spelling, including an explicit INT32 dtype.
        // Like arch35 FP4, only the ACL descriptor changes; the torch tensor remains packed.
        if (aclDtype == ACL_INT32) {
            aclDtype = ACL_INT4;
        }
        if (weights[0].scalar_type() == at::kInt) {
            TORCH_CHECK(aclDtype == ACL_INT4, name, " uses an INT32 carrier for INT4, but logical dtype is ",
                        static_cast<int>(aclDtype), ".");
        }
        CheckMegaMoeInt4Weights(weights, aclDtype, name);
    }
    return aclDtype;
}
} // namespace op_api

#endif // MEGA_MOE_WEIGHT_UTILS_H
