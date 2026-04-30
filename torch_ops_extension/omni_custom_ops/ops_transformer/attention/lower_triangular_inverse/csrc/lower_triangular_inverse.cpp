/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
constexpr int X_DIM_NUM = 5;

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_lower_triangular_inverse(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == X_DIM_NUM, "value dim should be 5, but actual is ", x.dim());
    at::Tensor result = at::empty_like(x);

    EXEC_NPU_CMD_V1(aclnnAiInfraLowerTriangularInverse, x, result);
    return result;
}

at::Tensor npu_lower_triangular_inverse_meta(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == X_DIM_NUM, "value dim should be 5, but actual is ", x.dim());
    at::Tensor result = at::empty_like(x);

    return result;
}
}

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_lower_triangular_inverse", &custom::npu_lower_triangular_inverse);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_lower_triangular_inverse", &custom::npu_lower_triangular_inverse_meta);
}