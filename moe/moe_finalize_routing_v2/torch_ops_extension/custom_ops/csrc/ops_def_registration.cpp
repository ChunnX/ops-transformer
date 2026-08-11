/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(custom, m)
{
    m.def("npu_moe_finalize_routing(Tensor expanded_x, Tensor expanded_row_idx, "
          "Tensor? x1=None, Tensor? x2=None, Tensor? bias=None, Tensor? scales=None, "
          "Tensor? expert_idx=None, Tensor? x=None, Tensor? alpha1=None, Tensor? alpha2=None, Tensor? v=None, "
          "int? drop_pad_mode=0, "
          "int[]? zero_expert_range=None, int[]? copy_expert_range=None, "
          "int[]? constant_expert_range=None, int? k=1) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
