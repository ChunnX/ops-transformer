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
#include <torch/library.h>

TORCH_LIBRARY(gdn_ops_ext, m)
{
    m.def("fused_gdn_gating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, "
          "float beta=1.0, float threshold=20.0) -> (Tensor g, Tensor beta_output)");
    m.def("chunk_gated_delta_rule_compute_wy(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
          "int chunk_size) -> (Tensor q_kernel, Tensor k_kernel, Tensor w_kernel, Tensor u_kernel, "
          "Tensor g_kernel)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
