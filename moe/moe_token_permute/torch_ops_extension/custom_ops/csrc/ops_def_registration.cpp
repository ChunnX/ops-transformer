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
    m.def("npu_moe_token_permute(Tensor tokens, Tensor indices, int? num_out_tokens=None, "
          "bool padded_mode=False, int quant_mode=-1) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
