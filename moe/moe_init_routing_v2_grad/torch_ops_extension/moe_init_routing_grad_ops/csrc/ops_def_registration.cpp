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

// 在custom命名空间里注册moe_init_routing_grad算子
// 使用TORCH_LIBRARY_FRAGMENT以支持多个包向同一命名空间注册算子
TORCH_LIBRARY_FRAGMENT(custom, m)
{
    m.def("moe_init_routing_grad(Tensor grad_expanded_x, Tensor expanded_row_idx, "
          "int top_k, int drop_pad_mode=0, int active_num=0) -> Tensor");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
