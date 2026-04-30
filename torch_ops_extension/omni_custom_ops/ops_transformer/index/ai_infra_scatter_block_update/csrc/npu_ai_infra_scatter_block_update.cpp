/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
using namespace at_npu::native;

// 非原地版本：clone input后在副本上执行scatter，返回副本
at::Tensor npu_ai_infra_scatter_block_update(const at::Tensor &input, const at::Tensor &indices,
    const at::Tensor &update)
{
    at::Tensor out = input.clone();
    EXEC_NPU_CMD_V1(aclnnAiInfraScatterBlockUpdate, out, indices, update);
    return out;
}

// 原地版本：直接在input上执行scatter，返回input
void npu_ai_infra_scatter_block_update_(at::Tensor &input, const at::Tensor &indices,
    const at::Tensor &update)
{
    EXEC_NPU_CMD_V1(aclnnAiInfraScatterBlockUpdate, input, indices, update);
}

// META设备实现（非原地）：仅做shape推断，不执行计算
at::Tensor npu_ai_infra_scatter_block_update_meta(const at::Tensor &input, const at::Tensor &indices,
    const at::Tensor &update)
{
    at::Tensor out = at::empty_like(input);
    return out;
}

// META设备实现（原地）：仅做shape推断，不执行计算
void npu_ai_infra_scatter_block_update__meta(at::Tensor &input, const at::Tensor &indices,
    const at::Tensor &update)
{}
}

// 为NPU设备注册实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_ai_infra_scatter_block_update", &custom::npu_ai_infra_scatter_block_update);
    m.impl("npu_ai_infra_scatter_block_update_", &custom::npu_ai_infra_scatter_block_update_);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_ai_infra_scatter_block_update", &custom::npu_ai_infra_scatter_block_update_meta);
    m.impl("npu_ai_infra_scatter_block_update_", &custom::npu_ai_infra_scatter_block_update__meta);
}
