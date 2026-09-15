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
#include <c10/core/DeviceGuard.h>

// Compile the production binding against the exact ACLNN argument types. Only
// the device execution boundary is replaced for this CPU-only test.
#define CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
void aclnnInnerFlashMlaMetadata(const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                                int64_t, int64_t, int64_t, int64_t, const std::string &,
                                int64_t, int64_t, int64_t, int64_t, bool, int64_t, const at::Tensor &)
{
    TORCH_CHECK(false, "Device execution is outside the CPU Torch UT");
}
#define ACLNN_CMD(name, ...) name(__VA_ARGS__)
#include "../../../torch_extension/csrc/flash_mla_metadata.cpp"
