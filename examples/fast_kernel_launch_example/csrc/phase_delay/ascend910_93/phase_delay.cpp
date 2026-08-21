/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is part of the CANN Open Software.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <torch/all.h>
#include <torch/library.h>

#include "kernel_operator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace PhaseDelay {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("phase_delay(Tensor token, int delay_cycles) -> Tensor");
}

torch::Tensor phase_delay_meta(const torch::Tensor &token, int64_t delay_cycles)
{
    return torch::empty_like(token);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("phase_delay", phase_delay_meta);
}

__global__ __aicore__ void phase_delay_kernel(uint64_t delay_cycles)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t start = AscendC::GetSystemCycle();
    while (AscendC::GetSystemCycle() - start < delay_cycles) {
    }
}

torch::Tensor phase_delay_npu(const torch::Tensor &token, int64_t delay_cycles)
{
    TORCH_CHECK(delay_cycles >= 0, "delay_cycles must be non-negative");
    const c10::OptionalDeviceGuard guard(token.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        phase_delay_kernel<<<1, nullptr, stream>>>(static_cast<uint64_t>(delay_cycles));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("PhaseDelay", acl_call);
    return token;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("phase_delay", phase_delay_npu);
}

} // namespace PhaseDelay
} // namespace ascend_ops
