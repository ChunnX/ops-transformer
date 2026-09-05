// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> bandwidth_test(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &dstrank_id,
    std::string group, int64_t world_size, int64_t max_bs, int64_t mode,
    int64_t ccl_buffer_size, std::string comm_alg, int64_t aiv_num, int64_t batch_tokens)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(context.device() == x.device() && dstrank_id.device() == x.device(),
                "all inputs must be on the same NPU");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kHalf, "x must be 2D float16");
    TORCH_CHECK(world_size > 0 && world_size <= 1024 && max_bs > 0 &&
                    max_bs <= INT32_MAX / world_size,
                "invalid world_size or max_bs");
    TORCH_CHECK(mode == 0 || mode == 1 || (mode == 3 && comm_alg == "urma"),
                "mode must be 0, 1 or 3; mode=3 requires comm_alg=urma");
    at::Tensor y;
    at::Tensor receive_cnt;
    {
        const c10::OptionalDeviceGuard device_guard(x.device());
        y = at::empty({max_bs * world_size, x.size(1)}, x.options());
        receive_cnt = at::empty({world_size}, x.options().dtype(at::kInt));
    }
    ACLNN_CMD(aclnnInnerBandwidthTest, context, x, dstrank_id, group, world_size,
              max_bs, mode, ccl_buffer_size, comm_alg, aiv_num, batch_tokens, y, receive_cnt);
    return {y, receive_cnt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bandwidth_test", &bandwidth_test, "bandwidth_test");
}
} // namespace op_api
