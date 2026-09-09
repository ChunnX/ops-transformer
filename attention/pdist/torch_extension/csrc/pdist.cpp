/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor pdist(const at::Tensor &x, double p)
{
    TORCH_CHECK(x.dim() == 2, "pdist: x must be 2-D [N, M], but got ", x.dim(), "-D");
    int64_t nSize = x.size(0);
    int64_t outLen = nSize * (nSize - 1) / 2;

    at::Tensor y{nullptr};
    {
        const c10::OptionalDeviceGuard device_guard(x.device());
        y = at::empty({outLen}, x.options());
    }

    ACLNN_CMD(aclnnPdist, x, p, y);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pdist", &pdist, "pdist");
}

} // namespace op_api
