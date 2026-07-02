/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
* \file kv_compress_epilog.cpp
* \brief
*/

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

namespace {
constexpr int64_t MIN_TAIL_DIM = 64;
constexpr int64_t TAIL_DIM_ALIGN = 64;
constexpr int64_t QUANT_MODE_MIN = 0;
constexpr int64_t QUANT_MODE_MAX = 2;    // 0:group(bf16) 1:group(e8m0) 2:rope hifloat8 + nope FLOAT4_E2M1
}  // namespace


void kv_compress_epilog(
    at::Tensor &cache, const at::Tensor &x, const at::Tensor &slot_mapping,
    int64_t quant_group_size, int64_t quant_mode, bool round_scale, double x_scale)
{
    TORCH_CHECK(cache.numel() > 0, "Tensor cache is empty.");
    TORCH_CHECK(x.dim() >= 2, "x should be at least 2-dim, but got ", x.dim());
    TORCH_CHECK(slot_mapping.dim() == x.dim() - 1,
        "slot_mapping dim should equal x dim - 1, but got slot_mapping dim ", slot_mapping.dim(),
        " and x dim ", x.dim());
    TORCH_CHECK(quant_mode >= QUANT_MODE_MIN && quant_mode <= QUANT_MODE_MAX,
        "quant_mode should be 0, 1 or 2, but got ", quant_mode);
    int64_t tail_dim = x.size(x.dim() - 1);
    TORCH_CHECK(tail_dim > MIN_TAIL_DIM && tail_dim % TAIL_DIM_ALIGN == 0,
        "The last dim (d) of x should be greater than 64 and 64-aligned, but got ", tail_dim);

    auto local_device = c10::Device(cache.device());
    const c10::OptionalDeviceGuard device_guard(local_device);

    float x_scale_f = static_cast<float>(x_scale);
    StorageShapeTensor x_wrapped{x};
    StorageShapeTensor slot_mapping_wrapped{slot_mapping};
    ACLNN_CMD(aclnnKvCompressEpilog, cache, x_wrapped, slot_mapping_wrapped,
              quant_group_size, quant_mode, round_scale, x_scale_f);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kv_compress_epilog", &kv_compress_epilog, "kv_compress_epilog");
}
}  // namespace op_api