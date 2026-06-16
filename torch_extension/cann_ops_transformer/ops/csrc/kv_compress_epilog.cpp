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
constexpr int64_t MIN_TAIL_DIM = 64;     // x 尾轴 d 必须大于 64
constexpr int64_t TAIL_DIM_ALIGN = 64;   // x 尾轴 d 必须 64 对齐
constexpr int64_t QUANT_MODE_MIN = 0;
constexpr int64_t QUANT_MODE_MAX = 2;    // 0:group(bf16) 1:group(e8m0) 2:rope hifloat8 + nope FLOAT4_E2M1
}  // namespace

// 在 KV Cache 的 Epilog 阶段对 cache 进行原地逐组量化压缩更新，封装 aclnnKvCompressEpilog。
// 底层 aclnn 接口对 cacheRef 原地散写；为保持函数式语义，这里对 cache 进行 clone 后在副本上
// 原地更新并返回（未被 slot_mapping 命中的行需保留原值，故以 clone 作为输出基底）。
at::Tensor npu_kv_compress_epilog(
    const at::Tensor &cache, const at::Tensor &x, const at::Tensor &slot_mapping,
    int64_t quant_group_size, int64_t quant_mode, bool round_scale, double x_scale)
{
    // 入参校验
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

    at::Tensor cache_out{nullptr};
    {
        // DeviceGuard：在申请输出之前切到输入所在设备
        auto local_device = c10::Device(cache.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        cache_out = cache.clone();
    }

    // aclnn 的 xScale 为 float，需由 double 转 float 以匹配接口原型
    float x_scale_f = static_cast<float>(x_scale);
    StorageShapeTensor x_wrapped{x};
    StorageShapeTensor slot_mapping_wrapped{slot_mapping};
    ACLNN_CMD(aclnnKvCompressEpilog, cache_out, x_wrapped, slot_mapping_wrapped,
              quant_group_size, quant_mode, round_scale, x_scale_f);
    return cache_out;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_kv_compress_epilog", &npu_kv_compress_epilog, "kv_compress_epilog");
}
}  // namespace op_api