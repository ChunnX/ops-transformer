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
 * \file npu_compressor.cpp
 * \brief Compressor operator implementation for PyTorch NPU extension
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t VALUE_0 = 0;

std::vector<bool> is_contiguous_axes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();

    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguous_stride[i]);
    }
    return result;
}

at::Tensor construct_compressor_output_tensor(const at::Tensor &x, const at::Tensor &norm_weight,
                                              const at::Tensor &rope_sin, int64_t cmp_ratio, int64_t coff)
{
    auto x_dim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> cmp_kv_size;
    at::Tensor cmp_kv;
    int64_t cmp_s = 0;

    if (x_dim == DIM_THREE) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    } else {
        cmp_s = rope_sin.size(0);
        cmp_kv_size = {cmp_s, norm_weight.size(0)};
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(x.dtype()));
    return cmp_kv;
}

at::Tensor npu_compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &state_cache,
                          const at::Tensor &ape, const at::Tensor &norm_weight, const at::Tensor &rope_sin,
                          const at::Tensor &rope_cos, int64_t rope_head_dim, int64_t cmp_ratio,
                          const c10::optional<at::Tensor> &state_block_table,
                          const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                          const c10::optional<at::Tensor> &start_pos, int64_t coff, double norm_eps,
                          int64_t rotary_mode, int64_t cache_mode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_TWO || x_dim == DIM_THREE, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(norm_weight.defined(), "Check norm_weight != nullptr failed");
    auto norm_weight_dim = norm_weight.dim();
    TORCH_CHECK(norm_weight_dim == DIM_ONE, "norm_weight dim num[", norm_weight_dim, "] should be 1");

    TORCH_CHECK(rope_sin.defined(), "Check rope_sin != nullptr failed");
    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == x_dim, "rope_sin dim num[", rope_sin_dim, "] should be equal to x dim num[", x_dim,
                "]");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");

    at::Tensor cmp_kv = construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);

    auto state_cache_dim = state_cache.dim();
    TORCH_CHECK(state_cache_dim == DIM_THREE, "state_cache dim num[", state_cache_dim, "] should be 3");

    auto contiguous_axes_result = is_contiguous_axes(state_cache);
    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    ACLNN_CMD(aclnnCompressor, x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos, state_block_table,
              cu_seqlens, seqused, start_pos, rope_head_dim, cmp_ratio, coff, norm_eps, rotary_mode, cache_mode,
              state_cache_stride_dim0, cmp_kv);

    return cmp_kv;
}

at::Tensor npu_compressor_meta(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                               at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                               const at::Tensor &rope_sin, const at::Tensor &rope_cos, int64_t rope_head_dim,
                               int64_t cmp_ratio, const c10::optional<at::Tensor> &state_block_table,
                               const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                               const c10::optional<at::Tensor> &start_pos, int64_t coff, double norm_eps,
                               int64_t rotary_mode, int64_t cache_mode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_TWO || x_dim == DIM_THREE, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(norm_weight.defined(), "Check norm_weight != nullptr failed");
    auto norm_weight_dim = norm_weight.dim();
    TORCH_CHECK(norm_weight_dim == DIM_ONE, "norm_weight dim num[", norm_weight_dim, "] should be 1");

    TORCH_CHECK(rope_sin.defined(), "Check rope_sin != nullptr failed");
    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == x_dim, "rope_sin dim num[", rope_sin_dim, "] should be equal to x dim num[", x_dim,
                "]");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");

    return construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_compressor", &npu_compressor, "compressor");
}
} // namespace op_api