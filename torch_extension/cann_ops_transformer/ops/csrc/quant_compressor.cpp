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
 * \file quant_compressor.cpp
 * \brief QuantCompressor operator implementation for PyTorch NPU extension
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t VALUE_0 = 0;
const int64_t VALUE_1 = 1;

inline TensorWrapper make_wrapper(const at::Tensor &tensor, aclDataType tensor_acltype)
{
    return {tensor, tensor_acltype};
}

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

at::Tensor construct_quant_compressor_output_tensor(const at::Tensor &x, const at::Tensor &wkv,
                                                    const c10::optional<at::Tensor> &cu_seqlens,
                                                    int64_t cmp_ratio, int64_t coff)
{
    auto x_dim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> cmp_kv_size;
    at::Tensor cmp_kv;
    int64_t cmp_s = 0;

    TORCH_CHECK(wkv.defined(), "Check wkv != nullptr failed");
    auto wkv_dim = wkv.dim();
    TORCH_CHECK(wkv_dim == DIM_TWO, "wkv dim num[", wkv_dim, "] should be 2");

    auto D = wkv.size(0) / coff;
    if (x_dim == DIM_THREE) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, D};
    } else {
        TORCH_CHECK(cu_seqlens.has_value(), "Check cu_seqlens != nullptr failed");
        int64_t b_size = cu_seqlens->size(0) - 1;
        cmp_s = std::min(x.size(0), x.size(0) / cmp_ratio + b_size);
        cmp_kv_size = {cmp_s, D};
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(at::kBFloat16));
    return cmp_kv;
}

at::Tensor quant_compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                            at::Tensor &state_cache, const at::Tensor &ape,
                            int64_t quant_mode, int64_t cmp_ratio,
                            const c10::optional<at::Tensor> &x_descale,
                            const c10::optional<at::Tensor> &wkv_descale,
                            const c10::optional<at::Tensor> &wgate_descale,
                            const c10::optional<at::Tensor> &state_block_table,
                            const c10::optional<at::Tensor> &cu_seqlens,
                            const c10::optional<at::Tensor> &seqused,
                            const c10::optional<at::Tensor> &start_pos,
                            int64_t coff, int64_t cache_mode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_TWO || x_dim == DIM_THREE, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");
    TORCH_CHECK(coff > VALUE_0, "coff should be greater than 0");

    at::Tensor cmp_kv = construct_quant_compressor_output_tensor(x, wkv, cu_seqlens, cmp_ratio, coff);

    auto state_cache_dim = state_cache.dim();
    TORCH_CHECK(state_cache_dim == DIM_THREE, "state_cache dim num[", state_cache_dim, "] should be 3");

    auto contiguous_axes_result = is_contiguous_axes(state_cache);
    TORCH_CHECK(contiguous_axes_result[DIM_ONE] && contiguous_axes_result[DIM_TWO],
                "state_cache must be contiguous on all axes except axis 0");

    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    TensorWrapper x_wrapper = make_wrapper(x, ACL_HIFLOAT8);
    TensorWrapper wkv_wrapper = make_wrapper(wkv, ACL_HIFLOAT8);
    TensorWrapper wgate_wrapper = make_wrapper(wgate, ACL_HIFLOAT8);

    ACLNN_CMD(aclnnQuantCompressor, x_wrapper, wkv_wrapper, wgate_wrapper, state_cache, ape, x_descale,
              wkv_descale, wgate_descale, state_block_table, cu_seqlens, seqused, start_pos, quant_mode,
              cmp_ratio, coff, cache_mode, state_cache_stride_dim0, cmp_kv);

    return cmp_kv;
}

at::Tensor quant_compressor_meta(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                                 at::Tensor &state_cache, const at::Tensor &ape,
                                 int64_t quant_mode, int64_t cmp_ratio,
                                 const c10::optional<at::Tensor> &x_descale,
                                 const c10::optional<at::Tensor> &wkv_descale,
                                 const c10::optional<at::Tensor> &wgate_descale,
                                 const c10::optional<at::Tensor> &state_block_table,
                                 const c10::optional<at::Tensor> &cu_seqlens,
                                 const c10::optional<at::Tensor> &seqused,
                                 const c10::optional<at::Tensor> &start_pos,
                                 int64_t coff, int64_t cache_mode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_TWO || x_dim == DIM_THREE, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");
    TORCH_CHECK(coff > VALUE_0, "coff should be greater than 0");

    return construct_quant_compressor_output_tensor(x, wkv, cu_seqlens, cmp_ratio, coff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_compressor", &quant_compressor, "quant_compressor");
}
} // namespace op_api
