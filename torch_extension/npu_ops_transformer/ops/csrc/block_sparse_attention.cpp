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
 * \file block_sparse_attention.cpp
 * \brief Torch extension wrapper for `aclnnBlockSparseAttention` (from
 *        `attention/block_sparse_attention/`). The kernel + aclnn library
 *        (`libcust_opapi.so` shipping the symbol `aclnnBlockSparseAttention`)
 *        must be built and installed via:
 *
 *            bash build.sh --make_clean -j96 --pkg --soc=ascend910b \
 *                          --ops=block_sparse_attention
 *            ./build/cann-ops-transformer-custom_linux-aarch64.run
 *
 *        At runtime, `ACLNN_CMD` dlopens `$ASCEND_OPP_PATH/vendors/.../op_api/lib/libcust_opapi.so`
 *        first, then falls back to `libopapi.so` — no link-time dependency on
 *        the vendor lib (see common/inc/aclnn_common.h:141).
 *
 *        Notes:
 *        • `block_sparse_mask` must be int8 [B, H, q_blocks, kv_blocks],
 *          1 = block kept, 0 = block skipped.
 *        • `block_shape` is [block_x, block_y] in tokens (typically [128, 128]).
 *        • `atten_mask` is intentionally NOT exposed: the kernel does not
 *          support fine-grained intra-block masking on CANN 8.5.0-beta.1
 *          (per the kernel's own README). The wrapper passes nullptr.
 *        • `softmaxLseOptional` shape, when enabled, is [B, H, S, 1] float32
 *          (the kernel's infershape hardcodes LSE_DIM_D=1).
 *        • `pre_tokens` / `next_tokens` should be 2147483647 (max int32 =
 *          "no causal window") — the kernel README says window restrictions
 *          are "当前不支持".
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor> npu_block_sparse_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &block_sparse_mask,
    at::IntArrayRef block_shape,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    c10::string_view q_input_layout,
    c10::string_view kv_input_layout,
    int64_t mask_type,
    double scale_value,
    int64_t inner_precise,
    int64_t block_size,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool softmax_lse_flag)
{
    TORCH_CHECK(query.is_contiguous(),            "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(),              "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(),            "value must be contiguous");
    TORCH_CHECK(block_sparse_mask.is_contiguous(), "block_sparse_mask must be contiguous");
    TORCH_CHECK(block_sparse_mask.scalar_type() == at::kChar,
                "block_sparse_mask must be int8 (torch.int8)");
    TORCH_CHECK(block_shape.size() == 2,
                "block_shape must be exactly 2 values [block_x, block_y]");
    TORCH_CHECK(q_input_layout == "BNSD" || q_input_layout == "TND",
                "q_input_layout must be 'BNSD' or 'TND'");
    TORCH_CHECK(kv_input_layout == "BNSD" || kv_input_layout == "TND",
                "kv_input_layout must be 'BNSD' or 'TND'");

    // Pre-allocate outputs. Use at::empty with explicit sizes+options instead
    // of empty_like so torch_npu doesn't attempt to inherit `query`'s memory
    // format and warn under allow_internel_format=False.
    at::Tensor attention_out = at::empty(query.sizes(), query.options());

    // LSE shape: [B, N, S, 1] float32 when enabled (kernel's infershape hardcodes
    // LSE_DIM_D=1, see attention/block_sparse_attention/op_host/block_sparse_attention_infershape.cpp).
    // When disabled, an empty {0} tensor is passed (kernel ignores it).
    at::Tensor softmax_lse;
    auto opts_f32 = query.options().dtype(at::kFloat);
    if (softmax_lse_flag) {
        TORCH_CHECK(q_input_layout == "BNSD",
                    "softmax_lse_flag=True only supported for BNSD Q layout");
        int64_t B = query.size(0);
        int64_t N = num_heads;
        int64_t S = query.size(2);
        softmax_lse = at::empty({B, N, S, 1}, opts_f32);
    } else {
        softmax_lse = at::empty({0}, opts_f32);
    }

    // Undefined tensors → ConvertType(at::Tensor) returns nullptr
    // (see common/inc/aclnn_common.h:222). Used here for atten_mask and
    // blockTableOptional which we always pass as nullptr.
    const at::Tensor null_tensor;  // default-constructed = undefined

    // Layout strings → char*. The aclnn signature takes `char *`, the generic
    // ConvertType<T> template (line 399 of aclnn_common.h) passes them through.
    std::string q_layout(q_input_layout);
    std::string kv_layout(kv_input_layout);
    char *q_layout_ptr = const_cast<char *>(q_layout.c_str());
    char *kv_layout_ptr = const_cast<char *>(kv_layout.c_str());

    // ACLNN_CMD → ConvertTypes(Ts &...args) takes args by non-const lvalue
    // reference (see common/inc/aclnn_common.h:487, :619). All call-site args
    // MUST be lvalues — rvalue temporaries like `static_cast<int64_t>(...)`
    // don't bind. Materialize the int64 LSE flag in a named local.
    int64_t softmax_lse_flag_i64 = softmax_lse_flag ? 1 : 0;

    // Single line replaces ~180 lines of manual aclTensor / workspace / sync
    // plumbing from the standalone torch_interface.cpp. Arg order must match
    // aclnnBlockSparseAttentionGetWorkspaceSize() in
    // attention/block_sparse_attention/op_host/op_api/aclnn_block_sparse_attention.h.
    ACLNN_CMD(aclnnBlockSparseAttention,
              query, key, value,
              block_sparse_mask,
              null_tensor,                    // attenMaskOptional — always nullptr
              block_shape,
              actual_seq_lengths,
              actual_seq_lengths_kv,
              null_tensor,                    // blockTableOptional — always nullptr (PagedAttention unused)
              q_layout_ptr,
              kv_layout_ptr,
              num_key_value_heads,
              mask_type,
              scale_value,
              inner_precise,
              block_size,
              pre_tokens,
              next_tokens,
              softmax_lse_flag_i64,
              attention_out,
              softmax_lse);

    return std::make_tuple(attention_out, softmax_lse);
}

}  // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_block_sparse_attention", &op_api::npu_block_sparse_attention,
          "block_sparse_attention");
}
