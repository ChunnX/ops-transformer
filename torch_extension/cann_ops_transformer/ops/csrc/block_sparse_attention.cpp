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
 * \brief Torch extension wrapper for `aclnnBlockSparseAttentionV2` (from
 *        `attention/block_sparse_attention/`). The kernel + aclnn library
 *        (`libcust_opapi.so` shipping the symbol `aclnnBlockSparseAttentionV2`)
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
 *        • Layouts: q/kv_input_layout ∈ {"TND", "BNSD", "BSND"} and must match.
 *          BSND is only supported on Ascend 950 (arch35); the Python frontend
 *          guards this before reaching the kernel.
 *        • `atten_mask` is intentionally NOT exposed: the kernel does not
 *          support fine-grained intra-block masking on CANN 8.5.0-beta.1
 *          (per the kernel's own README). The wrapper passes nullptr.
 *        • `attention_out_dtype` is the dtype of `attentionOut`. For FP16/BF16 inputs it
 *          must equal the input dtype (the only combination the kernel emits a
 *          binary for). For FLOAT8_E4M3FN inputs the kernel dequantizes and
 *          writes FP16 or BF16 — there is NO fp8→fp8 binary, so `attention_out_dtype` MUST
 *          be float16 or bfloat16 (the Python frontend defaults it to bfloat16).
 *        • `softmaxLseOptional` shape, when enabled, is layout-dependent
 *          (the kernel's infershape hardcodes LSE_DIM_D=1):
 *            TND → [T, N, 1];  BNSD → [B, N, S, 1];  BSND → [B, S, N, 1].
 *        • `q/k/v_dequant_scale` are FP8-only (FLOAT8_E4M3FN inputs) FP32 4-D
 *          reverse-quant factors. On the FP16/BF16 path they are absent and the
 *          wrapper passes nullptr (the V2 contract). See aclnnBlockSparseAttentionV2.md.
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
    bool softmax_lse_flag,
    const c10::optional<at::Tensor> &q_dequant_scale,
    const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale,
    at::ScalarType attention_out_dtype)
{
    TORCH_CHECK(query.is_contiguous(),            "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(),              "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(),            "value must be contiguous");
    TORCH_CHECK(block_sparse_mask.is_contiguous(), "block_sparse_mask must be contiguous");
    TORCH_CHECK(block_sparse_mask.scalar_type() == at::kChar,
                "block_sparse_mask must be int8 (torch.int8)");
    TORCH_CHECK(block_shape.size() == 2,
                "block_shape must be exactly 2 values [block_x, block_y]");
    TORCH_CHECK(q_input_layout == "BNSD" || q_input_layout == "TND" || q_input_layout == "BSND",
                "q_input_layout must be 'BNSD', 'BSND' or 'TND'");
    TORCH_CHECK(kv_input_layout == "BNSD" || kv_input_layout == "TND" || kv_input_layout == "BSND",
                "kv_input_layout must be 'BNSD', 'BSND' or 'TND'");
    TORCH_CHECK(q_input_layout == kv_input_layout,
                "q_input_layout and kv_input_layout must be identical, got q=",
                std::string(q_input_layout), ", kv=", std::string(kv_input_layout));
    TORCH_CHECK(block_size == 0,
                "block_size must be 0 — paged KV cache is not yet supported by the kernel");
    TORCH_CHECK(mask_type == 0,
                "mask_type must be 0 — fine-grained attention masks are not yet supported by the kernel");

    // q/k/v must share one dtype (kernel constraint) and be one of the three
    // supported input dtypes. The (input dtype → attentionOut dtype) pairing is
    // what selects the kernel binary: FP16→FP16, BF16→BF16, FP8_E4M3FN→{FP16,BF16}.
    // There is no fp8→fp8 binary, so allocating attentionOut as FP8 (the old
    // `query.options()` behaviour) made the op fail with "Cannot find binary".
    const at::ScalarType q_dtype = query.scalar_type();
    TORCH_CHECK(key.scalar_type() == q_dtype && value.scalar_type() == q_dtype,
                "query/key/value must share the same dtype");
    if (q_dtype == at::kFloat8_e4m3fn) {
        TORCH_CHECK(attention_out_dtype == at::kHalf || attention_out_dtype == at::kBFloat16,
                    "attention_out_dtype must be torch.float16 or torch.bfloat16 for float8_e4m3fn "
                    "inputs (the kernel dequantizes; there is no fp8 output binary)");
    } else {
        TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                    "query/key/value dtype must be float16, bfloat16 or float8_e4m3fn");
        TORCH_CHECK(attention_out_dtype == q_dtype,
                    "attention_out_dtype must equal the input dtype for float16/bfloat16 inputs");
    }

    // FP8-only dequant scales (aclnnBlockSparseAttentionV2). Absent on the
    // FP16/BF16 path → passed through as nullptr. When present, the kernel
    // validates the full FP8 shape rules; here we only check the basics.
    auto is_present = [](const c10::optional<at::Tensor> &t) {
        return t.has_value() && t->defined();
    };
    // All-or-none contract: the FP8 dequant path needs q, k and v scales
    // together. Reject a partial set here rather than failing obscurely
    // deep inside the kernel.
    const bool q_present = is_present(q_dequant_scale);
    const bool k_present = is_present(k_dequant_scale);
    const bool v_present = is_present(v_dequant_scale);
    TORCH_CHECK(q_present == k_present && k_present == v_present,
                "q_dequant_scale, k_dequant_scale and v_dequant_scale must be provided "
                "all together (FP8 dequant path) or all omitted (FP16/BF16 path); got "
                "q=", q_present, ", k=", k_present, ", v=", v_present);
    // The dequant path and the FP8 input dtype are two sides of the same coin:
    // FP8 inputs require the scales, and FP16/BF16 inputs must not carry them.
    if (q_dtype == at::kFloat8_e4m3fn) {
        TORCH_CHECK(q_present,
                    "float8_e4m3fn inputs require q_dequant_scale, k_dequant_scale and "
                    "v_dequant_scale (FP32 4-D reverse-quant factors)");
    } else {
        TORCH_CHECK(!q_present,
                    "q/k/v_dequant_scale are only valid for float8_e4m3fn inputs");
    }
    auto check_dequant = [&is_present](const c10::optional<at::Tensor> &t, const char *name) {
        if (is_present(t)) {
            TORCH_CHECK(t->scalar_type() == at::kFloat, name, " must be float32 when provided");
            TORCH_CHECK(t->is_contiguous(),             name, " must be contiguous when provided");
        }
    };
    check_dequant(q_dequant_scale, "q_dequant_scale");
    check_dequant(k_dequant_scale, "k_dequant_scale");
    check_dequant(v_dequant_scale, "v_dequant_scale");

    // Pre-allocate outputs. Use at::empty with explicit sizes+options instead
    // of empty_like so torch_npu doesn't attempt to inherit `query`'s memory
    // format and warn under allow_internal_format=False. attentionOut keeps
    // query's shape but takes `attention_out_dtype` — which differs from the input dtype
    // on the FP8 dequant path (fp8 in → fp16/bf16 out).
    at::Tensor attention_out = at::empty(query.sizes(), query.options().dtype(attention_out_dtype));

    // LSE shape is layout-dependent (kernel infershape hardcodes LSE_DIM_D=1, see
    // attention/block_sparse_attention/op_host/block_sparse_attention_infershape.cpp):
    //   TND  → [T, N, 1]   (query is [T, N, D])
    //   BNSD → [B, N, S, 1] (query is [B, N, S, D])
    //   BSND → [B, S, N, 1] (query is [B, S, N, D])
    // BNSD and BSND both take query dims {0,1,2} + a trailing 1; TND takes {0,1} + 1.
    // When disabled, an empty {0} tensor is passed (kernel ignores it).
    at::Tensor softmax_lse;
    auto opts_f32 = query.options().dtype(at::kFloat);
    if (softmax_lse_flag) {
        if (q_input_layout == "TND") {
            softmax_lse = at::empty({query.size(0), query.size(1), 1}, opts_f32);
        } else {  // BNSD or BSND
            softmax_lse = at::empty({query.size(0), query.size(1), query.size(2), 1}, opts_f32);
        }
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
    // aclnnBlockSparseAttentionV2GetWorkspaceSize() in
    // attention/block_sparse_attention/op_host/op_api/aclnn_block_sparse_attention_v2.h.
    // The c10::optional dequant scales bind to ConvertTypes(Ts&...) as const
    // lvalues; ConvertType(const c10::optional<at::Tensor>&) yields nullptr
    // when they are absent (the FP16/BF16 path).
    ACLNN_CMD(aclnnBlockSparseAttentionV2,
              query, key, value,
              block_sparse_mask,
              null_tensor,                    // attenMaskOptional — always nullptr
              block_shape,
              actual_seq_lengths,
              actual_seq_lengths_kv,
              null_tensor,                    // blockTableOptional — always nullptr (PagedAttention unused)
              q_dequant_scale,                // qDequantScaleOptional — FP8 only; nullptr for FP16/BF16
              k_dequant_scale,                // kDequantScaleOptional — FP8 only; nullptr for FP16/BF16
              v_dequant_scale,                // vDequantScaleOptional — FP8 only; nullptr for FP16/BF16
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
