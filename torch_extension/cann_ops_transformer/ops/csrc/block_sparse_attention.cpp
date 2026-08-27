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
 * \brief Torch extension wrapper for `aclnnBlockSparseAttentionV3` (from
 *        `attention/block_sparse_attention/`). The kernel + aclnn library
 *        (`libcust_opapi.so` shipping the symbol `aclnnBlockSparseAttentionV3`)
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
 *        V3 vs V2: for FP16/BF16 (quant_mode 0) and FP8 (quant_mode 1) both
 *        entry points reach the identical inner `l0op::BlockSparseAttention`
 *        with identical arguments (V2 derives quantMode internally and passes
 *        pQuantScale=nullptr, dstTypeMax=0.0), so behavior and performance on
 *        those paths are unchanged by construction. V3 additionally exposes
 *        the MXFP4 quantization modes (quant_mode 2/3, Ascend 950 only).
 *
 *        Quantization modes (quant_mode):
 *        • 0 — no quant:  q/k/v FP16/BF16, no scales.
 *        • 1 — FP8:       q/k/v FLOAT8_E4M3FN, q/k/v_dequant_scale FP32 4-D.
 *          For backward compatibility, quant_mode=0 with FP8 inputs is
 *          auto-upgraded to 1 (mirrors aclnn V2's internal derivation).
 *        • 2 — MXFP4 OCP / 3 — MXFP4 CX: q/k/v torch.float4_e2m1fn_x2 (two
 *          FP4 values packed per byte; the tensor's last dim is D/2 packed
 *          elements — ConvertType unpacks the logical shape, see
 *          aclnn_common.h CollectB4ShapeInfo). q/k/v_dequant_scale are
 *          torch.float8_e8m0fnu MX block scales:
 *            BNSD: q [B,N,S,D/64,2]  k [B,kvN,S,D/64,2]  v [B,kvN,ceil(S/64),D,2]
 *            (BSND swaps the leading S/N dims; TND uses the 4-D variants.)
 *          Constraints (enforced by aclnn/tiling, mirrored here for clear
 *          errors): inner_precise must be 4, softmax_lse_flag must be false,
 *          and attention_out is FP16/BF16 with logical last dim 2*packed.
 *          quant_mode 3 additionally takes dst_type_max (0.0, or [6.0, 12.0]).
 *
 *        Notes:
 *        • `block_sparse_mask` must be int8 [B, H, q_blocks, kv_blocks],
 *          1 = block kept, 0 = block skipped.
 *        • `block_shape` is [block_x, block_y] in tokens (typically [128, 128]).
 *        • Layouts: q/kv_input_layout ∈ {"TND", "BNSD", "BSND"} and must match.
 *          BSND is only supported on Ascend 950 (arch35); the Python frontend
 *          guards this before reaching the kernel.
 *        • `atten_mask` and `block_table` are intentionally NOT exposed: the
 *          kernel supports neither fine-grained masks (mask_type must be 0)
 *          nor PagedAttention (block_size must be 0) on current CANN; the
 *          wrapper passes nullptr for both.
 *        • `p_quant_scale` is accepted for aclnn V3 signature parity but the
 *          current kernel requires it to be absent in every quant_mode; the
 *          wrapper rejects a non-None value up front.
 *        • `attention_out_dtype` is the dtype of `attentionOut`. quant_mode 0:
 *          must equal the input dtype. quant_mode 1/2/3: FP16 or BF16 (the
 *          kernel dequantizes; there is no fp8/fp4 output binary).
 *        • `softmaxLseOptional` shape, when enabled, is layout-dependent
 *          (the kernel's infershape hardcodes LSE_DIM_D=1):
 *            TND → [T, N, 1];  BNSD → [B, N, S, 1];  BSND → [B, S, N, 1].
 *        • `pre_tokens` / `next_tokens` should be 2147483647 (max int32 =
 *          "no causal window") — the kernel README says window restrictions
 *          are "当前不支持".
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

namespace {
// Mirrors the kernel's quantMode attr (block_sparse_attention_def.cpp /
// aclnn_block_sparse_attention_v3.cpp).
enum QuantMode : int64_t {
    NO_QUANT = 0,
    FP8_QUANT = 1,
    MXFP4_OCP_QUANT = 2,
    MXFP4_CX_QUANT = 3,
};
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_block_sparse_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &block_sparse_mask,
    at::IntArrayRef block_shape, at::IntArrayRef actual_seq_lengths, at::IntArrayRef actual_seq_lengths_kv,
    int64_t num_heads, int64_t num_key_value_heads, c10::string_view q_input_layout, c10::string_view kv_input_layout,
    int64_t mask_type, double scale_value, int64_t inner_precise, int64_t block_size, int64_t pre_tokens,
    int64_t next_tokens, bool softmax_lse_flag, const c10::optional<at::Tensor> &q_dequant_scale,
    const c10::optional<at::Tensor> &k_dequant_scale, const c10::optional<at::Tensor> &v_dequant_scale,
    const c10::optional<at::Tensor> &p_quant_scale, int64_t quant_mode, double dst_type_max,
    at::ScalarType attention_out_dtype)
{
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(block_sparse_mask.is_contiguous(), "block_sparse_mask must be contiguous");
    TORCH_CHECK(block_sparse_mask.scalar_type() == at::kChar, "block_sparse_mask must be int8 (torch.int8)");
    TORCH_CHECK(block_shape.size() == 2, "block_shape must be exactly 2 values [block_x, block_y]");
    TORCH_CHECK(q_input_layout == "BNSD" || q_input_layout == "TND" || q_input_layout == "BSND",
                "q_input_layout must be 'BNSD', 'BSND' or 'TND'");
    TORCH_CHECK(kv_input_layout == "BNSD" || kv_input_layout == "TND" || kv_input_layout == "BSND",
                "kv_input_layout must be 'BNSD', 'BSND' or 'TND'");
    TORCH_CHECK(q_input_layout == kv_input_layout,
                "q_input_layout and kv_input_layout must be identical, got q=", std::string(q_input_layout),
                ", kv=", std::string(kv_input_layout));
    TORCH_CHECK(block_size == 0, "block_size must be 0 — paged KV cache is not yet supported by the kernel");
    TORCH_CHECK(mask_type == 0,
                "mask_type must be 0 — fine-grained attention masks are not yet supported by the kernel");
    TORCH_CHECK(quant_mode >= NO_QUANT && quant_mode <= MXFP4_CX_QUANT,
                "quant_mode must be 0 (none), 1 (FP8), 2 (MXFP4 OCP) or 3 (MXFP4 CX), got ", quant_mode);

    // q/k/v must share one dtype (kernel constraint). The (quant_mode, input
    // dtype → attentionOut dtype) pairing selects the kernel binary:
    //   0: FP16→FP16, BF16→BF16;  1: FP8_E4M3FN→{FP16,BF16};
    //   2/3: FP4_E2M1(packed x2)→{FP16,BF16}.
    const at::ScalarType q_dtype = query.scalar_type();
    TORCH_CHECK(key.scalar_type() == q_dtype && value.scalar_type() == q_dtype,
                "query/key/value must share the same dtype");

    // Backward compatibility with the previous (V2-era) wrapper: FP8 inputs
    // without an explicit quant_mode auto-select the FP8 path, exactly like
    // aclnn V2 derived quantMode internally.
    int64_t effective_quant_mode = quant_mode;
    if (quant_mode == NO_QUANT && q_dtype == at::kFloat8_e4m3fn) {
        effective_quant_mode = FP8_QUANT;
    }

    // Per-mode input dtype / output dtype rules (mirrors aclnn V3's
    // CheckQuantModeAndDtype so misuse fails here with a clear message).
    const bool is_mxfp4 = (effective_quant_mode == MXFP4_OCP_QUANT || effective_quant_mode == MXFP4_CX_QUANT);
    if (effective_quant_mode == NO_QUANT) {
        TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                    "quant_mode 0 requires float16 or bfloat16 query/key/value");
        TORCH_CHECK(attention_out_dtype == q_dtype,
                    "attention_out_dtype must equal the input dtype for float16/bfloat16 inputs");
    } else if (effective_quant_mode == FP8_QUANT) {
        TORCH_CHECK(q_dtype == at::kFloat8_e4m3fn, "quant_mode 1 (FP8) requires float8_e4m3fn query/key/value");
        TORCH_CHECK(attention_out_dtype == at::kHalf || attention_out_dtype == at::kBFloat16,
                    "attention_out_dtype must be torch.float16 or torch.bfloat16 for float8_e4m3fn "
                    "inputs (the kernel dequantizes; there is no fp8 output binary)");
    } else { // MXFP4 OCP / CX
        TORCH_CHECK(q_dtype == at::ScalarType::Float4_e2m1fn_x2,
                    "quant_mode 2/3 (MXFP4) requires torch.float4_e2m1fn_x2 query/key/value "
                    "(two FP4 values packed per byte, last dim = D/2 packed elements)");
        TORCH_CHECK(attention_out_dtype == at::kHalf || attention_out_dtype == at::kBFloat16,
                    "attention_out_dtype must be torch.float16 or torch.bfloat16 for MXFP4 inputs");
        TORCH_CHECK(inner_precise == 4, "inner_precise must be 4 for MXFP4 (quant_mode 2/3), got ", inner_precise);
        TORCH_CHECK(!softmax_lse_flag, "softmax_lse_flag is not supported for MXFP4 (quant_mode 2/3)");
    }
    // dst_type_max is meaningful only for MXFP4 (custom quant range for CX):
    // 0.0, or a value in [6.0, 12.0]. Keep it 0.0 elsewhere so a stray value
    // is not silently ignored.
    if (is_mxfp4) {
        TORCH_CHECK(dst_type_max == 0.0 || (dst_type_max >= 6.0 && dst_type_max <= 12.0),
                    "dst_type_max must be 0.0 or in [6.0, 12.0] for MXFP4 (quant_mode 2/3), got ", dst_type_max);
    } else {
        TORCH_CHECK(dst_type_max == 0.0, "dst_type_max is only meaningful for MXFP4 (quant_mode 2/3); got ",
                    dst_type_max, " with quant_mode ", effective_quant_mode);
    }

    // The current kernel accepts pQuantScale in no quant_mode (aclnn V3
    // rejects a non-null value). Exposed for signature parity / future use.
    auto is_present = [](const c10::optional<at::Tensor> &t) { return t.has_value() && t->defined(); };
    TORCH_CHECK(!is_present(p_quant_scale),
                "p_quant_scale must be None — the current kernel does not accept it in any quant_mode");

    // Dequant scales: all-or-none, and present exactly when quantized.
    // Reject a partial set here rather than failing obscurely in the kernel.
    const bool q_present = is_present(q_dequant_scale);
    const bool k_present = is_present(k_dequant_scale);
    const bool v_present = is_present(v_dequant_scale);
    TORCH_CHECK(q_present == k_present && k_present == v_present,
                "q_dequant_scale, k_dequant_scale and v_dequant_scale must be provided "
                "all together (quantized path) or all omitted (FP16/BF16 path); got "
                "q=",
                q_present, ", k=", k_present, ", v=", v_present);
    if (effective_quant_mode == NO_QUANT) {
        TORCH_CHECK(!q_present, "q/k/v_dequant_scale are only valid for quantized inputs (quant_mode 1/2/3)");
    } else {
        TORCH_CHECK(q_present, is_mxfp4 ? "MXFP4 inputs require q/k/v_dequant_scale (float8_e8m0fnu MX block scales)" :
                                          "float8_e4m3fn inputs require q_dequant_scale, k_dequant_scale and "
                                          "v_dequant_scale (FP32 4-D reverse-quant factors)");
    }
    // Scale dtype per mode: FP8 → FP32; MXFP4 → FLOAT8_E8M0 (mirrors aclnn V3).
    const at::ScalarType expected_scale_dtype = is_mxfp4 ? at::ScalarType::Float8_e8m0fnu : at::kFloat;
    auto check_dequant = [&](const c10::optional<at::Tensor> &t, const char *name) {
        if (is_present(t)) {
            TORCH_CHECK(t->scalar_type() == expected_scale_dtype, name, " must be ",
                        is_mxfp4 ? "float8_e8m0fnu" : "float32", " for quant_mode ", effective_quant_mode);
            TORCH_CHECK(t->is_contiguous(), name, " must be contiguous when provided");
        }
    };
    check_dequant(q_dequant_scale, "q_dequant_scale");
    check_dequant(k_dequant_scale, "k_dequant_scale");
    check_dequant(v_dequant_scale, "v_dequant_scale");

    // Pre-allocate outputs. attentionOut keeps query's logical shape with
    // `attention_out_dtype`. For MXFP4 the input is PACKED (float4_e2m1fn_x2:
    // last dim holds D/2 bytes = D values), while attentionOut is FP16/BF16
    // with the LOGICAL head dim — so its last dim is 2x the packed input's.
    at::Tensor attention_out;
    if (is_mxfp4) {
        auto out_sizes = query.sizes().vec();
        out_sizes.back() *= 2; // packed D/2 → logical D
        attention_out = at::empty(out_sizes, query.options().dtype(attention_out_dtype));
    } else {
        attention_out = at::empty(query.sizes(), query.options().dtype(attention_out_dtype));
    }

    // LSE shape is layout-dependent (kernel infershape hardcodes LSE_DIM_D=1, see
    // attention/block_sparse_attention/op_host/block_sparse_attention_infershape.cpp):
    //   TND  → [T, N, 1]   (query is [T, N, D])
    //   BNSD → [B, N, S, 1] (query is [B, N, S, D])
    //   BSND → [B, S, N, 1] (query is [B, S, N, D])
    // BNSD and BSND both take query dims {0,1,2} + a trailing 1; TND takes {0,1} + 1.
    // When disabled (always the case for MXFP4), an empty {0} tensor is passed.
    at::Tensor softmax_lse;
    auto opts_f32 = query.options().dtype(at::kFloat);
    if (softmax_lse_flag) {
        if (q_input_layout == "TND") {
            softmax_lse = at::empty({query.size(0), query.size(1), 1}, opts_f32);
        } else { // BNSD or BSND
            softmax_lse = at::empty({query.size(0), query.size(1), query.size(2), 1}, opts_f32);
        }
    } else {
        softmax_lse = at::empty({0}, opts_f32);
    }

    // Undefined tensors → ConvertType(at::Tensor) returns nullptr
    // (see common/inc/aclnn_common.h:222). Used here for atten_mask and
    // blockTableOptional which we always pass as nullptr.
    const at::Tensor null_tensor; // default-constructed = undefined

    // Layout strings → char*. The aclnn signature takes `char *`, the generic
    // ConvertType<T> template of aclnn_common.h passes them through.
    std::string q_layout(q_input_layout);
    std::string kv_layout(kv_input_layout);
    char *q_layout_ptr = const_cast<char *>(q_layout.c_str());
    char *kv_layout_ptr = const_cast<char *>(kv_layout.c_str());

    // ACLNN_CMD → ConvertTypes(Ts &...args) takes args by non-const lvalue
    // reference. All call-site args MUST be lvalues — rvalue temporaries like
    // `static_cast<int64_t>(...)` don't bind. Materialize computed values in
    // named locals (function parameters are already lvalues).
    int64_t softmax_lse_flag_i64 = softmax_lse_flag ? 1 : 0;

    // Arg order must match aclnnBlockSparseAttentionV3GetWorkspaceSize() in
    // attention/block_sparse_attention/op_host/op_api/aclnn_block_sparse_attention_v3.h:
    // (..., vDequantScaleOptional, pQuantScaleOptional, qInputLayout, ...,
    //  softmaxLseFlag, quantMode, dstTypeMax, attentionOut, softmaxLseOptional).
    // The c10::optional scales bind to ConvertTypes(Ts&...) as const lvalues;
    // ConvertType(const c10::optional<at::Tensor>&) yields nullptr when absent.
    ACLNN_CMD(aclnnBlockSparseAttentionV3, query, key, value, block_sparse_mask,
              null_tensor, // attenMaskOptional — always nullptr
              block_shape, actual_seq_lengths, actual_seq_lengths_kv,
              null_tensor,     // blockTableOptional — always nullptr (PagedAttention unused)
              q_dequant_scale, // qDequantScaleOptional — FP8: FP32; MXFP4: E8M0; nullptr for FP16/BF16
              k_dequant_scale, // kDequantScaleOptional — idem
              v_dequant_scale, // vDequantScaleOptional — idem
              p_quant_scale,   // pQuantScaleOptional — must be nullptr on the current kernel
              q_layout_ptr, kv_layout_ptr, num_key_value_heads, mask_type, scale_value, inner_precise, block_size,
              pre_tokens, next_tokens, softmax_lse_flag_i64, effective_quant_mode, dst_type_max, attention_out,
              softmax_lse);

    return std::make_tuple(attention_out, softmax_lse);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_block_sparse_attention", &op_api::npu_block_sparse_attention, "block_sparse_attention");
}
