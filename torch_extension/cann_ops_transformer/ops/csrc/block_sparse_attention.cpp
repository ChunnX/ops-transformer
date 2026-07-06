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

// ACL dtype ids for the MXFP4 (mxfp4) path. quant_mode=2 selects these:
// q/k/v are uint8-disguised float4_e2m1fn_x2, descale tensors are
// float8_e8m0fnu. The wrapper builds TensorWrapper(t, dtype) with these
// values so the kernel sees ACL_FLOAT4_E2M1 / ACL_FLOAT8_E8M0 instead of
// ACL_UINT8. Mirrors quant_flash_attn.cpp. make_wrapper is in aclnn_common.h:246.
const int64_t FLOAT4_E2M1 = static_cast<int64_t>(at::ScalarType::Float4_e2m1fn_x2);
const int64_t FLOAT8_E8M0 = static_cast<int64_t>(at::ScalarType::Float8_e8m0fnu);

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
	int64_t quant_mode,
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

	// quant_mode selects the quantization path and the kernel binary:
	//   0 = no quantization (FP16/BF16)   → attentionOut = input dtype
	//   1 = FP8 (float8_e4m3fn)           → attentionOut ∈ {FP16, BF16} (dequant)
	//   2 = mxfp4 (float4_e2m1fn_x2, uint8-disguised) → attentionOut ∈ {BF16}
	// The (input dtype → attentionOut dtype) pairing is what selects the kernel
	// binary. There is no fp8/MXFP4 output binary, so allocating attentionOut as the
	// input dtype made the op fail with "Cannot find binary".
	const at::ScalarType q_scalar_type = query.scalar_type();
	TORCH_CHECK(key.scalar_type() == q_scalar_type && value.scalar_type() == q_scalar_type,
				"query/key/value must share the same dtype");
	if (quant_mode == 0) {
		TORCH_CHECK(q_scalar_type == at::kHalf || q_scalar_type == at::kBFloat16,
					"quant_mode=0 requires query/key/value dtype to be float16 or bfloat16, "
					"got ", q_scalar_type);
		TORCH_CHECK(attention_out_dtype == q_scalar_type,
					"attention_out_dtype must equal the input dtype for float16/bfloat16 inputs");
	} else if (quant_mode == 1) {
		TORCH_CHECK(q_scalar_type == at::kFloat8_e4m3fn,
					"quant_mode=1 requires query/key/value dtype to be float8_e4m3fn, "
					"got ", q_scalar_type);
		TORCH_CHECK(attention_out_dtype == at::kHalf || attention_out_dtype == at::kBFloat16,
					"attention_out_dtype must be torch.float16 or torch.bfloat16 for float8_e4m3fn "
					"inputs (the kernel dequantizes; there is no fp8 output binary)");
	} else {  // quant_mode == 2
		// MXFP4 path: float4_e2m1fn_x2 disguised as uint8. The wrapper infers
		// FLOAT4_E2M1 / FLOAT8_E8M0 from quant_mode (no caller-passed dtype args).
		// Kernel dequantizes to bf16 only (no fp16 output binary for mxfp4).
		TORCH_CHECK(q_scalar_type == at::kByte,
					"quant_mode=2 requires query/key/value dtype to be uint8 "
					"(float4_e2m1fn_x2 disguised as uint8), got ", q_scalar_type);
		TORCH_CHECK(attention_out_dtype == at::kBFloat16,
					"attention_out_dtype must be torch.bfloat16 for "
					"float4_e2m1fn_x2 (uint8-disguised) inputs (kernel dequantizes; "
					"there is no MXFP4 output binary and only bf16 is supported)");
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
	// Dequant scale contract — three quant_mode branches:
	//   • quant_mode=1 (FP8): q/k/v_dequant_scale are FP32 4-D reverse-quant
	//     factors. Required. Passed to ACLNN as raw at::Tensor.
	//   • quant_mode=2 (mxfp4): q/k/v_dequant_scale are descale tensors of
	//     dtype float8_e8m0fnu (validated below). Required. Wrapped via
	//     make_wrapper(..., FLOAT8_E8M0) at call time so the kernel resolves
	//     the binary by ACL_FLOAT8_E8M0.
	//   • quant_mode=0 (FP16/BF16): non-quantized path. Must NOT carry scales.
	if (quant_mode == 1) {
		TORCH_CHECK(q_present,
					"quant_mode=1 (FP8) requires q_dequant_scale, k_dequant_scale and "
					"v_dequant_scale (FP32 4-D reverse-quant factors)");
	} else if (quant_mode == 2) {
		// mxfp4 path: descale tensors mandatory.
		TORCH_CHECK(q_present,
					"quant_mode=2 (mxfp4) requires q_dequant_scale, k_dequant_scale and "
					"v_dequant_scale as descale tensors");
	} else {
		TORCH_CHECK(!q_present,
					"q/k/v_dequant_scale are only valid for quant_mode=1 (FP8) or "
					"quant_mode=2 (mxfp4) inputs");
	}
	// Contiguity applies to all paths. The FP32 dtype check is FP8-only:
	// MXFP4 descale tensors keep their physical storage dtype (logical dtype
	// is carried by the TensorWrapper), so we must NOT force them to FP32.
	auto check_dequant = [&is_present](const c10::optional<at::Tensor> &t, const char *name) {
		if (is_present(t)) {
			TORCH_CHECK(t->is_contiguous(), name, " must be contiguous when provided");
		}
	};
	check_dequant(q_dequant_scale, "q_dequant_scale");
	check_dequant(k_dequant_scale, "k_dequant_scale");
	check_dequant(v_dequant_scale, "v_dequant_scale");
	// Dtype contract per quant_mode:
	//   • quant_mode=1 (FP8): scales must be float32 (FP32 reverse-quant factors).
	//   • quant_mode=2 (mxfp4): descale tensors must be float8_e8m0fnu (the kernel
	//     resolves the binary by ACL_FLOAT8_E8M0; a wrong dtype makes it fail with
	//     "Cannot find binary"). Mirrors moe_token_permute.cpp:104-110.
	if (quant_mode == 1 && q_present) {
		TORCH_CHECK(q_dequant_scale->scalar_type() == at::kFloat,
					"q_dequant_scale must be float32 for quant_mode=1 (FP8) inputs");
		TORCH_CHECK(k_dequant_scale->scalar_type() == at::kFloat,
					"k_dequant_scale must be float32 for quant_mode=1 (FP8) inputs");
		TORCH_CHECK(v_dequant_scale->scalar_type() == at::kFloat,
					"v_dequant_scale must be float32 for quant_mode=1 (FP8) inputs");
	}
	if (quant_mode == 2 && q_present) {
		TORCH_CHECK(q_dequant_scale->scalar_type() == at::kFloat8_e8m0fnu,
					"q_dequant_scale must be float8_e8m0fnu for quant_mode=2 (mxfp4) inputs");
		TORCH_CHECK(k_dequant_scale->scalar_type() == at::kFloat8_e8m0fnu,
					"k_dequant_scale must be float8_e8m0fnu for quant_mode=2 (mxfp4) inputs");
		TORCH_CHECK(v_dequant_scale->scalar_type() == at::kFloat8_e8m0fnu,
					"v_dequant_scale must be float8_e8m0fnu for quant_mode=2 (mxfp4) inputs");
	}

	// Pre-allocate outputs. Use at::empty with explicit sizes+options instead
	// of empty_like so torch_npu doesn't attempt to inherit `query`'s memory
	// format and warn under allow_internal_format=False. attentionOut keeps
	// query's shape but takes `attention_out_dtype` — which differs from the input dtype
	// on the FP8 dequant path (fp8 in → fp16/bf16 out).
	//
	// MXFP4 (float4_e2m1fn_x2, uint8-disguised): the kernel dequantizes 4-bit
	// floats to bf16/fp16, and two MXFP4 values pack into one uint8 byte, so
	// attentionOut's last dim is 2*D (mirrors quant_flash_attn.cpp:83-92;
	// qDtypeRatio=2 for MXFP4, 1 otherwise). FP8/FP16/BF16 inputs keep
	// query's shape unchanged.
	at::Tensor attention_out;
	if (quant_mode == 2) {
		auto out_sizes = query.sizes().vec();
		out_sizes.back() *= 2;
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

	// MXFP4 path (quant_mode=2): float4_e2m1fn_x2 is disguised as uint8. The
	// kernel resolves the binary by ACL dtype, so we MUST wrap q/k/v with
	// TensorWrapper that carries ACL_FLOAT4_E2M1 (via make_wrapper in
	// aclnn_common.h:246) — passing the raw uint8 tensor would make the
	// kernel look for an ACL_UINT8 binary that doesn't exist. The FP8 /
	// FP16 / BF16 path is unchanged: it keeps passing the raw at::Tensor
	// (ConvertType has overloads for both at::Tensor and TensorWrapper, but
	// the two types cannot share one reference variable, so the call is
	// split into two branches).
	// Arg order must match aclnnBlockSparseAttentionV2GetWorkspaceSize() in
	// attention/block_sparse_attention/op_host/op_api/aclnn_block_sparse_attention_v2.h.
	// On the MXFP4 path the descale tensors are also wrapped with FLOAT8_E8M0
	// (inferred from quant_mode=2) and passed to the same
	// qDequantScaleOptional slot; on the FP8/FP16/BF16 path the raw
	// c10::optional<at::Tensor> is passed and yields nullptr when absent.
	if (quant_mode == 2) {
		// MXFP4: pass TensorWrappers so the kernel sees ACL_FLOAT4_E2M1.
		// q/k/v_dequant_scale (mandatory, validated above) are the descale
		// tensors; their logical ACL dtype is FLOAT8_E8M0 (float8_e8m0fnu,
		// inferred from quant_mode=2). Wrap them so the kernel resolves the
		// right binary — passing the raw at::Tensor would expose the
		// physical storage dtype (e.g. uint8) and fail "Cannot find binary".
		TensorWrapper q_wrapper = make_wrapper(query, FLOAT4_E2M1);
		TensorWrapper k_wrapper = make_wrapper(key,   FLOAT4_E2M1);
		TensorWrapper v_wrapper = make_wrapper(value, FLOAT4_E2M1);
		TensorWrapper q_descale_wrapper = make_wrapper(q_dequant_scale, FLOAT8_E8M0);
		TensorWrapper k_descale_wrapper = make_wrapper(k_dequant_scale, FLOAT8_E8M0);
		TensorWrapper v_descale_wrapper = make_wrapper(v_dequant_scale, FLOAT8_E8M0);
		ACLNN_CMD(aclnnBlockSparseAttentionV2,
				q_wrapper, k_wrapper, v_wrapper,
				block_sparse_mask,
				null_tensor,                    // attenMaskOptional — always nullptr
				block_shape,
				actual_seq_lengths,
				actual_seq_lengths_kv,
				null_tensor,                    // blockTableOptional — always nullptr
				q_descale_wrapper,              // MXFP4 descale (wrapped, ACL_FLOAT8_E8M0)
				k_descale_wrapper,
				v_descale_wrapper,
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
	} else {
		// FP8 / FP16 / BF16: original path, unchanged.
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
	}

	return std::make_tuple(attention_out, softmax_lse);
}

}  // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("npu_block_sparse_attention", &op_api::npu_block_sparse_attention,
		"block_sparse_attention");
}