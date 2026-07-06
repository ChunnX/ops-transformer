# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Python frontend + JIT builder for `aclnnBlockSparseAttentionV2`
(from `attention/block_sparse_attention/`).

Usage:
	import torch, torch_npu, cann_ops_transformer
	out, lse = cann_ops_transformer.ops.block_sparse_attention(
		query, key, value, block_sparse_mask, [128, 128],
		actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
		num_heads=H, num_key_value_heads=H,
		scale_value=1.0/math.sqrt(D), softmax_lse_flag=True,
	)

`aclnnBlockSparseAttentionV2` adds `q/k/v_dequant_scale` over V1 — FP8-only
(FLOAT8_E4M3FN) FP32 reverse-quant factors. They default to None and are passed
through as nullptr on the FP16/BF16 path, so existing callers are unaffected.

Prerequisite:
	The kernel + aclnn library must already be built and installed into the
	Ascend custom-transformer vendor path:

		bash build.sh --make_clean -j96 --pkg --soc=ascend910b \\
					--ops=block_sparse_attention
		./build/cann-ops-transformer-custom_linux-aarch64.run

	`ACLNN_CMD` (in the C++ wrapper) dlopens `libcust_opapi.so` at runtime to
	resolve `aclnnBlockSparseAttentionV2` — no link-time dependency on the
	vendor library.
"""

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

# Supported layouts and dtypes (single source of truth, mirrors the kernel:
# block_sparse_attention_tiling.cpp / block_sparse_attention.cpp).
_QUANT_MODE = (0, 1, 2)
_LAYOUTS = ("TND", "BNSD", "BSND")
_FP_DTYPES = (torch.float16, torch.bfloat16)          # FP16/BF16: out dtype == in dtype
_FP8_DTYPES = (torch.float8_e4m3fn,)                  # FP8: out dtype ∈ {fp16, bf16}, 950 only
_FP8_DEFAULT_OUT_DTYPE = torch.bfloat16
_FP8_DESCALE_DTYPES = (torch.float32,)                # FP8 reverse-quant factor dtype
_FP4_DISGUISE_DTYPES = (torch.uint8,)                 # float4_e2m1fn_x2 disguised as uint8: out = bf16
_FP4_LOGICAL_DTYPES = (torch_npu.float4_e2m1fn_x2,)   # valid logical dtypes when q/k/v is uint8
_FP4_DEFAULT_OUT_DTYPE = torch.bfloat16
_FP4_DESCALE_DTYPES = (torch.float8_e8m0fnu,)     # mxfp4 descale tensor dtype


def _check_quant_mode(quant_mode):
	"""Validate quant_mode is one of the supported values.

	Called at the very top of both entry points (eager `block_sparse_attention`
	and the registered `_meta` impl) so the downstream helpers
	(`_guard_soc_capabilities`, `_validate_quant_mode`, `_resolve_out_dtype`)
	can trust the value and skip re-checking. Keeps the error message in one
	place instead of duplicating it across three helpers.
	"""
	if quant_mode not in _QUANT_MODE:
		raise ValueError(
			f"quant_mode must be one of {_QUANT_MODE} "
			f"(0=no quant, 1=FP8, 2=mxfp4), got {quant_mode}."
	)


def _resolve_out_dtype(in_dtype, attention_out_dtype, quant_mode):
	"""Resolve/validate the attentionOut dtype for a given quant_mode.

	Pure (no device access) so the Meta/FakeTensor path and the eager path
	agree. The (quant_mode → out dtype) pair is exactly what selects the kernel
	binary, so getting this right is what keeps callers off the
	"Cannot find binary for op BlockSparseAttention" cliff:
	• quant_mode=0 (FP16/BF16) → out must equal in (default: in).
	• quant_mode=1 (FP8_E4M3FN) → out ∈ {fp16, bf16} (default: bf16); there is
		NO fp8 out binary, so the old "out = in dtype" behaviour could never find
		a kernel.
	• quant_mode=2 (FP4_E2M1FN_X2, uint8-disguised) → out = bf16 (default: bf16);
		the kernel dequantizes 4-bit floats to bf16 only (no fp16 out binary).
	"""
	# Pre-condition: quant_mode is already validated by `_check_quant_mode`
	# at the entry point, so we dispatch directly.
	if quant_mode == 0:
		if in_dtype not in _FP_DTYPES:
			raise ValueError(
				f"quant_mode=0 requires query/key/value dtype to be float16 or "
				f"bfloat16, got {in_dtype}"
			)
		if attention_out_dtype is not None and attention_out_dtype != in_dtype:
			raise ValueError(
				f"attention_out_dtype must equal the input dtype ({in_dtype}) for "
				f"quant_mode=0 (float16/bfloat16) inputs, got {attention_out_dtype}"
			)
		return in_dtype
	if quant_mode == 1:
		if in_dtype not in _FP8_DTYPES:
			raise ValueError(
				f"quant_mode=1 requires query/key/value dtype to be float8_e4m3fn, "
				f"got {in_dtype}"
			)
		if attention_out_dtype is None:
			return _FP8_DEFAULT_OUT_DTYPE
		if attention_out_dtype not in _FP_DTYPES:
			raise ValueError(
				f"attention_out_dtype must be torch.float16 or torch.bfloat16 for "
				f"quant_mode=1 (float8_e4m3fn) inputs, got {attention_out_dtype}"
			)
		return attention_out_dtype
	# quant_mode == 2: mxfp4 (float4_e2m1fn_x2 disguised as uint8).
	if quant_mode == 2:
		if in_dtype not in _FP4_DISGUISE_DTYPES:
			raise ValueError(
				f"quant_mode=2 requires query/key/value dtype to be uint8 "
				f"(float4_e2m1fn_x2 disguised as uint8), got {in_dtype}"
			)
		if attention_out_dtype is None:
			return _FP4_DEFAULT_OUT_DTYPE
		if attention_out_dtype != _FP4_DEFAULT_OUT_DTYPE:
			raise ValueError(
				f"attention_out_dtype must be torch.bfloat16 for "
				f"quant_mode=2 (float4_e2m1fn_x2, uint8-disguised) inputs, "
				f"got {attention_out_dtype}"
		)
	return attention_out_dtype


def _validate_quant_mode(query, key, value, quant_mode,
						q_dequant_scale, k_dequant_scale, v_dequant_scale):
	"""Validate quant_mode against input dtypes and descale tensors.

	quant_mode semantics:
	0 = no quantization (FP16/BF16) — descale tensors must be None.
	1 = FP8 quantization (float8_e4m3fn) — q/k/v must be float8_e4m3fn and
		descale tensors (FP32 reverse-quant factors) must be provided.
	2 = mxfp4 quantization (float4_e2m1fn_x2 disguised as uint8) — q/k/v
		must be uint8 and descale tensors (dtype float8_e8m0fnu) must be
		provided.

	The logical dtypes (float4_e2m1fn_x2 / float8_e8m0fnu) are inferred from
	quant_mode inside the C++ wrapper, so the Python caller only passes this
	single integer instead of 6 separate dtype parameters.

	Pre-condition: quant_mode is already validated by `_check_quant_mode`
	at the entry point, so only the per-mode descale rules are checked here.
	"""
	if quant_mode == 1:
		# FP8: q/k/v must be float8_e4m3fn.
		if query.dtype not in _FP8_DTYPES:
			raise ValueError(
				f"quant_mode=1 (FP8) requires query.dtype to be torch.float8_e4m3fn, "
				f"got {query.dtype}."
			)
		if key.dtype not in _FP8_DTYPES:
			raise ValueError(
				f"quant_mode=1 (FP8) requires key.dtype to be torch.float8_e4m3fn, "
				f"got {key.dtype}."
			)
		if value.dtype not in _FP8_DTYPES:
			raise ValueError(
				f"quant_mode=1 (FP8) requires value.dtype to be torch.float8_e4m3fn, "
				f"got {value.dtype}."
			)
		# descale tensors are mandatory for FP8 and must be float32.
		if q_dequant_scale is None or k_dequant_scale is None or v_dequant_scale is None:
			raise ValueError(
				"q_dequant_scale/k_dequant_scale/v_dequant_scale can not be None "
				"for quant_mode=1 (FP8) inputs (FP32 reverse-quant factors)."
			)
		if q_dequant_scale.dtype not in _FP8_DESCALE_DTYPES:
			raise ValueError(
				f"q_dequant_scale.dtype must be torch.float32 for "
				f"quant_mode=1 (FP8) inputs, got {q_dequant_scale.dtype}."
			)
		if k_dequant_scale.dtype not in _FP8_DESCALE_DTYPES:
			raise ValueError(
				f"k_dequant_scale.dtype must be torch.float32 for "
				f"quant_mode=1 (FP8) inputs, got {k_dequant_scale.dtype}."
			)
		if v_dequant_scale.dtype not in _FP8_DESCALE_DTYPES:
			raise ValueError(
				f"v_dequant_scale.dtype must be torch.float32 for "
				f"quant_mode=1 (FP8) inputs, got {v_dequant_scale.dtype}."
			)
	elif quant_mode == 2:
		# mxfp4: q/k/v must be uint8 (float4_e2m1fn_x2 disguised as uint8).
		if query.dtype not in _FP4_DISGUISE_DTYPES:
			raise ValueError(
				f"quant_mode=2 (mxfp4) requires query.dtype to be torch.uint8 "
				f"(float4_e2m1fn_x2 disguised as uint8), got {query.dtype}."
			)
		if key.dtype not in _FP4_DISGUISE_DTYPES:
			raise ValueError(
				f"quant_mode=2 (mxfp4) requires key.dtype to be torch.uint8, got {key.dtype}."
			)
		if value.dtype not in _FP4_DISGUISE_DTYPES:
			raise ValueError(
				f"quant_mode=2 (mxfp4) requires value.dtype to be torch.uint8, got {value.dtype}."
			)
		# descale tensors are mandatory for mxfp4 and must be float8_e8m0fnu.
		if q_dequant_scale is None or k_dequant_scale is None or v_dequant_scale is None:
			raise ValueError(
				"q_dequant_scale/k_dequant_scale/v_dequant_scale can not be None "
				"for quant_mode=2 (mxfp4) inputs (descale tensors)."
			)
		if q_dequant_scale.dtype not in _FP4_DESCALE_DTYPES:
			raise ValueError(
				f"q_dequant_scale.dtype must be torch.float8_e8m0fnu for "
				f"quant_mode=2 (mxfp4) inputs, got {q_dequant_scale.dtype}."
			)
		if k_dequant_scale.dtype not in _FP4_DESCALE_DTYPES:
			raise ValueError(
				f"k_dequant_scale.dtype must be torch.float8_e8m0fnu for "
				f"quant_mode=2 (mxfp4) inputs, got {k_dequant_scale.dtype}."
			)
		if v_dequant_scale.dtype not in _FP4_DESCALE_DTYPES:
			raise ValueError(
				f"v_dequant_scale.dtype must be torch.float8_e8m0fnu for "
				f"quant_mode=2 (mxfp4) inputs, got {v_dequant_scale.dtype}."
			)


def _is_arch220_soc(device_name):
	"""True for Atlas A2 (910B*) and A3 (910_93*) — arch v220.

	These chips only compile the FP16/BF16 TND/BNSD kernels: no FP8 dequant
	path and no BSND layout. Matching is positive (block only chips we can
	confidently identify as v220) so a future/unknown 950-class SoC is allowed
	through to the kernel rather than wrongly rejected here.
	"""
	n = device_name.upper().replace("-", "").replace("_", "")
	return "910B" in n or "91093" in n


def _guard_soc_capabilities(query, q_input_layout, kv_input_layout, quant_mode):
	"""Raise a clear error for FP8 / mxfp4 / BSND on a chip that cannot run them.

	Without this the op fails deep in the kernel with a cryptic
	"Cannot find bin of op BlockSparseAttention … float8_e4m3fn …" /
	"Cannot find binary for op BlockSparseAttention" at launch time, because the
	arch-v220 build never compiled those binaries.
	"""
	needs_arch35 = (
		quant_mode in (1, 2)  # FP8 or mxfp4
		or q_input_layout == "BSND"
		or kv_input_layout == "BSND"
	)
	if not needs_arch35:
		return
	try:
		index = query.device.index
		if index is None:
			index = torch_npu.npu.current_device()
		device_name = torch_npu.npu.get_device_name(index)
	except Exception:
		# Can't identify the SoC — don't block; let the kernel decide.
		return
	if _is_arch220_soc(device_name):
		reason = []
		if quant_mode == 1:
			reason.append("float8_e4m3fn inputs (quant_mode=1)")
		if quant_mode == 2:
			reason.append("float4_e2m1fn_x2 (uint8-disguised) inputs (quant_mode=2)")
		if "BSND" in (q_input_layout, kv_input_layout):
			reason.append("the BSND layout")
		raise RuntimeError(
			f"{' and '.join(reason)} require an Ascend 950 (arch35) chip, but the "
			f"current device is '{device_name}' (Atlas A2/A3, arch v220). "
			f"v220 supports float16/bfloat16 with the TND/BNSD layouts only. "
			f"Use float16/bfloat16 + TND/BNSD here, or run FP8/mxfp4/BSND on a 950."
		)


class BlockSparseAttentionOpBuilder(OpBuilder):
	def __init__(self):
		super().__init__("block_sparse_attention")

	def sources(self):
		return ['ops/csrc/block_sparse_attention.cpp']

	def schema(self) -> str:
		return (
			"block_sparse_attention("
			"Tensor query, Tensor key, Tensor value, "
			"Tensor block_sparse_mask, "
			"int[] block_shape, "
			"int[] actual_seq_lengths, "
			"int[] actual_seq_lengths_kv, "
			"int num_heads, "
			"int num_key_value_heads, "
			"str q_input_layout=\"BNSD\", "
			"str kv_input_layout=\"BNSD\", "
			"int mask_type=0, "
			"int quant_mode=0, "
			"float scale_value=1.0, "
			"int inner_precise=0, "
			"int block_size=0, "
			"int pre_tokens=2147483647, "
			"int next_tokens=2147483647, "
			"bool softmax_lse_flag=False, "
			"Tensor? q_dequant_scale=None, "
			"Tensor? k_dequant_scale=None, "
			"Tensor? v_dequant_scale=None, "
			"ScalarType? attention_out_dtype=None"
			") -> (Tensor, Tensor)"
		)

	def register_meta(self):
		@impl(AS_LIBRARY, self.name, "Meta")
		def _meta(query, key, value, block_sparse_mask, block_shape,
				actual_seq_lengths, actual_seq_lengths_kv,
				num_heads, num_key_value_heads,
				q_input_layout, kv_input_layout,
				mask_type, quant_mode=0,
				scale_value=1.0, inner_precise=0, block_size=0,
				pre_tokens=2147483647, next_tokens=2147483647, softmax_lse_flag=False,
				q_dequant_scale=None, k_dequant_scale=None,
				v_dequant_scale=None, attention_out_dtype=None):
			# Mirror the C++ wrapper's input constraints (block_sparse_attention.cpp)
			# so graph / FakeTensor tracing fails at the same boundary as real
			# execution instead of only erroring deep inside the kernel at run time.
			# These guard concrete (non-symbolic) arguments, so a plain raise is
			# equivalent to torch._check here and avoids accessing torch's protected
			# `_check` helper (CodeCheck protected-access / W0212). RuntimeError
			# matches what the C++ TORCH_CHECK surfaces as on the real path.
			_check_quant_mode(quant_mode)
			if block_size != 0:
				raise RuntimeError(
					"block_size must be 0 — paged KV cache is not yet supported by the kernel"
				)
			if mask_type != 0:
				raise RuntimeError(
					"mask_type must be 0 — fine-grained attention masks are not yet supported by the kernel"
				)
			# FP8 dequant is all-or-none: the kernel needs q, k and v scales together.
			n_dequant = sum(
				t is not None
				for t in (q_dequant_scale, k_dequant_scale, v_dequant_scale)
			)
			if n_dequant not in (0, 3):
				raise RuntimeError(
					"q_dequant_scale, k_dequant_scale and v_dequant_scale must be "
					"provided all together (FP8 dequant path) or all omitted "
					"(FP16/BF16 path)"
				)
			if q_input_layout not in _LAYOUTS or kv_input_layout not in _LAYOUTS:
				raise RuntimeError(
					f"q/kv_input_layout must be one of {_LAYOUTS}, got "
					f"q={q_input_layout!r}, kv={kv_input_layout!r}"
				)
			if q_input_layout != kv_input_layout:
				raise RuntimeError(
					"q_input_layout and kv_input_layout must be identical"
				)
			# quant_mode is already validated by _check_quant_mode above;
			# _resolve_out_dtype also enforces the (quant_mode → out dtype) rules.
			# attentionOut keeps query's shape but takes the resolved out dtype
			# (fp8 in → fp16/bf16 out); _resolve_out_dtype enforces the rules.
			resolved_out_dtype = _resolve_out_dtype(query.dtype, attention_out_dtype, quant_mode)
			# FP4 (float4_e2m1fn_x2, uint8-disguised): kernel dequantizes 4-bit floats
			# to bf16 only and two fp4 values pack into one uint8 byte, so attentionOut's
			# last dim is 2*D (mirrors quant_flash_attn.cpp:83-92; qDtypeRatio=2 for FP4).
			# FP8/FP16/BF16 inputs keep query's shape. MUST match the C++ wrapper's
			# allocation in block_sparse_attention.cpp.
			if query.dtype in _FP4_DISGUISE_DTYPES:
				attn_shape = list(query.shape)
				attn_shape[-1] *= 2
				attn = query.new_empty(tuple(attn_shape), dtype=resolved_out_dtype)
			else:
				attn = query.new_empty(query.shape, dtype=resolved_out_dtype)
			if softmax_lse_flag:
				# LSE_DIM_D=1 in the kernel's infershape; shape is layout-dependent
				# (block_sparse_attention_infershape.cpp):
				#   TND  → [T, N, 1];  BNSD → [B, N, S, 1];  BSND → [B, S, N, 1].
				if q_input_layout == "TND":
					lse_shape = (query.size(0), query.size(1), 1)
				else:  # BNSD or BSND: query dims {0,1,2} + trailing 1
					lse_shape = (query.size(0), query.size(1), query.size(2), 1)
				lse = query.new_empty(lse_shape, dtype=torch.float32)
			else:
				lse = query.new_empty((0,), dtype=torch.float32)
			return attn, lse


_op_builder = BlockSparseAttentionOpBuilder()
_op_module = _op_builder.load()      # JIT-compiles + loads block_sparse_attention.so


@impl(AS_LIBRARY, _op_builder.name, "PrivateUse1")
def block_sparse_attention(query,
						key,
						value,
						block_sparse_mask,
						block_shape,
						actual_seq_lengths,
						actual_seq_lengths_kv,
						num_heads,
						num_key_value_heads,
						q_input_layout="BNSD",
						kv_input_layout="BNSD",
						mask_type=0,
						quant_mode=0,
						scale_value=1.0,
						inner_precise=0,
						block_size=0,
						pre_tokens=2147483647,
						next_tokens=2147483647,
						softmax_lse_flag=False,
						q_dequant_scale=None,
						k_dequant_scale=None,
						v_dequant_scale=None,
						attention_out_dtype=None):
	"""Block-sparse attention with optional softmax_lse output.

	Args:
		query, key, value: same dtype, one of float16 / bfloat16 / float8_e4m3fn,
			or uint8 (float4_e2m1fn_x2 disguised as uint8, requires quant_mode=2).
			BNSD: [B, N, S, D]; BSND: [B, S, N, D]; TND: [T, N, D].
		block_sparse_mask: int8 [B, N, ceil(S_q/block_x), ceil(S_kv/block_y)],
			1 = block kept, 0 = block skipped.
		block_shape: [block_x, block_y], typically [128, 128].
		actual_seq_lengths, actual_seq_lengths_kv: list[int] per batch.
		num_heads, num_key_value_heads: Q heads, KV heads (GQA = same).
		q_input_layout, kv_input_layout: "TND", "BNSD" or "BSND" (must match).
			BSND requires an Ascend 950 (arch35) chip.
		mask_type: 0 = no mask (default).
		quant_mode: quantization mode. 0 = no quantization (FP16/BF16, default);
			1 = FP8 quantization (float8_e4m3fn inputs + FP32 reverse-quant scales);
			2 = mxfp4 quantization (float4_e2m1fn_x2 disguised as uint8 + float8_e8m0fnu
			descale tensors). The C++ wrapper infers the logical dtypes
			(ACL_FLOAT4_E2M1 / ACL_FLOAT8_E8M0) from quant_mode, so the caller
			does NOT need to pass q/k/v_dtype or q/k/v_descale_dtype separately.
		scale_value: softmax scale (typically 1/sqrt(D)).
		inner_precise: 0 = fp32 softmax (high precision), 1 = fp16, 4 = mixed.
			NOTE: Ascend 950PR/950DT only supports 4; A2/A3 only support 0/1.
		block_size: paged-attention block size. Must be 0 — kernel does not
			yet support paged cache (see block_sparse_attention_tiling.cpp:528).
		pre_tokens, next_tokens: 2147483647 = no causal window (default).
		softmax_lse_flag: if True, return a float32 LSE shaped per layout
			(TND [T,N,1]; BNSD [B,N,S,1]; BSND [B,S,N,1]); else an empty {0}.
		q_dequant_scale, k_dequant_scale, v_dequant_scale: scale tensors for the
			quantized paths. For quant_mode=1 (FP8): FP32 4-D reverse-quant factors.
			For quant_mode=2 (mxfp4): descale tensors (logical dtype float8_e8m0fnu,
			inferred by the C++ wrapper). Required (all three) for quant_mode=1/2;
			must be None for quant_mode=0 (FP16/BF16). Shapes: q is
			[B, N, ceil(maxQ/128), 1]; k/v are [B, kvN, ceil(maxKV/{256|512}), 1].
		attention_out_dtype: dtype of attention_out. Default None → input dtype for
			FP16/BF16; bfloat16 for FP8/FP4. FP8 inputs MUST map to float16 or
			bfloat16 (no fp8 output kernel); FP4 (quant_mode=2) inputs MUST map to
			bfloat16 only (no fp16/fp4 output binary). FP16/BF16 inputs must map
			to the same dtype as the input.

	Returns:
		(attention_out, softmax_lse)

	Raises:
		RuntimeError: if float8_e4m3fn inputs or the BSND layout are requested on
			an Atlas A2/A3 (arch v220) chip, which only supports float16/bfloat16
			with the TND/BNSD layouts.
	"""
	# Validate quant_mode first — every downstream helper (_guard_soc_capabilities,
	# _validate_quant_mode, _resolve_out_dtype) assumes it is already in _QUANT_MODE.
	_check_quant_mode(quant_mode)
	# Reject FP8 / BSND early on chips that never compiled those binaries,
	# instead of the cryptic "Cannot find binary for op" failure at launch.
	_guard_soc_capabilities(query, q_input_layout, kv_input_layout, quant_mode)
	# Validate quant_mode against input dtypes and descale tensors.
	# No-op for quant_mode=0 (FP16/BF16). Mirrors quant_flash_attn.py:142-151.
	_validate_quant_mode(query, key, value, quant_mode,
						q_dequant_scale, k_dequant_scale, v_dequant_scale)
	# Resolve here so eager execution and the registered Meta impl agree on the
	# attentionOut dtype (and so the kernel binary can be found).
	resolved_out_dtype = _resolve_out_dtype(query.dtype, attention_out_dtype, quant_mode)
	return _op_module.npu_block_sparse_attention(
		query, key, value, block_sparse_mask,
		block_shape, actual_seq_lengths, actual_seq_lengths_kv,
		num_heads, num_key_value_heads,
		q_input_layout, kv_input_layout,
		mask_type, quant_mode,
		scale_value, inner_precise, block_size,
		pre_tokens, next_tokens, softmax_lse_flag,
		q_dequant_scale, k_dequant_scale, v_dequant_scale,
		resolved_out_dtype,
	)