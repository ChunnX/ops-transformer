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
Python frontend + JIT builder for `aclnnBlockSparseAttentionV3`
(from `attention/block_sparse_attention/`).

Usage:
    import torch, torch_npu, cann_ops_transformer
    out, lse = cann_ops_transformer.ops.block_sparse_attention(
        query, key, value, block_sparse_mask, [128, 128],
        actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
        num_heads=H, num_key_value_heads=H,
        scale_value=1.0/math.sqrt(D), softmax_lse_flag=True,
    )

V3 vs V2: for FP16/BF16 (quant_mode 0) and FP8 (quant_mode 1) both aclnn entry
points reach the identical inner implementation with identical arguments, so
behavior and performance on those paths are unchanged. V3 additionally exposes
the MXFP4 quantization modes (quant_mode 2/3, Ascend 950 only) plus the
`p_quant_scale` / `dst_type_max` parameters.

Quantization modes (`quant_mode`, default 0):
    0 — none:      q/k/v float16/bfloat16, no dequant scales.
    1 — FP8:       q/k/v float8_e4m3fn + FP32 4-D q/k/v_dequant_scale.
                   (FP8 inputs with quant_mode=0 auto-upgrade to 1, matching
                   the V2-era behavior — existing callers are unaffected.)
    2 — MXFP4 OCP / 3 — MXFP4 CX (Ascend 950 only):
                   q/k/v torch.float4_e2m1fn_x2 (two FP4 values packed per
                   byte: the last dim holds D/2 packed elements) +
                   torch.float8_e8m0fnu MX block scales. Requires
                   inner_precise=4; softmax_lse is not supported. quant_mode 3
                   additionally honors dst_type_max (0.0, or [6.0, 12.0]).

Prerequisite:
    The kernel + aclnn library must already be built and installed into the
    Ascend custom-transformer vendor path:

        bash build.sh --make_clean -j96 --pkg --soc=ascend910b \\
                      --ops=block_sparse_attention
        ./build/cann-ops-transformer-custom_linux-aarch64.run

    `ACLNN_CMD` (in the C++ wrapper) dlopens `libcust_opapi.so` at runtime to
    resolve `aclnnBlockSparseAttentionV3` — no link-time dependency on the
    vendor library.
"""

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import get_as_library

# Supported layouts and dtypes (single source of truth, mirrors the kernel:
# block_sparse_attention_tiling.cpp / aclnn_block_sparse_attention_v3.cpp).
_LAYOUTS = ("TND", "BNSD", "BSND")
_FP_DTYPES = (torch.float16, torch.bfloat16)  # FP16/BF16: out dtype == in dtype
_FP8_DTYPES = (torch.float8_e4m3fn,)  # FP8: out dtype ∈ {fp16, bf16}, 950 only
# MXFP4 packed dtype (two FP4 values per byte; torch >= 2.8). None on older
# torch builds — then quant_mode 2/3 is reported as unsupported by dtype.
_FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
_MX_SCALE_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_QUANT_DEFAULT_OUT_DTYPE = torch.bfloat16

# quant_mode values (mirrors the kernel attr / aclnn V3).
QUANT_MODE_NONE = 0
QUANT_MODE_FP8 = 1
QUANT_MODE_MXFP4_OCP = 2
QUANT_MODE_MXFP4_CX = 3
_MXFP4_MODES = (QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX)


def _resolve_quant_mode(in_dtype, quant_mode):
    """Resolve/validate quant_mode against the input dtype.

    Pure (no device access) so the Meta/FakeTensor path and the eager path
    agree. FP8 inputs with quant_mode=0 auto-upgrade to 1 — that is exactly
    what aclnn V2 did internally, so V2-era callers keep working unchanged.
    The (quant_mode, dtype) pairing mirrors aclnn V3's CheckQuantModeAndDtype.
    """
    if quant_mode not in (QUANT_MODE_NONE, QUANT_MODE_FP8, *_MXFP4_MODES):
        raise ValueError(
            f"quant_mode must be 0 (none), 1 (FP8), 2 (MXFP4 OCP) or 3 (MXFP4 CX), got {quant_mode}"
        )
    if quant_mode == QUANT_MODE_NONE and in_dtype in _FP8_DTYPES:
        quant_mode = QUANT_MODE_FP8  # backward-compatible auto-upgrade
    if quant_mode == QUANT_MODE_NONE:
        if in_dtype not in _FP_DTYPES:
            raise ValueError(
                f"quant_mode 0 requires float16/bfloat16 query/key/value, got {in_dtype}"
            )
    elif quant_mode == QUANT_MODE_FP8:
        if in_dtype not in _FP8_DTYPES:
            raise ValueError(
                f"quant_mode 1 (FP8) requires float8_e4m3fn query/key/value, got {in_dtype}"
            )
    else:  # MXFP4 OCP / CX
        if _FP4_DTYPE is None or in_dtype != _FP4_DTYPE:
            raise ValueError(
                f"quant_mode {quant_mode} (MXFP4) requires torch.float4_e2m1fn_x2 "
                f"query/key/value (packed, last dim = D/2), got {in_dtype}"
            )
    return quant_mode


def _resolve_out_dtype(in_dtype, attention_out_dtype):
    """Resolve/validate the attentionOut dtype for a given input dtype.

    Pure (no device access) so the Meta/FakeTensor path and the eager path
    agree. The (in dtype → out dtype) pair is exactly what selects the kernel
    binary, so getting this right is what keeps callers off the
    "Cannot find binary for op BlockSparseAttention" cliff:
      • FP16/BF16 in       → out must equal in (default: in).
      • FP8/MXFP4 in       → out ∈ {fp16, bf16} (default: bf16); there is NO
        fp8/fp4 out binary — the kernel always dequantizes to fp16/bf16.
    """
    quantized_in = in_dtype in _FP8_DTYPES or (
        _FP4_DTYPE is not None and in_dtype == _FP4_DTYPE
    )
    if quantized_in:
        if attention_out_dtype is None:
            return _QUANT_DEFAULT_OUT_DTYPE
        if attention_out_dtype not in _FP_DTYPES:
            raise ValueError(
                f"attention_out_dtype must be torch.float16 or torch.bfloat16 for "
                f"quantized (float8_e4m3fn / float4_e2m1fn_x2) inputs, got {attention_out_dtype}"
            )
        return attention_out_dtype
    if in_dtype in _FP_DTYPES:
        if attention_out_dtype is not None and attention_out_dtype != in_dtype:
            raise ValueError(
                f"attention_out_dtype must equal the input dtype ({in_dtype}) for "
                f"float16/bfloat16 inputs, got {attention_out_dtype}"
            )
        return in_dtype
    raise ValueError(
        f"query/key/value dtype must be float16, bfloat16, float8_e4m3fn or "
        f"float4_e2m1fn_x2, got {in_dtype}"
    )


def _check_mxfp4_constraints(
    quant_mode, inner_precise, softmax_lse_flag, p_quant_scale, dst_type_max
):
    """Mirror aclnn V3's MXFP4 / additional-param constraints for early, clear
    errors on both the Meta and eager paths (aclnn_block_sparse_attention_v3.cpp:
    ValidateMxfp4Constraints / ValidateAdditionalParams)."""
    if p_quant_scale is not None:
        raise ValueError(
            "p_quant_scale must be None — the current kernel does not accept it in any quant_mode"
        )
    if quant_mode in _MXFP4_MODES:
        if inner_precise != 4:
            raise ValueError(
                f"inner_precise must be 4 for MXFP4 (quant_mode {quant_mode}), got {inner_precise}"
            )
        if softmax_lse_flag:
            raise ValueError(
                f"softmax_lse_flag is not supported for MXFP4 (quant_mode {quant_mode})"
            )
        if not (dst_type_max == 0.0 or 6.0 <= dst_type_max <= 12.0):
            raise ValueError(
                f"dst_type_max must be 0.0 or in [6.0, 12.0] for MXFP4, got {dst_type_max}"
            )
    elif dst_type_max != 0.0:
        raise ValueError(
            f"dst_type_max is only meaningful for MXFP4 (quant_mode 2/3); got "
            f"{dst_type_max} with quant_mode {quant_mode}"
        )


def _is_arch220_soc(device_name):
    """True for Atlas A2 (910B*) and A3 (910_93*) — arch v220.

    These chips only compile the FP16/BF16 TND/BNSD kernels: no FP8/MXFP4
    dequant paths and no BSND layout. Matching is positive (block only chips we
    can confidently identify as v220) so a future/unknown 950-class SoC is
    allowed through to the kernel rather than wrongly rejected here.
    """
    n = device_name.upper().replace("-", "").replace("_", "")
    return "910B" in n or "91093" in n


def _guard_soc_capabilities(query, q_input_layout, kv_input_layout):
    """Raise a clear error for FP8 / MXFP4 / BSND on a chip that cannot run them.

    Without this the op fails deep in the kernel with a cryptic
    "Cannot find bin of op BlockSparseAttention … float8_e4m3fn …" /
    "Cannot find binary for op BlockSparseAttention" at launch time, because the
    arch-v220 build never compiled those binaries.
    """
    is_quantized_dtype = query.dtype in _FP8_DTYPES or (
        _FP4_DTYPE is not None and query.dtype == _FP4_DTYPE
    )
    needs_arch35 = (
        is_quantized_dtype or q_input_layout == "BSND" or kv_input_layout == "BSND"
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
        if query.dtype in _FP8_DTYPES:
            reason.append("float8_e4m3fn inputs")
        if _FP4_DTYPE is not None and query.dtype == _FP4_DTYPE:
            reason.append("float4_e2m1fn_x2 (MXFP4) inputs")
        if "BSND" in (q_input_layout, kv_input_layout):
            reason.append("the BSND layout")
        raise RuntimeError(
            f"{' and '.join(reason)} require an Ascend 950 (arch35) chip, but the "
            f"current device is '{device_name}' (Atlas A2/A3, arch v220). "
            f"v220 supports float16/bfloat16 with the TND/BNSD layouts only. "
            f"Use float16/bfloat16 + TND/BNSD here, or run FP8/MXFP4/BSND on a 950."
        )


class BlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_sparse_attention")

    def sources(self):
        return ["ops/csrc/block_sparse_attention.cpp"]

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
            'str q_input_layout="BNSD", '
            'str kv_input_layout="BNSD", '
            "int mask_type=0, "
            "float scale_value=1.0, "
            "int inner_precise=0, "
            "int block_size=0, "
            "int pre_tokens=2147483647, "
            "int next_tokens=2147483647, "
            "bool softmax_lse_flag=False, "
            "Tensor? q_dequant_scale=None, "
            "Tensor? k_dequant_scale=None, "
            "Tensor? v_dequant_scale=None, "
            "Tensor? p_quant_scale=None, "
            "int quant_mode=0, "
            "float dst_type_max=0., "
            "ScalarType? attention_out_dtype=None"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta(
            query,
            key,
            value,
            block_sparse_mask,
            block_shape,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            num_heads,
            num_key_value_heads,
            q_input_layout,
            kv_input_layout,
            mask_type,
            scale_value,
            inner_precise,
            block_size,
            pre_tokens,
            next_tokens,
            softmax_lse_flag,
            q_dequant_scale=None,
            k_dequant_scale=None,
            v_dequant_scale=None,
            p_quant_scale=None,
            quant_mode=0,
            dst_type_max=0.0,
            attention_out_dtype=None,
        ):
            # Mirror the C++ wrapper's input constraints (block_sparse_attention.cpp)
            # so graph / FakeTensor tracing fails at the same boundary as real
            # execution instead of only erroring deep inside the kernel at run time.
            # These guard concrete (non-symbolic) arguments, so a plain raise is
            # equivalent to torch._check here and avoids accessing torch's protected
            # `_check` helper (CodeCheck protected-access / W0212). RuntimeError
            # matches what the C++ TORCH_CHECK surfaces as on the real path.
            if block_size != 0:
                raise RuntimeError(
                    "block_size must be 0 — paged KV cache is not yet supported by the kernel"
                )
            if mask_type != 0:
                raise RuntimeError(
                    "mask_type must be 0 — fine-grained attention masks are not yet supported by the kernel"
                )
            resolved_quant_mode = _resolve_quant_mode(query.dtype, quant_mode)
            _check_mxfp4_constraints(
                resolved_quant_mode,
                inner_precise,
                softmax_lse_flag,
                p_quant_scale,
                dst_type_max,
            )
            # Dequant scales are all-or-none: the quantized kernel needs q, k
            # and v scales together; the FP16/BF16 path takes none.
            n_dequant = sum(
                t is not None
                for t in (q_dequant_scale, k_dequant_scale, v_dequant_scale)
            )
            if n_dequant not in (0, 3):
                raise RuntimeError(
                    "q_dequant_scale, k_dequant_scale and v_dequant_scale must be "
                    "provided all together (quantized path) or all omitted "
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
            # attentionOut takes the resolved out dtype (fp8/fp4 in → fp16/bf16
            # out); _resolve_out_dtype enforces the rules. Shape: same as query,
            # except MXFP4 where query is PACKED (float4_e2m1fn_x2 last dim =
            # D/2) and attentionOut carries the logical head dim (2x packed).
            resolved_out_dtype = _resolve_out_dtype(query.dtype, attention_out_dtype)
            if resolved_quant_mode in _MXFP4_MODES:
                out_shape = list(query.shape)
                out_shape[-1] *= 2  # packed D/2 → logical D
                attn = query.new_empty(tuple(out_shape), dtype=resolved_out_dtype)
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


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def block_sparse_attention(
    query,
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
    scale_value=1.0,
    inner_precise=0,
    block_size=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    softmax_lse_flag=False,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    quant_mode=0,
    dst_type_max=0.0,
    attention_out_dtype=None,
):
    """Block-sparse attention with optional softmax_lse output.

    Args:
        query, key, value: same dtype, one of float16 / bfloat16 /
            float8_e4m3fn / float4_e2m1fn_x2 (MXFP4, packed: last dim = D/2).
            BNSD: [B, N, S, D]; BSND: [B, S, N, D]; TND: [T, N, D].
        block_sparse_mask: int8 [B, N, ceil(S_q/block_x), ceil(S_kv/block_y)],
            1 = block kept, 0 = block skipped.
        block_shape: [block_x, block_y], typically [128, 128].
        actual_seq_lengths, actual_seq_lengths_kv: list[int] per batch.
        num_heads, num_key_value_heads: Q heads, KV heads (GQA = same).
        q_input_layout, kv_input_layout: "TND", "BNSD" or "BSND" (must match).
            BSND requires an Ascend 950 (arch35) chip.
        mask_type: 0 = no mask (default).
        scale_value: softmax scale (typically 1/sqrt(D)).
        inner_precise: 0 = fp32 softmax (high precision), 1 = fp16, 4 = mixed.
            NOTE: Ascend 950PR/950DT only supports 4; A2/A3 only support 0/1.
            MXFP4 (quant_mode 2/3) requires 4.
        block_size: paged-attention block size. Must be 0 — kernel does not
            yet support paged cache (see block_sparse_attention_tiling.cpp:528).
        pre_tokens, next_tokens: 2147483647 = no causal window (default).
        softmax_lse_flag: if True, return a float32 LSE shaped per layout
            (TND [T,N,1]; BNSD [B,N,S,1]; BSND [B,S,N,1]); else an empty {0}.
            Not supported for MXFP4 (quant_mode 2/3).
        q_dequant_scale, k_dequant_scale, v_dequant_scale: reverse-quant factors
            for quantized inputs (aclnnBlockSparseAttentionV3). Required (all
            three) exactly when the input is quantized; must be None for
            FP16/BF16 inputs (then passed through as nullptr).
              FP8   (quant_mode 1): FP32 4-D. q [B, N, ceil(maxQ/128), 1];
                k/v [B, kvN, ceil(maxKV/{256|512}), 1].
              MXFP4 (quant_mode 2/3): float8_e8m0fnu MX block scales. BNSD:
                q [B, N, S, D/64, 2]; k [B, kvN, S, D/64, 2];
                v [B, kvN, ceil(S/64), D, 2] (BSND swaps the leading S/N dims;
                TND uses 4-D variants — see aclnnBlockSparseAttentionV3.md).
        p_quant_scale: reserved by aclnn V3; the current kernel requires it to
            be None in every quant_mode.
        quant_mode: 0 = none (default), 1 = FP8, 2 = MXFP4 OCP, 3 = MXFP4 CX.
            FP8 inputs with quant_mode=0 auto-upgrade to 1 (V2-compatible).
            Modes 1/2/3 require an Ascend 950 (arch35) chip.
        dst_type_max: MXFP4 CX custom quant range: 0.0 (use the dtype max), or
            a value in [6.0, 12.0]. Must be 0.0 for quant_mode 0/1.
        attention_out_dtype: dtype of attention_out. Default None → input dtype
            for FP16/BF16; bfloat16 for FP8/MXFP4. Quantized inputs MUST map to
            float16 or bfloat16 (there is no fp8/fp4 output kernel); FP16/BF16
            inputs must map to the same dtype as the input. For MXFP4 the
            output shape carries the logical head dim: [..., 2 * packed_D].

    Returns:
        (attention_out, softmax_lse)

    Raises:
        RuntimeError: if float8_e4m3fn / float4_e2m1fn_x2 inputs or the BSND
            layout are requested on an Atlas A2/A3 (arch v220) chip, which only
            supports float16/bfloat16 with the TND/BNSD layouts.
    """
    # Reject FP8 / MXFP4 / BSND early on chips that never compiled those
    # binaries, instead of the cryptic "Cannot find binary for op" at launch.
    _guard_soc_capabilities(query, q_input_layout, kv_input_layout)
    # Resolve here so eager execution and the registered Meta impl agree on the
    # quant mode and attentionOut dtype (and so the kernel binary can be found).
    resolved_quant_mode = _resolve_quant_mode(query.dtype, quant_mode)
    _check_mxfp4_constraints(
        resolved_quant_mode,
        inner_precise,
        softmax_lse_flag,
        p_quant_scale,
        dst_type_max,
    )
    resolved_out_dtype = _resolve_out_dtype(query.dtype, attention_out_dtype)
    # JIT-compile + load block_sparse_attention.so on first call, not at import
    # time, so `import cann_ops_transformer` stays free of this op's build cost.
    # OpBuilder.load() caches in `_loaded_ops`, so later calls are a dict lookup.
    op_module = _op_builder.load()
    return op_module.npu_block_sparse_attention(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_heads,
        num_key_value_heads,
        q_input_layout,
        kv_input_layout,
        mask_type,
        scale_value,
        inner_precise,
        block_size,
        pre_tokens,
        next_tokens,
        softmax_lse_flag,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        p_quant_scale,
        resolved_quant_mode,
        dst_type_max,
        resolved_out_dtype,
    )
