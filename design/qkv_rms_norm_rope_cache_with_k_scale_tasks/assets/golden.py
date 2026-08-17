"""Numpy golden for QkvRmsNormRopeCacheWithKScale.

This is a faithful reference implementation of the computation chain defined in
``03_golden_design.md`` (and section 8 of the main design document):

    qkv -> split Q/K/V
      Q: RMSNorm -> RoPE(q_rope_out) -> (@ rotation, dense Cube matmul)
         -> dynamic FP8 quant -> q_out / q_scale
      K: RMSNorm -> RoPE(k_rope_out) -> (@ rotation, dense Cube matmul)
         -> dynamic FP8 quant -> k_cache / k_scale_cache (scatter by slot_mapping)
      V: v * v_scale -> static FP8 quant -> v_cache (scatter by slot_mapping)

Design-faithful points:
  * Nq/Nk/Nv come from ``head_nums`` (NOT inferred from cache shapes).
  * ``layout_qkv`` controls the qkv input shape, and ``layout_q_out`` controls
    the q_out/q_scale writeback shape; internal Q/K/V math stays token-major
    [T, H, D].
  * RoPE position s(t) is derived from the ``query_start_loc`` local prefix sum
    and per-batch ``seq_lens - local_len`` history offset. Inputs where
    ``seq_lens[b]`` is smaller than the current batch token count are
    outside the valid contract.
  * Q and K use the same half-split RoPE equations, materialized as
    ``q_rope_out`` and ``k_rope_out`` so they are not confused with the
    external ``rotation`` Cube right matrix.
  * ``rotation`` is an external dense [D, D] matrix shared by Q/K; no
    Hadamard assumption is made.
  * RMSNorm / RoPE run in FP32; the RoPE result is cast to BF16 before the Cube
    matmul, which accumulates in FP32 (bf16 x bf16 -> fp32).
  * Q/K dynamic quant matches the arch35 ``dynamic_quant`` FP8 E4M3FN path:
        scale = max_abs / 448 ; y = cast_e4m3fn_sat_rint(C / scale)
    No special ``scale == 0`` / ``max_abs == 0`` branch is added; that
    abnormal-value policy is owned by ``dynamic_quant_e4m3fn`` (an all-zero
    row yields ``scale == 0`` and NaN output, matching the no-guard kernel).
  * ``v_scale`` is a multiplier (not a divisor); ``v_scale == 0`` is legal.
  * Cache tensors are input/output aliased: golden starts from the initial cache
    contents and only overwrites the touched slots.

BF16 and FP8-E4M3FN casts use ``ml_dtypes``. ``cast_to_e4m3fn_sat_rint``
returns an array with dtype ``ml_dtypes.float8_e4m3fn``. Use
``e4m3fn_to_uint8`` only when a bit-level storage-byte comparison is needed.

FP8 E4M3FN: 1 sign / 4 exp (bias 7) / 3 mantissa, no inf, max finite = 448,
the all-ones exp+mantissa pattern (0x7F / 0xFF) is NaN.
"""

import numpy as np

try:
    import ml_dtypes
except ImportError as exc:
    raise ImportError(
        "golden.py requires ml_dtypes for bfloat16 and float8_e4m3fn support"
    ) from exc

FP8_E4M3FN_MAX = 448.0
BF16_DTYPE = np.dtype(ml_dtypes.bfloat16)
FP8_E4M3FN_DTYPE = np.dtype(ml_dtypes.float8_e4m3fn)


# ---------------------------------------------------------------------------
# BF16 cast (round to nearest even)
# ---------------------------------------------------------------------------
def to_bf16(x):
    """Round float -> bfloat16 (RNE), returned as float32 holding bf16 values."""
    return np.asarray(x, dtype=np.float32).astype(BF16_DTYPE).astype(np.float32)


# ---------------------------------------------------------------------------
# FP8 E4M3FN cast (RINT via ml_dtypes, SAT to max finite 448)
# ---------------------------------------------------------------------------
def cast_to_e4m3fn_sat_rint(x):
    """Quantize to dtype float8_e4m3fn with saturation to finite max."""
    arr = np.asarray(x)
    if arr.dtype == FP8_E4M3FN_DTYPE:
        return arr.copy()
    arr_f32 = arr.astype(np.float32, copy=False)
    clipped = np.clip(arr_f32, -FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
    return clipped.astype(FP8_E4M3FN_DTYPE)


# ---------------------------------------------------------------------------
# Dynamic FP8 quant (Q/K), arch35 dynamic_quant E4M3FN path
# ---------------------------------------------------------------------------
def dynamic_quant_e4m3fn(x, axis=-1):
    """Per-row dynamic FP8 E4M3FN quant: scale = max_abs / 448; y = cast(x/scale).

    Returns ``(y_fp8, scale)`` where ``scale`` keeps FP32 precision.

    Abnormal-value policy (owned here, not in the test harness):
      * Aligned to the arch35 ``dynamic_quant`` E4M3FN path: NO special branch
        for ``max_abs == 0`` / ``scale == 0``. The kernel performs a plain
        ``Div -> Cast`` with no zero-row guard, so an all-zero row yields
        ``scale == 0`` and ``x / scale == 0 / 0 == NaN`` for every element of
        that row. This golden reproduces that NaN rather than masking it, so
        UT/ST can lock the real hardware/toolchain behavior.
      * Adding a zero-row guard (e.g. emit all-zero or skip) would be an
        explicit semantic deviation from dynamic_quant and must be decided
        separately and reflected here before the golden changes.
    """
    max_abs = np.max(np.abs(x), axis=axis)                   # [...]
    scale = (max_abs / np.float32(FP8_E4M3FN_MAX)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x / np.expand_dims(scale, axis)                  # 0/0 -> NaN by design
    return cast_to_e4m3fn_sat_rint(y), scale


def e4m3fn_to_uint8(x):
    """Return FP8-E4M3FN storage bytes.

    Input may already be ``float8_e4m3fn`` or a numeric array that should first
    be cast through ``cast_to_e4m3fn_sat_rint``. If input is ``uint8``, it is
    treated as already encoded storage bytes.
    """
    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        return arr.copy()
    fp8 = arr if arr.dtype == FP8_E4M3FN_DTYPE else cast_to_e4m3fn_sat_rint(arr)
    return fp8.view(np.uint8).copy()


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------
def apply_half_split_rope(y, cos_sin, query_start_loc, seq_lens):
    """Apply full-D half-split RoPE and return the per-row RoPE tensor.

    The helper returns ``q_rope_out`` / ``k_rope_out`` rows before the dense
    Cube matmul with the external shared ``rotation`` matrix. For
    each token ``t`` the RoPE position is:

        local_len(b) = query_start_loc[b + 1] - query_start_loc[b]
        local_pos(t) = t - query_start_loc[b(t)]
        history_offset(b) = seq_lens[b] - local_len(b)
        s(t) = history_offset(b(t)) + local_pos(t)

        low'  = low  * cos - high * sin
        high' = high * cos + low  * sin
    """
    y = np.asarray(y, dtype=np.float32)
    D = int(y.shape[-1])
    assert D % 2 == 0, "D must be even (RoPE half-split)"
    half = D // 2

    T = int(y.shape[0])
    cu = np.asarray(query_start_loc).astype(np.int64)
    full_seq = np.asarray(seq_lens).astype(np.int64)
    assert cu.ndim == 1 and full_seq.ndim == 1, "query_start_loc and seq_lens must be 1-D"
    assert len(cu) == len(full_seq) + 1, "seq_lens length must equal Batch"

    b_of_t = np.searchsorted(cu, np.arange(T, dtype=np.int64), side="right") - 1
    local_len = cu[b_of_t + 1] - cu[b_of_t]
    local_pos = np.arange(T, dtype=np.int64) - cu[b_of_t]
    history_offset = full_seq[b_of_t] - local_len
    s_of_t = history_offset + local_pos

    cos_sin_f = np.asarray(cos_sin, dtype=np.float32)
    assert np.all(history_offset >= 0), (
        "seq_lens must be >= current batch token counts")
    assert np.all(s_of_t >= 0), "RoPE position must be non-negative"
    assert np.all(s_of_t < cos_sin_f.shape[0]), "cos_sin must cover all RoPE positions"
    cos = cos_sin_f[s_of_t, 0:half]                          # [T, half]
    sin = cos_sin_f[s_of_t, half:D]                          # [T, half]

    y_low = y[:, :, 0:half]
    y_high = y[:, :, half:D]
    out_low = y_low * cos[:, None, :] - y_high * sin[:, None, :]
    out_high = y_high * cos[:, None, :] + y_low * sin[:, None, :]
    return np.concatenate([out_low, out_high], axis=-1)       # [T, H, D]


def qkv_rms_norm_rope_cache_with_k_scale_golden(
    qkv,            # [Nq+Nk+Nv, T, D] or [T, Nq+Nk+Nv, D] by layout_qkv
    q_gamma,        # [D]               fp32
    k_gamma,        # [D]               fp32
    cos_sin,        # [MaxSeqLen, D]    fp32 (first D/2 cos, last D/2 sin)
    slot_mapping,   # [T]               int32
    k_cache,        # [BlockNum, Nk, BlockSize, D]   float8_e4m3fn, initial
    v_cache,        # [BlockNum, Nv, BlockSize, D]   float8_e4m3fn, initial
    k_scale_cache,  # [BlockNum, Nk, BlockSize, 1]   fp32 (initial)
    query_start_loc,     # [Batch+1]         int32 local prefix sum
    seq_lens,  # [Batch]         int32 actual lengths after current invocation
    rotation,       # [D, D]            bf16-valued (fp32 storage), shared by Q/K
    v_scale,        # [Nv]              fp32
    head_nums,      # [Nq, Nk, Nv]
    layout_qkv="TND",     # "NTD" or "TND", default TND, controls qkv
    layout_q_out="NTD",   # "NTD" or "TND", default NTD, controls q_out/q_scale
    epsilon=1e-6,
):
    """Return (q_out, q_scale, k_cache, v_cache, k_scale_cache).

    ``q_out``, ``k_cache`` and ``v_cache`` are returned with dtype
    ``ml_dtypes.float8_e4m3fn``. Use ``e4m3fn_to_uint8`` on these arrays if the
    consumer needs raw storage bytes. Cache outputs are copies of the inputs
    with only the touched slots updated (input/output alias semantics).
    """
    Nq, Nk, Nv = (int(head_nums[0]), int(head_nums[1]), int(head_nums[2]))
    layout = "TND" if layout_qkv is None or str(layout_qkv) == "" else str(layout_qkv).upper()
    out_layout = "NTD" if layout_q_out is None or str(layout_q_out) == "" else str(layout_q_out).upper()
    assert layout in ("NTD", "TND"), "layout_qkv must be 'NTD' or 'TND'"
    assert out_layout in ("NTD", "TND"), "layout_q_out must be 'NTD' or 'TND'"
    assert not (layout == "NTD" and out_layout == "TND"), (
        "layout_qkv='NTD' with layout_q_out='TND' is not supported")
    assert Nq > 0 and Nk > 0 and Nv > 0, "head_nums values must be positive"
    assert Nq <= 64, "current arch35 implementation requires Nq <= 64"
    assert Nq == 8 * Nk, "current arch35 implementation requires Nq == 8 * Nk"
    assert Nk == Nv, "current arch35 implementation requires Nk == Nv"
    assert np.ndim(qkv) == 3, "qkv must be rank 3"

    qkv_n = Nq + Nk + Nv
    if layout == "NTD":
        T = int(qkv.shape[1])
        D = int(qkv.shape[2])
        assert qkv.shape[0] == qkv_n, "NTD qkv.shape[0] must equal sum(head_nums)"
    else:
        T = int(qkv.shape[0])
        D = int(qkv.shape[2])
        assert qkv.shape[1] == qkv_n, "TND qkv.shape[1] must equal sum(head_nums)"
    assert D == 128, "current arch35 implementation only supports D == 128"
    assert D % 2 == 0, "D must be even (RoPE half-split)"

    block_size = int(k_cache.shape[2])
    assert np.asarray(q_gamma).shape == (D,), "q_gamma shape must be [D]"
    assert np.asarray(k_gamma).shape == (D,), "k_gamma shape must be [D]"
    assert np.asarray(cos_sin).ndim == 2 and np.asarray(cos_sin).shape[1] == D, "cos_sin shape must be [MaxSeqLen, D]"
    assert np.asarray(slot_mapping).shape == (T,), "slot_mapping shape must be [T]"
    assert np.asarray(k_cache).shape == (np.asarray(k_cache).shape[0], Nk, block_size, D), (
        "k_cache shape must be [BlockNum, Nk, BlockSize, D]")
    assert np.asarray(v_cache).shape == (np.asarray(k_cache).shape[0], Nv, block_size, D), (
        "v_cache shape must be [BlockNum, Nv, BlockSize, D]")
    assert np.asarray(k_scale_cache).shape == (np.asarray(k_cache).shape[0], Nk, block_size, 1), (
        "k_scale_cache shape must be [BlockNum, Nk, BlockSize, 1]")
    assert np.asarray(query_start_loc).shape == (np.asarray(seq_lens).size + 1,), (
        "query_start_loc shape must be [Batch+1]")
    assert np.asarray(rotation).shape == (D, D), "rotation shape must be [D, D]"
    assert np.asarray(v_scale).shape == (Nv,), "v_scale shape must be [Nv]"

    qkv_f = np.asarray(qkv, dtype=np.float32)
    q_gamma_f = np.asarray(q_gamma, dtype=np.float32)
    k_gamma_f = np.asarray(k_gamma, dtype=np.float32)
    rotation_f = to_bf16(rotation).astype(np.float32)  # external shared bf16 matrix
    slot = np.asarray(slot_mapping).astype(np.int64)

    # --- 1. split Q/K/V ---------------------------------------------------
    if layout == "NTD":
        q = np.transpose(qkv_f[0:Nq, :, :], (1, 0, 2))         # [T, Nq, D]
        k = np.transpose(qkv_f[Nq:Nq + Nk, :, :], (1, 0, 2))   # [T, Nk, D]
        v = np.transpose(
            qkv_f[Nq + Nk:Nq + Nk + Nv, :, :], (1, 0, 2))      # [T, Nv, D]
    else:
        q = qkv_f[:, 0:Nq, :]                                  # [T, Nq, D]
        k = qkv_f[:, Nq:Nq + Nk, :]                            # [T, Nk, D]
        v = qkv_f[:, Nq + Nk:Nq + Nk + Nv, :]                  # [T, Nv, D]

    # --- 2. Q/K RMSNorm (fp32, separate gamma) ----------------------------
    q_sum_sq = np.sum(q * q, axis=-1, keepdims=True)         # [T, Nq, 1]
    k_sum_sq = np.sum(k * k, axis=-1, keepdims=True)         # [T, Nk, 1]
    q_rms = np.sqrt(q_sum_sq / D + np.float32(epsilon))      # [T, Nq, 1]
    k_rms = np.sqrt(k_sum_sq / D + np.float32(epsilon))      # [T, Nk, 1]
    q_norm = q / q_rms * q_gamma_f[None, None, :]            # [T, Nq, D]
    k_norm = k / k_rms * k_gamma_f[None, None, :]            # [T, Nk, D]

    # --- 3. Q/K RoPE (fp32), position from local prefix + actual length ----
    q_rope_out = apply_half_split_rope(
        q_norm, cos_sin, query_start_loc, seq_lens)  # [T,Nq,D]
    k_rope_out = apply_half_split_rope(
        k_norm, cos_sin, query_start_loc, seq_lens)  # [T,Nk,D]

    # --- 4. dense Cube rotation matmul: Q/K use shared right matrix --------
    q_rope_bf16 = to_bf16(q_rope_out).astype(np.float32)
    k_rope_bf16 = to_bf16(k_rope_out).astype(np.float32)
    q_after_cube = q_rope_bf16 @ rotation_f                  # [T, Nq, D] fp32
    k_after_cube = k_rope_bf16 @ rotation_f                  # [T, Nk, D] fp32
    qk_after_cube = np.concatenate([q_after_cube, k_after_cube], axis=1)

    # --- 5. Q/K dynamic FP8 quant (per [t,h] row over D) -----------------
    # max_abs == 0 / scale == 0 handling lives in dynamic_quant_e4m3fn:
    # no zero-row guard, an all-zero row yields scale == 0 and NaN output.
    qk_quant, scale = dynamic_quant_e4m3fn(qk_after_cube, axis=-1)  # [T,Nq+Nk,D], [T,Nq+Nk]

    # --- 6. V static FP8 quant -------------------------------------------
    v_scaled = v * np.asarray(v_scale, dtype=np.float32)[None, :, None]
    v_out = cast_to_e4m3fn_sat_rint(v_scaled)               # [T, Nv, D] float8_e4m3fn

    # --- 7. outputs ------------------------------------------------------
    if out_layout == "NTD":
        q_out = np.transpose(qk_quant[:, 0:Nq, :], (1, 0, 2)).copy()  # [Nq, T, D]
        q_scale = np.transpose(scale[:, 0:Nq], (1, 0)).copy().astype(np.float32)  # [Nq, T]
    else:
        q_out = qk_quant[:, 0:Nq, :].copy()                    # [T, Nq, D]
        q_scale = scale[:, 0:Nq].copy().astype(np.float32)      # [T, Nq]

    k_cache_out = np.asarray(k_cache).copy()
    v_cache_out = np.asarray(v_cache).copy()
    k_scale_cache_out = np.asarray(k_scale_cache, dtype=np.float32).copy()

    block_id = slot // block_size
    block_off = slot % block_size
    for t in range(T):
        bi, bo = int(block_id[t]), int(block_off[t])
        # K cache + K scale
        for nk in range(Nk):
            k_cache_out[bi, nk, bo, :] = qk_quant[t, Nq + nk, :]
            k_scale_cache_out[bi, nk, bo, 0] = scale[t, Nq + nk]
        # V cache
        for nv in range(Nv):
            v_cache_out[bi, nv, bo, :] = v_out[t, nv, :]

    return q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out
