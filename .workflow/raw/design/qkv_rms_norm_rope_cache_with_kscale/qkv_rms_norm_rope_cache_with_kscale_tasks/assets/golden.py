"""Numpy golden for QkvRmsNormRopeCacheWithKScale.

This is a faithful reference implementation of the computation chain defined in
``03_golden_design.md`` (and section 8 of the main design document):

    qkv -> split Q/K/V
      Q/K: RMSNorm -> RoPE -> (@ rotation, dense Cube matmul) -> dynamic FP8 quant
           Q -> q_out / q_scale
           K -> k_cache / k_scale_cache   (scatter by slot_mapping)
      V  : v * v_scale -> static FP8 quant -> v_cache (scatter by slot_mapping)

Design-faithful points:
  * Nq/Nk/Nv come from ``head_nums`` (NOT inferred from cache shapes).
  * RoPE position s(t) is derived from the ``cu_seq_len`` prefix sum.
  * ``rotation`` is an external dense [D, D] matrix; no Hadamard assumption.
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

Since ``ml_dtypes`` may be unavailable, BF16 and FP8-E4M3FN casts are implemented
here in pure numpy. ``cast_to_e4m3fn_sat_rint`` returns the exact representable
values stored as float32; ``e4m3fn_to_uint8`` is provided for bit-level compares.

FP8 E4M3FN: 1 sign / 4 exp (bias 7) / 3 mantissa, no inf, max finite = 448,
the all-ones exp+mantissa pattern (0x7F / 0xFF) is NaN.
"""

import numpy as np

FP8_E4M3FN_MAX = 448.0
EXP_BIAS = 7
MIN_NORMAL_EXP = -6      # smallest normal exponent (2**-6)
MANTISSA_BITS = 3


# ---------------------------------------------------------------------------
# BF16 cast (round to nearest even)
# ---------------------------------------------------------------------------
def to_bf16(x):
    """Round float -> bfloat16 (RNE), returned as float32 holding bf16 values."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # Preserve NaN payload (avoid turning NaN into Inf by the rounding add).
    is_nan = np.isnan(x)
    lsb = (u >> 16) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)
    rounded = np.where(is_nan, x, rounded)
    return rounded.astype(np.float32)


# ---------------------------------------------------------------------------
# FP8 E4M3FN cast (RINT, SAT, max finite 448)
# ---------------------------------------------------------------------------
def cast_to_e4m3fn_sat_rint(x):
    """Quantize to FP8 E4M3FN with round-to-nearest-even and saturation.

    Returns the exact representable values as float32. Values whose magnitude
    rounds above 448 saturate to 448 (SAT). Subnormals down to 2**-9 are
    representable; smaller magnitudes flush to zero. NaN stays NaN.
    """
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    sign = np.sign(x)
    a = np.abs(x)

    nan_mask = np.isnan(x)
    a_safe = np.where((a == 0) | nan_mask, 1.0, a)  # avoid log2(0)/log2(nan)

    # Binade exponent: 2**exp <= a < 2**(exp+1).
    exp = np.floor(np.log2(a_safe)).astype(np.int64)
    # Quantization step exponent: normals use 2**(exp-3); subnormals are clamped
    # to the smallest-normal binade so their step is fixed at 2**(-6-3) = 2**-9.
    q_exp = np.maximum(exp, MIN_NORMAL_EXP) - MANTISSA_BITS
    step = np.exp2(q_exp.astype(np.float64))

    # Round-half-to-even in units of the step, then rescale.
    quant = np.round(a / step) * step

    # Saturate to max finite value.
    quant = np.where(quant > FP8_E4M3FN_MAX, FP8_E4M3FN_MAX, quant)

    out = sign * quant
    out = np.where(a == 0, x, out)          # keep signed zero
    out = np.where(nan_mask, np.nan, out)
    return out.astype(np.float32)


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
    """Encode FP8-E4M3FN-representable values to their 8-bit storage pattern.

    Input must already be representable (e.g. output of cast_to_e4m3fn_sat_rint).
    Useful for bit-level golden comparison against kernel output bytes.
    """
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    sign_bit = (np.signbit(x)).astype(np.uint8) << 7
    a = np.abs(x)

    out = np.zeros(a.shape, dtype=np.uint8)
    nan = np.isnan(a)

    # Normals: 2**-6 <= a <= 448
    norm = (a >= 2.0 ** MIN_NORMAL_EXP) & ~nan
    exp = np.zeros(a.shape, dtype=np.int64)
    man = np.zeros(a.shape, dtype=np.int64)
    if np.any(norm):
        e = np.floor(np.log2(np.where(norm, a, 1.0))).astype(np.int64)
        frac = a / np.exp2(e.astype(np.float64)) - 1.0          # in [0,1)
        m = np.round(frac * (1 << MANTISSA_BITS)).astype(np.int64)
        # carry: frac rounded up to 1.0 bumps the exponent
        carry = m == (1 << MANTISSA_BITS)
        e = np.where(carry, e + 1, e)
        m = np.where(carry, 0, m)
        exp = np.where(norm, e + EXP_BIAS, exp)
        man = np.where(norm, m, man)
        out = np.where(norm, ((exp << MANTISSA_BITS) | man).astype(np.uint8), out)

    # Subnormals: 0 < a < 2**-6 (biased exp field = 0)
    sub = (a > 0) & (a < 2.0 ** MIN_NORMAL_EXP) & ~nan
    if np.any(sub):
        m = np.round(a / (2.0 ** (MIN_NORMAL_EXP - MANTISSA_BITS))).astype(np.int64)
        out = np.where(sub, m.astype(np.uint8), out)

    out = np.where(nan, np.uint8(0x7F), out)   # NaN pattern
    return (out | sign_bit).astype(np.uint8)


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------
def qkv_rms_norm_rope_cache_with_kscale_golden(
    qkv,            # [T, Nq+Nk+Nv, D]  bf16-valued (fp32 storage)
    q_gamma,        # [D]               fp32
    k_gamma,        # [D]               fp32
    cos_sin,        # [MaxSeqLen, D]    fp32 (first D/2 cos, last D/2 sin)
    slot_mapping,   # [T]               int32
    k_cache,        # [BlockNum, Nk, BlockSize, D]   fp8 (fp32 storage, initial)
    v_cache,        # [BlockNum, Nv, BlockSize, D]   fp8 (fp32 storage, initial)
    k_scale_cache,  # [BlockNum, Nk, BlockSize, 1]   fp32 (initial)
    cu_seq_len,     # [Batch+1]         int32 prefix sum
    rotation,       # [D, D]            bf16-valued (fp32 storage)
    v_scale,        # [Nv]              fp32
    head_nums,      # [Nq, Nk, Nv]
    epsilon=1e-6,
):
    """Return (q_out, q_scale, k_cache, v_cache, k_scale_cache).

    Cache outputs are copies of the inputs with only the touched slots updated
    (input/output alias semantics).
    """
    Nq, Nk, Nv = (int(head_nums[0]), int(head_nums[1]), int(head_nums[2]))
    T = int(qkv.shape[0])
    D = int(qkv.shape[1 + 1])  # qkv: [T, H, D]
    assert qkv.shape[1] == Nq + Nk + Nv, "qkv.shape[1] must equal sum(head_nums)"
    assert D % 2 == 0, "D must be even (RoPE half-split)"
    half = D // 2

    block_size = int(k_cache.shape[2])

    qkv_f = np.asarray(qkv, dtype=np.float32)
    rotation_f = to_bf16(rotation).astype(np.float32)        # external bf16 matrix
    cos_sin_f = np.asarray(cos_sin, dtype=np.float32)
    cu = np.asarray(cu_seq_len).astype(np.int64)
    slot = np.asarray(slot_mapping).astype(np.int64)

    # --- 1. split Q/K/V ---------------------------------------------------
    q = qkv_f[:, 0:Nq, :]
    k = qkv_f[:, Nq:Nq + Nk, :]
    v = qkv_f[:, Nq + Nk:Nq + Nk + Nv, :]
    qk = np.concatenate([q, k], axis=1)                      # [T, Nq+Nk, D]

    # --- 2. RMSNorm (fp32) ------------------------------------------------
    sum_sq = np.sum(qk * qk, axis=-1, keepdims=True)         # [T, Nq+Nk, 1]
    rms = np.sqrt(sum_sq / D + np.float32(epsilon))          # [T, Nq+Nk, 1]
    gamma = np.concatenate(
        [np.broadcast_to(q_gamma, (Nq, D)),
         np.broadcast_to(k_gamma, (Nk, D))], axis=0).astype(np.float32)   # [Nq+Nk, D]
    y_norm = qk / rms * gamma[None, :, :]                    # [T, Nq+Nk, D]

    # --- 3. RoPE (fp32), position from cu_seq_len prefix sum --------------
    b_of_t = np.searchsorted(cu, np.arange(T, dtype=np.int64), side="right") - 1
    s_of_t = np.arange(T, dtype=np.int64) - cu[b_of_t]       # batch-local position
    cos = cos_sin_f[s_of_t, 0:half]                          # [T, half]
    sin = cos_sin_f[s_of_t, half:D]                          # [T, half]
    y_low = y_norm[:, :, 0:half]
    y_high = y_norm[:, :, half:D]
    out_low = y_low * cos[:, None, :] - y_high * sin[:, None, :]
    out_high = y_high * cos[:, None, :] + y_low * sin[:, None, :]
    rope_out = np.concatenate([out_low, out_high], axis=-1)  # [T, Nq+Nk, D]

    # --- 4. rotation matmul: cast RoPE to bf16, accumulate fp32 -----------
    rope_bf16 = to_bf16(rope_out).astype(np.float32)
    qk_after_cube = rope_bf16 @ rotation_f                   # [T, Nq+Nk, D] fp32

    # --- 5. Q/K dynamic FP8 quant (per [t,h] row over D) -----------------
    # max_abs == 0 / scale == 0 handling lives in dynamic_quant_e4m3fn:
    # no zero-row guard, an all-zero row yields scale == 0 and NaN output.
    qk_quant, scale = dynamic_quant_e4m3fn(qk_after_cube, axis=-1)  # [T,Nq+Nk,D], [T,Nq+Nk]

    # --- 6. V static FP8 quant -------------------------------------------
    v_scaled = v * np.asarray(v_scale, dtype=np.float32)[None, :, None]
    v_out = cast_to_e4m3fn_sat_rint(v_scaled)               # [T, Nv, D]

    # --- 7. outputs ------------------------------------------------------
    q_out = qk_quant[:, 0:Nq, :].copy()                     # [T, Nq, D]
    q_scale = scale[:, 0:Nq].copy().astype(np.float32)      # [T, Nq]

    k_cache_out = np.asarray(k_cache, dtype=np.float32).copy()
    v_cache_out = np.asarray(v_cache, dtype=np.float32).copy()
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
