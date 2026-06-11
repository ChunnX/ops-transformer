"""
HiF8 LUT decode + valid code generation.
Companion to hif8_utils.py — provides optimized decode path.

Usage:
    from hif8_lut import hif8_to_fp32_lut, get_valid_hif8_codes
"""
import numpy as np
from hif8_utils import hif8_to_fp32_numpy

# ═══════════════════════════════════════════════════════════════════
# LUT Decode: 256-entry table, pure integer lookup
# ═══════════════════════════════════════════════════════════════════

_LUT_FP32_BITS = None


def _build_lut():
    """Build 256-entry LUT once. Uses baseline decoder — bit-exact by construction."""
    global _LUT_FP32_BITS
    all_codes = np.arange(256, dtype=np.uint8)
    fp32_vals = hif8_to_fp32_numpy(all_codes)
    _LUT_FP32_BITS = fp32_vals.view(np.uint32).copy()


def hif8_to_fp32_lut(codes):
    """Decode HiF8 uint8 -> fp32 via 256-entry LUT.
    Bit-exact equivalent to hif8_to_fp32_numpy, ~24x faster.
    """
    global _LUT_FP32_BITS
    if _LUT_FP32_BITS is None:
        _build_lut()
    flat = codes.ravel().astype(np.uint8)
    bits = _LUT_FP32_BITS[flat]
    return bits.view(np.float32).reshape(codes.shape)


# ═══════════════════════════════════════════════════════════════════
# Valid code computation — for direct uint8 generation (skip encode)
# ═══════════════════════════════════════════════════════════════════
_VALID_CACHE = {}


def get_valid_hif8_codes(min_val, max_val):
    """Return HiF8 codes whose decoded fp32 value is in [min_val, max_val] and finite.
    These are exactly the codes fp32_to_hif8_numpy would produce for uniform input.
    Cached per (min_val, max_val) pair.
    """
    key = (float(min_val), float(max_val))
    if key in _VALID_CACHE:
        return _VALID_CACHE[key].copy()

    all_codes = np.arange(256, dtype=np.uint8)
    all_vals = hif8_to_fp32_numpy(all_codes)
    mask = np.isfinite(all_vals) & (all_vals >= min_val) & (all_vals <= max_val)
    valid = all_codes[mask]

    _VALID_CACHE[key] = valid
    return valid.copy()

_VU_CACHE = {}

def sample_hif8_value_uniform(min_val, max_val, shape):
    key = (float(min_val), float(max_val))
    if key not in _VU_CACHE:
        codes = get_valid_hif8_codes(min_val, max_val)
        vals = hif8_to_fp32_numpy(codes)
        sorted_idx = np.argsort(vals)
        sorted_vals = vals[sorted_idx].astype(np.float64)
        sorted_codes = codes[sorted_idx]
        mids = (sorted_vals[:-1] + sorted_vals[1:]) * 0.5
        lo = np.empty(len(sorted_vals), dtype=np.float64)
        hi = np.empty(len(sorted_vals), dtype=np.float64)
        lo[0] = min_val
        lo[1:] = mids
        hi[:-1] = mids
        hi[-1] = max_val
        widths = np.maximum(hi - lo, 0.0)
        total = widths.sum()
        probs = widths / total if total > 0 else np.ones_like(widths) / len(widths)
        _VU_CACHE[key] = (sorted_codes, probs)
    codes, probs = _VU_CACHE[key]
    return np.random.choice(codes, size=shape, p=probs)