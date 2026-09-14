#!/usr/bin/env python3
"""CPU golden for the MXFP8 weight-NZ GMM + SwiGLU V3 demo.

The script mirrors the deterministic data layout in
`test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp`, including FRACTAL_NZ
weight decoding, the four SwiGLU modes, and MXFP8 output quantization.
"""

import math
import sys
from pathlib import Path

import numpy as np


FP8_PATTERN = np.array(
    [0x38, 0xB8, 0x40, 0x30, 0xC0, 0x28, 0x3C, 0xBC, 0x48, 0xC8, 0x50, 0x20],
    dtype=np.uint8,
)
# K=128 matches the ACLNN demo and is the alignment condition for the V3
# Tensor API mode-2 path.
M, N, K = 2048, 128, 128
BLOCK_SIZE = 32
FP8_E4M3_EMAX = 8


def _decode_e4m3(code):
    code = int(code)
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    # E4M3FN reserves the two exp=15, mantissa=7 encodings.
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    if exponent == 0:
        value = (mantissa / 8.0) * (2.0 ** -6)
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return sign * value


FP8_E4M3_LUT = np.array([_decode_e4m3(i) for i in range(256)], dtype=np.float32)
FP8_FINITE_CODES = np.flatnonzero(np.isfinite(FP8_E4M3_LUT))
FP8_FINITE_VALUES = FP8_E4M3_LUT[FP8_FINITE_CODES]


def _decode_e8m0(code):
    code = int(code)
    if code == 0xFF:
        return math.nan
    return 2.0 ** (code - 127)


def _sigmoid(value):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-value))


def _make_inputs(transpose):
    x_raw = np.fromiter(
        (int(FP8_PATTERN[(i * 7 + 3) % len(FP8_PATTERN)]) for i in range(M * K)),
        dtype=np.uint8,
        count=M * K,
    ).reshape(M, K)
    storage_size = (K // 32) * (N // 16) * 16 * 32 if transpose else (N // 32) * (K // 16) * 16 * 32
    weight_raw = np.fromiter(
        (int(FP8_PATTERN[(i * 5 + 1) % len(FP8_PATTERN)]) for i in range(storage_size)),
        dtype=np.uint8,
        count=storage_size,
    )
    x = FP8_E4M3_LUT[x_raw]
    if transpose:
        # Storage is [K/32, N/16, 16, 32], i.e. [k1, n1, n0, k0].
        storage = weight_raw.reshape(K // 32, N // 16, 16, 32)
        weight = FP8_E4M3_LUT[storage.transpose(0, 3, 1, 2)].reshape(K, N)
    else:
        # Storage is [N/32, K/16, 16, 32], i.e. [n1, k1, k0, n0].
        storage = weight_raw.reshape(N // 32, K // 16, 16, 32)
        weight = FP8_E4M3_LUT[storage.transpose(1, 2, 0, 3)].reshape(K, N)
    return x.astype(np.float32), weight.astype(np.float32)


def _swiglu(mm_result, mode, activate_left):
    half = mm_result.shape[1] // 2
    if mode == 0:
        first, second = mm_result[:, :half], mm_result[:, half:]
        if activate_left:
            return _sigmoid(first) * first * second
        return _sigmoid(second) * first * second
    if mode == 1:
        # The V3 mode-1 stream is [activation0, linear0, activation1, linear1, ...].
        activation = np.minimum(mm_result[:, 0::2], 7.0)
        linear = np.clip(mm_result[:, 1::2], -7.0, 7.0)
        return activation * _sigmoid(1.702 * activation) * (linear + 1.0)
    first, second = mm_result[:, :half], mm_result[:, half:]
    if not activate_left:
        first, second = second, first
    if mode == 2:
        first = np.minimum(first, 7.0)
        second = np.clip(second, -7.0, 7.0)
        return first * _sigmoid(1.702 * first) * (second + 1.0)
    if mode == 3:
        first = np.minimum(first * _sigmoid(first), 7.0)
        second = np.clip(second, -7.0, 7.0)
        return first * second
    raise ValueError(f"unsupported swiglu mode: {mode}")


def _to_bfloat16(values):
    """Round float32 values to the BF16 representation used by CGMCT."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    round_bit = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + round_bit) & 0xFFFF0000).view(np.float32)


def _nearest_fp8_code(values):
    values = np.asarray(values, dtype=np.float32)
    clipped = np.clip(values, -448.0, 448.0)
    flat = clipped.reshape(-1, 1)
    distance = np.abs(flat - FP8_FINITE_VALUES.reshape(1, -1))
    min_distance = np.min(distance, axis=1, keepdims=True)
    ties = np.isclose(distance, min_distance, rtol=0.0, atol=1e-5)
    # CAST_RINT uses round-to-nearest-even. The FP8 significand parity is
    # the parity of the low mantissa bit (the implicit bit is always one).
    even_code = (FP8_FINITE_CODES & 1) == 0
    even_ties = ties & even_code[None, :]
    best = np.argmax(even_ties, axis=1)
    no_even_tie = ~np.any(even_ties, axis=1)
    best[no_even_tie] = np.argmin(distance[no_even_tie], axis=1)
    return FP8_FINITE_CODES[best].reshape(values.shape)


def _quantize_mxfp8(values):
    values = np.asarray(values, dtype=np.float32)
    output = np.empty_like(values, dtype=np.float32)
    scale_bytes = np.empty((values.shape[0], values.shape[1] // BLOCK_SIZE), dtype=np.uint8)
    for block_index, begin in enumerate(range(0, values.shape[1], BLOCK_SIZE)):
        end = begin + BLOCK_SIZE
        block = values[:, begin:end]
        abs_max = np.max(np.abs(block), axis=1)
        safe_max = np.where(abs_max == 0.0, 1.0, abs_max)
        shared_exp = np.floor(np.log2(safe_max)).astype(np.int32) - FP8_E4M3_EMAX
        shared_exp = np.clip(shared_exp, -127, 127)
        scale = np.exp2(shared_exp.astype(np.float32))
        # CGMCT applies CAST_RINT directly from the scaled BF16 value to E4M3;
        # it does not first round the scaled value to an integer.
        quantized_code = _nearest_fp8_code(block / scale[:, None])
        output[:, begin:end] = FP8_E4M3_LUT[quantized_code] * scale[:, None]
        scale_bytes[:, block_index] = np.clip(shared_exp + 127, 0, 254).astype(np.uint8)
    return output, scale_bytes


def _cpu_golden(mode, transpose, activate_left):
    x, weight = _make_inputs(transpose)
    # The MXFP8 CGMCT epilogue consumes BF16 MM results and stores the
    # SwiGLU result to BF16 before dynamic MX quantization.
    mm_result = _to_bfloat16(x @ weight)
    swiglu = _to_bfloat16(_swiglu(mm_result, mode, activate_left))
    return _quantize_mxfp8(swiglu)


def main(argv):
    if len(argv) != 6:
        print(
            f"usage: {argv[0]} <mode 0..3> <transpose 0|1> <activate_left 0|1> "
            "<output.bin> <output_scale.bin>",
            file=sys.stderr,
        )
        return 2
    mode, transpose, activate_left = (int(value) for value in argv[1:4])
    output_path = Path(argv[4])
    scale_path = Path(argv[5])
    expected, expected_scale = _cpu_golden(mode, bool(transpose), bool(activate_left))
    actual_code = np.fromfile(output_path, dtype=np.uint8)
    actual_scale_code = np.fromfile(scale_path, dtype=np.uint8)
    if actual_code.size != M * (N // 2) or actual_scale_code.size != M * 2:
        print(
            f"FAIL invalid artifact size: output={actual_code.size}, scale={actual_scale_code.size}",
            file=sys.stderr,
        )
        return 1
    actual = FP8_E4M3_LUT[actual_code].reshape(M, N // 2)
    actual_scale_code = actual_scale_code.reshape(M, 2)
    actual_scale = np.exp2(actual_scale_code.astype(np.int16) - 127).astype(np.float32)
    actual = actual * np.repeat(actual_scale, BLOCK_SIZE, axis=1)
    difference = np.abs(actual - expected)
    threshold = 1e-3 * np.maximum(1.0, np.abs(expected))
    bad = np.count_nonzero(difference > threshold)
    allowed = (expected.size - 1) // 1000
    scale_bad = np.count_nonzero(actual_scale_code != expected_scale)
    max_relative = float(np.max(difference / np.maximum(1.0, np.abs(expected))))
    print(
        f"golden mode={mode} transpose={transpose} activate_left={activate_left}: "
        f"bad={bad}/{expected.size} allowed={allowed}, scale_bad={scale_bad}, "
        f"max_relative_error={max_relative:.6g}"
    )
    if bad > allowed or scale_bad != 0:
        return 1
    print("golden PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
