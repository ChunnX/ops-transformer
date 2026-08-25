#!/usr/bin/env python3
"""Clean-room decoder and self-test for the QSFA B-I MXFP4 key record."""

from __future__ import annotations

import argparse
import math
import struct
import unittest
from typing import Iterable, Sequence

NOPE_DIM = 512
ROPE_DIM = 64
FP4_PACK_BYTES = NOPE_DIM // 2
ROPE_BYTES = ROPE_DIM * 2
MX_GROUP_SIZE = 32
SCALE_COUNT = NOPE_DIM // MX_GROUP_SIZE
RECORD_BYTES = FP4_PACK_BYTES + ROPE_BYTES + SCALE_COUNT
FP4_OFFSET = 0
ROPE_OFFSET = FP4_OFFSET + FP4_PACK_BYTES
SCALE_OFFSET = ROPE_OFFSET + ROPE_BYTES


class RecordFormatError(ValueError):
    """The caller supplied a record outside the frozen B-I contract."""


def decode_fp4_e2m1(nibble: int) -> float:
    """Decode one finite FP4 E2M1 nibble; bit 3 is sign, bits 2:1 exponent."""
    if not 0 <= nibble <= 0xF:
        raise RecordFormatError(f"FP4 nibble is outside [0, 15]: {nibble}")
    sign = -1.0 if nibble & 0x8 else 1.0
    exponent = (nibble >> 1) & 0x3
    mantissa = nibble & 0x1
    if exponent == 0:
        magnitude = 0.5 if mantissa else 0.0
    else:
        magnitude = (1.0 + 0.5 * mantissa) * (2.0 ** (exponent - 1))
    return sign * magnitude


def decode_e8m0(scale_byte: int) -> float:
    """Decode a finite E8M0 scale using unsigned exponent bits with bias 127."""
    if not 0 <= scale_byte <= 0xFF:
        raise RecordFormatError(f"E8M0 byte is outside [0, 255]: {scale_byte}")
    if scale_byte == 0xFF:
        raise RecordFormatError("E8M0 byte 0xff is unsupported/NaN in B-I")
    return math.ldexp(1.0, scale_byte - 127)


def bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def fp16_bits_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def float_to_bf16_bits(value: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    return (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16


def float_to_fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def decode_rope(raw: bytes, q_dtype: str) -> list[float]:
    if len(raw) != ROPE_BYTES:
        raise RecordFormatError(f"RoPE storage must be {ROPE_BYTES} bytes")
    unpack = bf16_bits_to_float if q_dtype == "bf16" else fp16_bits_to_float
    if q_dtype not in {"bf16", "fp16"}:
        raise RecordFormatError(f"unsupported Q_T dtype: {q_dtype}")
    return [unpack(struct.unpack_from("<H", raw, 2 * i)[0]) for i in range(ROPE_DIM)]


def raw_rope_storage(record: bytes) -> bytes:
    """Return the opaque RoPE byte suffix without any floating-point conversion."""
    if len(record) != RECORD_BYTES:
        raise RecordFormatError(f"B-I key record must be {RECORD_BYTES} bytes, got {len(record)}")
    return record[ROPE_OFFSET:SCALE_OFFSET]


def stage_scale_bf16_bits(record: bytes) -> list[int]:
    """Model the V0 scale scratch row: every E8M0 scale becomes two BF16 lanes."""
    if len(record) != RECORD_BYTES:
        raise RecordFormatError(f"B-I key record must be {RECORD_BYTES} bytes, got {len(record)}")
    staged: list[int] = []
    for scale_byte in record[SCALE_OFFSET:RECORD_BYTES]:
        staged_bits = float_to_bf16_bits(decode_e8m0(scale_byte))
        staged.extend([staged_bits, staged_bits])
    return staged


def decode_record(record: bytes, q_dtype: str) -> list[float]:
    if len(record) != RECORD_BYTES:
        raise RecordFormatError(f"B-I key record must be {RECORD_BYTES} bytes, got {len(record)}")
    scales = [decode_e8m0(record[SCALE_OFFSET + group]) for group in range(SCALE_COUNT)]
    nope: list[float] = []
    for byte_index in range(FP4_PACK_BYTES):
        packed = record[FP4_OFFSET + byte_index]
        group = (2 * byte_index) // MX_GROUP_SIZE
        scale = scales[group]
        nope.append(decode_fp4_e2m1(packed & 0xF) * scale)
        nope.append(decode_fp4_e2m1((packed >> 4) & 0xF) * scale)
    return nope + decode_rope(record[ROPE_OFFSET:SCALE_OFFSET], q_dtype)


def pack_record(nope_nibbles: Sequence[int], rope_values: Sequence[float], scale_bytes: Sequence[int], q_dtype: str) -> bytes:
    if len(nope_nibbles) != NOPE_DIM or len(rope_values) != ROPE_DIM or len(scale_bytes) != SCALE_COUNT:
        raise RecordFormatError("pack_record requires 512 FP4 values, 64 RoPE values, and 16 scales")
    data = bytearray(RECORD_BYTES)
    for i in range(0, NOPE_DIM, 2):
        lo, hi = nope_nibbles[i], nope_nibbles[i + 1]
        if not 0 <= lo <= 0xF or not 0 <= hi <= 0xF:
            raise RecordFormatError("FP4 nibble outside [0, 15]")
        data[FP4_OFFSET + i // 2] = lo | (hi << 4)
    if q_dtype == "bf16":
        pack_rope = float_to_bf16_bits
    elif q_dtype == "fp16":
        pack_rope = float_to_fp16_bits
    else:
        raise RecordFormatError(f"unsupported Q_T dtype: {q_dtype}")
    for i, value in enumerate(rope_values):
        struct.pack_into("<H", data, ROPE_OFFSET + 2 * i, pack_rope(value))
    for i, scale in enumerate(scale_bytes):
        if not 0 <= scale < 0xFF:
            raise RecordFormatError("B-I scale byte must be in [0, 254]")
        data[SCALE_OFFSET + i] = scale
    return bytes(data)


class OracleTests(unittest.TestCase):
    def test_group_scales_and_nibble_order(self) -> None:
        scales = [120 + i for i in range(SCALE_COUNT)]
        nibbles = [0x2 if i % 2 == 0 else 0xB for i in range(NOPE_DIM)]
        rope = [float(i - 31) / 8.0 for i in range(ROPE_DIM)]
        record = pack_record(nibbles, rope, scales, "bf16")
        output = decode_record(record, "bf16")
        self.assertEqual(len(output), 576)
        for group, scale_byte in enumerate(scales):
            scale = decode_e8m0(scale_byte)
            self.assertEqual(output[group * 32], decode_fp4_e2m1(0x2) * scale)
            self.assertEqual(output[group * 32 + 1], decode_fp4_e2m1(0xB) * scale)
        expected_rope = decode_rope(record[ROPE_OFFSET:SCALE_OFFSET], "bf16")
        self.assertEqual(output[NOPE_DIM:], expected_rope)

    def test_fp16_rope_is_preserved_as_fp16_values(self) -> None:
        record = pack_record([0] * NOPE_DIM, [0.25 * i for i in range(ROPE_DIM)], [127] * SCALE_COUNT, "fp16")
        self.assertEqual(decode_record(record, "fp16")[NOPE_DIM:], decode_rope(record[ROPE_OFFSET:SCALE_OFFSET], "fp16"))

    def test_invalid_record_and_nan_scale_fail(self) -> None:
        with self.assertRaises(RecordFormatError):
            decode_record(bytes(RECORD_BYTES - 1), "bf16")
        record = bytearray(pack_record([0] * NOPE_DIM, [0.0] * ROPE_DIM, [127] * SCALE_COUNT, "bf16"))
        record[SCALE_OFFSET] = 0xFF
        with self.assertRaises(RecordFormatError):
            decode_record(bytes(record), "bf16")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(OracleTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    parser.error("only --self-test is supported")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
