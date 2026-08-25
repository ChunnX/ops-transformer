#!/usr/bin/env python3
"""Repository-side deterministic test entrypoint for the B-I clean-room oracle."""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "oracle" / "mxfp4_b_oracle.py"
SPEC = importlib.util.spec_from_file_location("mxfp4_b_oracle", ORACLE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BatchAndBoundaryTests(unittest.TestCase):
    def test_two_records_keep_per_record_scales_and_rope(self) -> None:
        first = MODULE.pack_record([0x6] * 512, [0.5] * 64, [126] * 16, "bf16")
        second = MODULE.pack_record([0xE] * 512, [-0.75] * 64, [128] * 16, "bf16")
        decoded = [MODULE.decode_record(record, "bf16") for record in (first, second)]
        self.assertEqual(decoded[0][0], MODULE.decode_fp4_e2m1(0x6) * MODULE.decode_e8m0(126))
        self.assertEqual(decoded[1][0], MODULE.decode_fp4_e2m1(0xE) * MODULE.decode_e8m0(128))
        self.assertEqual(decoded[0][512:], MODULE.decode_rope(first[256:384], "bf16"))
        self.assertEqual(decoded[1][512:], MODULE.decode_rope(second[256:384], "bf16"))

    def test_group_15_boundary_is_d_480_to_511(self) -> None:
        nibbles = [0x2] * 512
        scales = [127] * 15 + [130]
        record = MODULE.pack_record(nibbles, [0.0] * 64, scales, "bf16")
        decoded = MODULE.decode_record(record, "bf16")
        self.assertEqual(decoded[479], MODULE.decode_fp4_e2m1(0x2) * MODULE.decode_e8m0(127))
        self.assertEqual(decoded[480], MODULE.decode_fp4_e2m1(0x2) * MODULE.decode_e8m0(130))
        self.assertEqual(decoded[511], MODULE.decode_fp4_e2m1(0x2) * MODULE.decode_e8m0(130))

    def test_e8m0_boundary_bytes_match_kernel_widening(self) -> None:
        # 0x00 is the E8M0 subnormal (2^-127), not +0.0; 0xfe is the max valid exponent (2^127).
        self.assertEqual(MODULE.decode_e8m0(0x00), 2.0 ** -127)
        self.assertEqual(MODULE.decode_e8m0(0xFE), 2.0 ** 127)
        self.assertNotEqual(MODULE.decode_e8m0(0x00), 0.0)
        # bf16 bit patterns the kernel must emit: 0x00 -> 0x0040, 0x01 -> 0x0080, 0xfe -> 0x7f00.
        self.assertEqual(MODULE.float_to_bf16_bits(MODULE.decode_e8m0(0x00)), 0x0040)
        self.assertEqual(MODULE.float_to_bf16_bits(MODULE.decode_e8m0(0x01)), 0x0080)
        self.assertEqual(MODULE.float_to_bf16_bits(MODULE.decode_e8m0(0xFE)), 0x7F00)

    def test_e8m0_scale_0xff_rejected_in_record(self) -> None:
        record = bytearray(MODULE.pack_record([0x2] * 512, [0.0] * 64, [127] * 16, "bf16"))
        record[MODULE.SCALE_OFFSET] = 0xFF
        with self.assertRaises(MODULE.RecordFormatError):
            MODULE.decode_record(bytes(record), "bf16")
        with self.assertRaises(MODULE.RecordFormatError):
            MODULE.decode_e8m0(0xFF)

    def test_scale_scratch_layout_duplicates_each_e8m0_group(self) -> None:
        scale_bytes = list(range(120, 136))
        record = MODULE.pack_record([0x2] * 512, [0.0] * 64, scale_bytes, "bf16")
        staged = MODULE.stage_scale_bf16_bits(record)
        self.assertEqual(len(staged), 32)
        for group, scale_byte in enumerate(scale_bytes):
            expected = MODULE.float_to_bf16_bits(MODULE.decode_e8m0(scale_byte))
            self.assertEqual(staged[2 * group:2 * group + 2], [expected, expected])

    def test_rope_raw_storage_is_byte_exact_including_nan_payloads(self) -> None:
        record = bytearray(MODULE.pack_record([0] * 512, [0.0] * 64, [127] * 16, "bf16"))
        raw_rope = bytes.fromhex("817f017f017e007e") * 16  # BF16 NaN/signaling/subnormal payloads
        self.assertEqual(len(raw_rope), MODULE.ROPE_BYTES)
        record[MODULE.ROPE_OFFSET:MODULE.SCALE_OFFSET] = raw_rope
        self.assertEqual(MODULE.raw_rope_storage(bytes(record)), raw_rope)


if __name__ == "__main__":
    unittest.main(verbosity=2)
