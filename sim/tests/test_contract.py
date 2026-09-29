# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU contract tests: no simulator and no fake results presented as a kernel run."""

import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, read, validate_config, write
from case import AddressMap, CHIP_STRIDE, GUARD, write_case
from tensors import fp4_nz_encode, fp4_nz_decode, routing, inputs
from validate import validate_case, validate_source_abi, check_outputs


class Contracts(unittest.TestCase):
    def setUp(self):
        self.cfg = read(ROOT / "sim/configs/bs72.json")

    def test_balanced_eight_rank_routing(self):
        totals = np.zeros(24, np.int64)
        for rank in range(8):
            ids = routing(self.cfg, rank)
            self.assertEqual(ids.shape, (72, 6))
            self.assertEqual(ids.dtype, np.dtype("<i4"))
            self.assertTrue(all(len(set(row)) == 6 for row in ids.tolist()))
            counts = np.bincount(ids.ravel(), minlength=24)
            np.testing.assert_array_equal(counts, np.full(24, 18))
            totals += counts
        np.testing.assert_array_equal(totals, np.full(24, 144))

    def test_fp4_nz_physical_offsets_and_nibbles(self):
        rng = np.random.default_rng(2)
        logical = rng.integers(0, 16, (3, 48, 96), dtype=np.uint8)
        encoded = fp4_nz_encode(logical)
        flat = encoded.ravel()
        # Independent scalar physical-address oracle, across expert/fractal boundaries.
        for e in range(3):
            for n in (0, 15, 16, 31, 32, 47):
                for k in (0, 1, 30, 31, 32, 33, 63, 64, 95):
                    element = (
                        e * 48 * 96
                        + (k // 32) * 48 * 32
                        + (n // 16) * 16 * 32
                        + (n % 16) * 32
                        + k % 32
                    )
                    nibble = (int(flat[element // 2]) >> (4 * (element % 2))) & 15
                    self.assertEqual(nibble, logical[e, n, k])
        np.testing.assert_array_equal(fp4_nz_decode(encoded, 3, 48, 96), logical)

    def test_prequantized_input_sizes(self):
        tensors = inputs(self.cfg, 3)
        self.assertEqual(tensors[1][1].nbytes, 72 * 5120)
        self.assertEqual(tensors[11][1].shape, (72, 80, 2))
        self.assertEqual(tensors[11][1].nbytes, 11520)
        self.assertTrue(np.all(tensors[11][1] == 125))

    def test_address_overlap_is_rejected(self):
        memory = AddressMap(7)
        with self.assertRaisesRegex(ValueError, "overlap"):
            memory.reserve("bad_tensor", 7 * CHIP_STRIDE + 0x9010000000, 512)
        with self.assertRaisesRegex(ValueError, "rank-local"):
            memory.reserve("cross_rank", 8 * CHIP_STRIDE - 32, 64)

    def test_shape_changes_require_consistent_key_and_capacity(self):
        for key, value in (
            ("tiling_key", 0),
            ("aiv_per_rank", 32),
            ("hidden", 5121),
            ("max_recv_tokens", 431),
        ):
            cfg = copy.deepcopy(self.cfg)
            cfg["model"][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_current_kernel_abi_is_1c2v_23_arguments(self):
        validate_source_abi()

    def test_full_serialized_case_and_failure_detection(self):
        # Small tensors retain all 8 real peer addresses and 32AIC/64AIV task fields.
        cfg = copy.deepcopy(self.cfg)
        cfg["simulation"]["dump_outputs"] = True
        cfg["model"].update(
            hidden=1024, intermediate=256, tokens_per_rank=4, max_tokens_per_rank=4
        )
        info = {
            "tiling_key": 730152,
            "tiling_bytes": 432,
            "workspace_bytes": 65536,
            "window_bytes": 65536,
            "context_bytes": 16400,
            "context_windows_offset": 16,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "tiling.bin").write_bytes(bytes(432))
            write(dest / "layout.json", info)
            # Dummy bytes are only serializer fixtures in a temporary directory.
            write_case(
                cfg,
                dest,
                info,
                {"aic": b"TEST_ONLY_AIC" * 100, "aiv": b"TEST_ONLY_AIV" * 100},
            )
            self.assertEqual(validate_case(dest)["tokens_per_expert"], 8)
            task_path = dest / "chip_0/mega_task.json"
            task = read(task_path)
            original = copy.deepcopy(task)
            task["sqe_type"] = "AIV"
            write(task_path, task)
            with self.assertRaisesRegex(ValueError, "sqe_type"):
                validate_case(dest)
            write(task_path, original)
            maps = read(dest / "memory_map.json")
            for rank in range(8):
                d = dest / f"chip_{rank}"
                for t in maps[str(rank)]["tensors"].values():
                    (d / t["guard"]).write_bytes(GUARD)
                (d / "actual_counts.bin").write_bytes(struct.pack("<iii", 8, 8, 8))
                (d / "actual_y.bin").write_bytes(struct.pack("<H", 0x3F80) * (4 * 1024))
            self.assertEqual(check_outputs(dest)["numerical_accuracy"], "not_checked")
            (dest / "chip_7/guard_workspace.bin").write_bytes(bytes(len(GUARD)))
            with self.assertRaisesRegex(ValueError, "boundary overwritten"):
                check_outputs(dest)
            payload = dest / "chip_0/x.bin"
            with payload.open("r+b") as f:
                f.write(b"\x7f")
            with self.assertRaisesRegex(ValueError, "modified"):
                validate_case(dest)

    def test_float32_target_and_serialization(self):
        cfg = read(ROOT / "sim/configs/shared_start_bs128.json")
        validate_config(cfg)
        totals = np.zeros(24, np.int64)
        for rank in range(8):
            tensor = inputs(cfg, rank)
            self.assertEqual(tensor[3][1].dtype, np.dtype("<f4"))
            self.assertEqual(tensor[3][1].nbytes, 128 * 6 * 4)
            totals += np.bincount(tensor[2][1].ravel(), minlength=24)
        np.testing.assert_array_equal(totals, np.full(24, 256))
        cfg["model"].update(
            hidden=1024, intermediate=256, tokens_per_rank=4, max_tokens_per_rank=4
        )
        info = {
            "tiling_key": 730152,
            "tiling_bytes": 432,
            "workspace_bytes": 65536,
            "window_bytes": 65536,
            "context_bytes": 16400,
            "context_windows_offset": 16,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "tiling.bin").write_bytes(bytes(432))
            write(dest / "layout.json", info)
            write_case(cfg, dest, info, None)
            self.assertFalse(validate_case(dest, inputs_only=True)["kernel_attached"])
            cfg["topk_weights_dtype"] = "bfloat16"
            write(dest / "config.json", cfg)
            with self.assertRaisesRegex(ValueError, "topk_weights"):
                validate_case(dest, inputs_only=True)

    def test_input_only_profile_matches_a5_loader(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"].update(
            hidden=1024, intermediate=256, tokens_per_rank=4, max_tokens_per_rank=4
        )
        info = {
            "tiling_key": 730152,
            "tiling_bytes": 432,
            "workspace_bytes": 65536,
            "window_bytes": 65536,
            "context_bytes": 16400,
            "context_windows_offset": 16,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "tiling.bin").write_bytes(bytes(432))
            write(dest / "layout.json", info)
            write_case(cfg, dest, info, None)
            self.assertFalse((dest / "top.json").exists())
            self.assertFalse((dest / "chip_0/aic.bin").exists())
            self.assertFalse(validate_case(dest, inputs_only=True)["kernel_attached"])
            task = read(dest / "chip_0/mega_task.json")
            self.assertEqual(task["output_para_array"], [])
            self.assertNotIn("ast_output_array", task)
            self.assertEqual(
                [p["para_offset"] for p in task["input_para_array"]], list(range(24))
            )
            with self.assertRaisesRegex(ValueError, "disabled"):
                check_outputs(dest)


if __name__ == "__main__":
    unittest.main()
