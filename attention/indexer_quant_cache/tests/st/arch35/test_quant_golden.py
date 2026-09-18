#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU reference regressions; device coverage is in ttk_aclnn_normal_quant.csv."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "indexer_golden", Path(__file__).resolve().parents[2] / "assets" / "golden.py"
)
golden = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(golden)
golden._ensure_dtypes()


class QuantGoldenTest(unittest.TestCase):
    def normal(self, x, dtype, round_scale=False):
        cache = np.zeros((1, 1, 1, 32), dtype=dtype)
        scale = np.ones((1, 1, 1, 1), dtype=np.float32)
        return golden.IndexerQuantCacheTestSpec.golden(
            cache,
            scale,
            np.resize(x, (1, 32)),
            np.array([0], np.int32),
            quant_mode=1,
            round_scale=round_scale,
        )

    def test_mxfp8_nan_scale(self):
        q, scale = golden._encode_mxfp8_row(
            np.full(32, np.nan, np.float32), 32, golden.F8E4M3, np.float32(448), True
        )
        self.assertEqual(int(scale[0]), 255)
        self.assertTrue(np.isnan(q.astype(np.float32)).all())

    def test_normal_inf(self):
        q, scale = self.normal(np.array([np.inf, -np.inf], np.float32), golden.F8E5M2)
        np.testing.assert_array_equal(
            q.astype(np.float32).ravel()[:2], [np.inf, -np.inf]
        )
        self.assertEqual(float(scale.item()), 0)

    def test_normal_mixed_nonfinite(self):
        q, scale = self.normal(
            np.array([1, -1, np.inf, np.nan], np.float32), golden.F8E5M2
        )
        self.assertAlmostEqual(float(scale.item()), 1 / 57344)
        np.testing.assert_array_equal(
            q.astype(np.float32).ravel()[:3], [57344, -57344, np.inf]
        )
        self.assertTrue(np.isnan(q.astype(np.float32).ravel()[3]))

    def test_normal_uint8_hifloat_bytes(self):
        for round_scale in (False, True):
            q, scale = self.normal(np.array([1, -1], np.float32), np.uint8, round_scale)
            expected = (
                np.array([32768, -32768], np.float32).astype(golden.HIF8).view(np.uint8)
            )
            np.testing.assert_array_equal(q.ravel()[:2], expected)
            self.assertEqual(float(scale.item()), 1 / 32768)

    def test_normal_tiny_bfloat16(self):
        q, scale = self.normal(
            np.array([np.finfo(np.float32).tiny], dtype=golden.BF16), golden.F8E4M3
        )
        self.assertTrue((q.astype(np.float32) == 448).all())
        self.assertGreater(float(scale.item()), 0)

    def test_zero_normal_scale(self):
        for dtype in (golden.F8E4M3, golden.F8E5M2, np.uint8):
            for rounding in (False, True):
                q, scale = self.normal(np.zeros(32, np.float16), dtype, rounding)
                self.assertFalse(q.view(np.uint8).any())
                self.assertEqual(float(scale.item()), 0.0)

    def test_invalid_slots_preserve_outputs(self):
        cache = np.ones((1, 4, 1, 32), dtype=golden.F8E4M3)
        scale = np.ones((1, 4, 1, 1), np.float32)
        result = golden.IndexerQuantCacheTestSpec.golden(
            cache,
            scale,
            np.zeros((4, 32), np.float16),
            np.array([-1, -2, 4, 2147483647], np.int32),
        )
        np.testing.assert_array_equal(result[0], cache)
        np.testing.assert_array_equal(result[1], scale)

    def test_torch_carriers_match_native_reference(self):
        for mode in ("mxfp8", "fp8", "hifloat8", "mxfp4"):
            packed = mode == "mxfp4"
            mx = mode in ("mxfp8", "mxfp4")
            cache = np.full((1, 2, 1, 16 if packed else 32), 0x22, np.uint8)
            scale = np.ones((1, 2, 1, 1), np.uint8 if mx else np.float32)
            x = np.ones((2, 32), np.float16)
            slots = np.array([0, -1], np.int32)
            actual = golden.IndexerQuantCacheTorchTestSpec.golden(
                cache, scale, x, slots, mode
            )
            native_cache = cache
            if packed:
                native_cache = np.full((1, 2, 1, 32), 2, np.uint8).view(golden.FP4E2M1)
            elif mode == "mxfp8":
                native_cache = cache.view(golden.F8E4M3)
            expected = golden.IndexerQuantCacheTestSpec.golden(
                native_cache,
                scale.view(golden.F8E8M0) if mx else scale,
                x,
                slots,
                {"mxfp8": 0, "fp8": 1, "hifloat8": 2, "mxfp4": 3}[mode],
            )
            bits = expected[0].view(np.uint8)
            if packed:
                bits = bits[..., 0::2] | (bits[..., 1::2] << 4)
            np.testing.assert_array_equal(actual[0], bits)
            np.testing.assert_array_equal(
                actual[1].view(np.uint8), expected[1].view(np.uint8)
            )
            np.testing.assert_array_equal(actual[0][:, 1], cache[:, 1])

    def test_torch_mxfp8_e5m2(self):
        cache = np.zeros((1, 1, 1, 32), dtype=golden.F8E5M2)
        scale = np.ones((1, 1, 1, 1), dtype=golden.F8E8M0)
        args = (cache, scale, np.ones((1, 32), np.float16), np.array([0], np.int32))
        actual = golden.IndexerQuantCacheTorchTestSpec.golden(*args, quant_mode="mxfp8")
        expected = golden.IndexerQuantCacheTestSpec.golden(*args, quant_mode=0)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


if __name__ == "__main__":
    unittest.main()
