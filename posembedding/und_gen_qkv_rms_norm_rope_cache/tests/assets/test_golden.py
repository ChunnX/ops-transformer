#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""CPU 回归：python -m unittest discover -s <本目录> -p test_golden.py。"""

import ctypes
import ctypes.util
import unittest

import numpy as np
import torch

from golden import _fma_f32, _isclose_ratio, _rope


class GoldenFmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
        cls.fmaf = cls.libm.fmaf
        cls.fmaf.argtypes = [ctypes.c_float] * 3
        cls.fmaf.restype = ctypes.c_float

    def assert_fmaf_equal(self, a, b, c):
        tensors = [torch.as_tensor(v, dtype=torch.float32) for v in (a, b, c)]
        actual = _fma_f32(*tensors).numpy()
        broadcast = np.broadcast_arrays(*(v.numpy() for v in tensors))
        expected = np.array(
            [
                self.fmaf(float(x), float(y), float(z))
                for x, y, z in zip(*(v.flat for v in broadcast))
            ],
            dtype=np.float32,
        ).reshape(actual.shape)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite_or_inf = ~np.isnan(expected)
        # 包括 Inf 符号与 +/-0；有限结果必须逐 bit 一致，而非宽松 isclose。
        np.testing.assert_array_equal(
            actual.view(np.uint32)[finite_or_inf],
            expected.view(np.uint32)[finite_or_inf],
        )

    def test_special_values_and_subnormals(self):
        values = np.array(
            [
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                np.finfo(np.float32).tiny,
                2.0**-149,
                -(2.0**-149),
                1.0,
                -1.0,
            ],
            dtype=np.float32,
        )
        self.assert_fmaf_equal(
            values[:, None, None], values[None, :, None], values[None, None, :]
        )

    def test_double_rounding_midpoint(self):
        # 乘积恰为 FP32 中点，极小加数会被普通 FP64 加法丢掉。
        a = np.float32(1 + 2.0**-23)
        b = np.float32(1.5)
        c = np.float32(-(2.0**-100))
        naive = np.float32(np.float64(a) * np.float64(b) + np.float64(c))
        self.assertNotEqual(float(naive), self.fmaf(float(a), float(b), float(c)))
        self.assert_fmaf_equal([a, -a, a, -a], b, [c, -c, -c, c])

    def test_random_bits_across_chunk_boundary(self):
        rng = np.random.default_rng(20260907)
        values = rng.integers(0, 2**32, size=(3, 65539), dtype=np.uint32)
        self.assert_fmaf_equal(*values.view(np.float32))

    def test_rope_overflow_and_cancellation(self):
        # 覆盖 025/026 的极大 cos/sin 以及 057 的极大归一化结果。
        maximum = torch.finfo(torch.float32).max
        x = torch.tensor(
            [[[2.0, -2.0, 1.5, -1.5], [maximum, maximum, maximum / 2, maximum / 2]]]
        )
        cos = torch.tensor([[maximum, -1.5]])
        sin = torch.tensor([[maximum, -1.5]])
        actual = _rope(x, cos, sin)
        expected = torch.empty_like(actual)
        for head in range(2):
            for col in range(2):
                lo, hi = x[0, head, col], x[0, head, col + 2]
                c, s = cos[0, col], sin[0, col]
                expected[0, head, col] = self.fmaf(float(hi), float(-s), float(lo * c))
                expected[0, head, col + 2] = self.fmaf(
                    float(lo), float(s), float(hi * c)
                )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        split_low = x[..., :2] * cos - x[..., 2:] * sin
        self.assertTrue(torch.isnan(split_low[0, 0, 0]))
        self.assertTrue(torch.isposinf(actual[0, 0, 0]))

    def test_comparator_still_rejects_special_value_mismatch(self):
        pairs = [
            (float("inf"), torch.finfo(torch.bfloat16).max),
            (float("inf"), -float("inf")),
            (float("nan"), float("inf")),
            (1.0, 2.0),
        ]
        for actual, expected in pairs:
            with self.subTest(actual=actual, expected=expected):
                result = _isclose_ratio(
                    torch.tensor([actual], dtype=torch.bfloat16),
                    torch.tensor([expected], dtype=torch.bfloat16),
                )
                self.assertFalse(result[0]["pass"])


if __name__ == "__main__":
    unittest.main()
