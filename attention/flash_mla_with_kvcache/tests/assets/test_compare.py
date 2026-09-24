#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU regression tests for the MLA comparison and diagnostic branches."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "mla_compare_under_test", Path(__file__).parent / "impl" / "compare.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestCompare(unittest.TestCase):
    def run_lse(self, actual, expected):
        output = np.zeros(1, dtype=np.float32)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            results = MODULE.compare(output, actual, output, expected)
        self.assertTrue(results[0]["pass"])
        return results[1], stream.getvalue()

    def test_lse_values(self):
        cases = [
            ("equal", [1.0], [1.0], True),
            ("within_tolerance", [1.05], [1.0], True),
            ("outside_tolerance", [1.2], [1.0], False),
            ("positive_inf", [np.inf], [np.inf], True),
            ("negative_inf", [-np.inf], [-np.inf], True),
            ("opposite_inf", [np.inf], [-np.inf], False),
            ("finite_inf", [1.0], [np.inf], False),
            ("both_nan", [np.nan], [np.nan], True),
            ("one_nan", [np.nan], [1.0], False),
            ("empty", [], [], True),
        ]
        for name, actual, expected, passed in cases:
            with self.subTest(name=name):
                result, log = self.run_lse(
                    np.array(actual, dtype=np.float32),
                    np.array(expected, dtype=np.float32),
                )
                self.assertEqual(result["pass"], passed)
                self.assertIn("[Output 1] " + ("PASS:" if passed else "FAIL:"), log)
                if not passed:
                    self.assertEqual(result["diff_indices"], [0])
                    self.assertIn("Error Line:", log)

    def test_lse_shape_mismatch(self):
        for shape in [(2,), (1, 1), (0,)]:
            with self.subTest(shape=shape):
                result, log = self.run_lse(np.zeros(shape), np.zeros(1))
                self.assertFalse(result["pass"])
                self.assertIn("shape mismatch", result["error_info"])
                self.assertIn("shape mismatch", log)

    def test_lse_optional(self):
        result, log = self.run_lse(None, None)
        self.assertTrue(result["pass"])
        self.assertIn("SUPPRESSED", log)
        result, log = self.run_lse(None, np.zeros(1))
        self.assertFalse(result["pass"])
        self.assertIn("NO_OUTPUT", result["precision"])

    def test_lse_head_tail_pass_and_fail(self):
        for passed in [True, False]:
            with self.subTest(passed=passed):
                actual = np.zeros(30)
                if not passed:
                    actual[15] = 1
                result, log = self.run_lse(actual, np.zeros(30))
                self.assertEqual(result["pass"], passed)
                for index in [0, 9, 20, 29]:
                    self.assertIn(f"[Output 1] {index:08d}", log)
                if not passed:
                    self.assertEqual(result["diff_indices"], [15])

    def test_regular_outputs(self):
        for actual, expected, passed in [
            (np.zeros(30), np.zeros(30), True),
            (np.ones(30), np.zeros(30), False),
            (np.zeros(2), np.zeros(1), False),
            (np.empty(0), np.empty(0), True),
            (None, np.zeros(1), False),
            (None, None, True),
        ]:
            with self.subTest(actual=actual, expected=expected):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = MODULE.compare(actual, expected)[0]
                self.assertEqual(result["pass"], passed)

    def test_regular_pass_with_mismatch(self):
        actual = np.zeros(1000)
        actual[500] = 0.001
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = MODULE.compare(actual, np.zeros(1000))[0]
        self.assertTrue(result["pass"])
        self.assertEqual(result["diff_indices"], [500])
        self.assertIn("PASS: precision=99.9%", stream.getvalue())
        self.assertIn("00000500", stream.getvalue())

    def test_regular_failure_ratio_boundary(self):
        for count in [5, 6]:
            actual = np.zeros(1000)
            actual[:count] = 0.001
            with contextlib.redirect_stdout(io.StringIO()):
                result = MODULE.compare(actual, np.zeros(1000))[0]
            self.assertEqual(result["pass"], count == 5)

    def test_regular_nonfinite_policy(self):
        for actual, expected, passed in [
            (np.inf, np.inf, True),
            (-np.inf, -np.inf, True),
            (np.nan, np.nan, True),
            (np.inf, -np.inf, False),
            (np.nan, 0.0, False),
            (np.inf, 0.0, False),
        ]:
            with self.subTest(actual=actual, expected=expected):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    np.errstate(invalid="ignore", divide="ignore"),
                ):
                    result = MODULE.compare(np.array([actual]), np.array([expected]))[0]
                self.assertEqual(result["pass"], passed)

    def test_scalar_and_noncontiguous_lse(self):
        for actual in [np.array(1.0), np.arange(12).reshape(3, 4).T]:
            result, _ = self.run_lse(actual, actual.copy())
            self.assertTrue(result["pass"])

    def test_regular_sparse_nonfinite_mismatch(self):
        for actual, expected in [
            (np.nan, 0.0),
            (0.0, np.nan),
            (np.inf, 0.0),
            (0.0, -np.inf),
            (np.inf, -np.inf),
            (np.nan, np.inf),
        ]:
            with self.subTest(actual=actual, expected=expected):
                npu = np.zeros(1000, dtype=np.float32)
                golden = npu.copy()
                npu[7], golden[7] = actual, expected
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    np.errstate(invalid="ignore", divide="ignore"),
                ):
                    result = MODULE.compare(npu, golden)[0]
                self.assertFalse(result["pass"])
                self.assertEqual(result["precision"], 99.9)
                self.assertEqual(result["diff_indices"], [7])
                self.assertEqual(result["metrics"]["nonfinite_mismatch_count"], 1)
                self.assertIn("nonfinite_mismatches=1", result["error_info"])

    def test_invalid_argument_count(self):
        for outputs in [(), (np.zeros(1),), (None, None, None)]:
            with self.subTest(count=len(outputs)):
                self.assertFalse(MODULE.compare(*outputs)["pass"])


class TestFlashAttnCompare(unittest.TestCase):
    def setUp(self):
        global MODULE
        original = MODULE
        spec = importlib.util.spec_from_file_location(
            "flash_attn_compare_under_test",
            Path(__file__).resolve().parents[3]
            / "flash_attn/tests/assets/impl/compare.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MODULE = module
        self.addCleanup(self.restore_module, original)

    @staticmethod
    def restore_module(module):
        global MODULE
        MODULE = module

    test_regular_outputs = TestCompare.test_regular_outputs
    test_regular_pass_with_mismatch = TestCompare.test_regular_pass_with_mismatch
    test_regular_failure_ratio_boundary = (
        TestCompare.test_regular_failure_ratio_boundary
    )
    test_regular_nonfinite_policy = TestCompare.test_regular_nonfinite_policy
    test_regular_sparse_nonfinite_mismatch = (
        TestCompare.test_regular_sparse_nonfinite_mismatch
    )
    test_invalid_argument_count = TestCompare.test_invalid_argument_count


if __name__ == "__main__":
    unittest.main()
