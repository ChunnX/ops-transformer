# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Boundary and incomplete-evidence checks for the one-percent release gate."""

import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from check_performance_gate import evaluate, load_details


def variant(durations):
    return {
        "run_count": len(durations),
        "mean_us": statistics.mean(durations),
        "runs": [
            {
                "mean_us": value,
                "rank_means_us": {str(rank): value for rank in range(16)},
                "samples_per_rank": {str(rank): 100 for rank in range(16)},
                "retained_per_rank": 50,
            }
            for value in durations
        ],
    }


def case(durations):
    return {
        "case_name": "dispatch_boundary",
        "operation": "dispatch",
        "baseline": variant([100.0] * len(durations)),
        "optimized": variant(durations),
    }


class PerformanceGateTest(unittest.TestCase):
    def test_inclusive_boundary_without_rounding(self):
        self.assertEqual(
            evaluate([case([101.0])], ["dispatch_boundary"])["status"], "passed"
        )
        self.assertEqual(
            evaluate([case([101.00001])], ["dispatch_boundary"])["status"], "failed"
        )

    def test_fast_repeat_cannot_hide_slow_repeat(self):
        report = evaluate([case([90.0, 102.0])], ["dispatch_boundary"])
        self.assertLess(report["cases"][0]["aggregate_regression_pct"], 0)
        self.assertEqual(report["status"], "failed")

    def test_fast_case_cannot_hide_slow_case(self):
        fast = case([50.0])
        fast["case_name"] = "dispatch_fast"
        report = evaluate([fast, case([102.0])], ["dispatch_fast", "dispatch_boundary"])
        self.assertEqual(report["failed_cases"], ["dispatch_boundary"])

    def test_missing_duplicate_and_extra_cases(self):
        row = case([99.0])
        for rows, expected in [
            ([], ["dispatch_boundary"]),
            ([row, row], ["dispatch_boundary"]),
            ([row], ["another_case"]),
            ([row], []),
        ]:
            with (
                self.subTest(rows=len(rows), expected=expected),
                self.assertRaises(ValueError),
            ):
                evaluate(rows, expected)

    def test_incomplete_rank_and_sample_evidence(self):
        for field in ["rank_means_us", "samples_per_rank"]:
            row = case([99.0])
            del row["optimized"]["runs"][0][field]["15"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate([row], ["dispatch_boundary"])
        row = case([99.0])
        row["optimized"]["runs"][0]["samples_per_rank"]["0"] = 99
        with self.assertRaises(ValueError):
            evaluate([row], ["dispatch_boundary"])

    def test_inconsistent_mean_and_unequal_repeats(self):
        row = case([99.0])
        row["optimized"]["mean_us"] = 98.0
        with self.assertRaises(ValueError):
            evaluate([row], ["dispatch_boundary"])
        row = case([99.0])
        row["optimized"] = variant([99.0, 99.0])
        with self.assertRaises(ValueError):
            evaluate([row], ["dispatch_boundary"])

    def test_limit_cannot_be_relaxed(self):
        for limit in [1.01, -1, float("nan")]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                evaluate([case([99.0])], ["dispatch_boundary"], limit)

    def test_partition_hash_and_record_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [case([99.0])]
            raw = json.dumps(records).encode()
            (root / "part.json").write_bytes(raw)
            metadata = {
                "format": "partitioned-records-v1",
                "records": 1,
                "parts": [
                    {
                        "path": "part.json",
                        "records": 1,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                ],
            }
            index = root / "index.json"
            index.write_text(json.dumps(metadata))
            self.assertEqual(load_details(index), records)
            wrong = copy.deepcopy(metadata)
            wrong["records"] = 2
            index.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                load_details(index)
            index.write_text(json.dumps(metadata))
            (root / "part.json").write_bytes(raw + b"\n")
            with self.assertRaises(ValueError):
                load_details(index)


if __name__ == "__main__":
    unittest.main()
