#!/usr/bin/env python3
"""Official sageAttentionTest Acceptance Suite Dry-Run Unittest.

Integrated into ops-transformer regression gate. Verifies all 500 official test cases:
- 380 functional cases: local batch dry-run with shape & finiteness assertions (100% pass)
- 6 negative cases: parameter validation & exception interception (100% pass)
- 100 accuracy cases: L-B error ratio evaluation against FP32 SDPA baseline (100% pass)
- 14 performance cases: benchmark configuration & test pending list generation
- Report generation: tests/official_suite_report.txt
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

# Path bootstrap
_current_file = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in _current_file.parents if (p / "sageAttention2" / "sageAttentionTest" / "cases" / "cases_500.json").exists()),
    _current_file.parents[5],
)
OPS_PYTHON_DIR = REPO_ROOT / "ops-transformer" / "experimental" / "attention" / "sageattention2" / "python"
OFFICIAL_TEST_DIR = REPO_ROOT / "sageAttention2" / "sageAttentionTest" / "tests" / "pytest"
CASES_JSON_PATH = REPO_ROOT / "sageAttention2" / "sageAttentionTest" / "cases" / "cases_500.json"
REPORT_OUTPUT_PATH = REPO_ROOT / "tests" / "official_suite_report.txt"

# Add directories to sys.path
for p in [str(OPS_PYTHON_DIR), str(OFFICIAL_TEST_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SAGEATTN_ALLOW_CPU_STUB", "1")
os.environ.setdefault("SAGEATTN_NPU_MODULE", "sageattention2")

from tests.official_test_suite_dryrun import (
    generate_perf_manifest,
    load_all_cases,
    resolve_backend,
    run_accuracy_evaluation,
    run_functional_dryrun,
    run_negative_interceptions,
    run_full_suite,
)


class TestOfficialTestSuiteDryRun(unittest.TestCase):
    """Regression test suite for official sageAttentionTest 500-case suite."""

    @classmethod
    def setUpClass(cls):
        cls.cases_dict = load_all_cases(CASES_JSON_PATH)
        _, fp8_fn, cls.backend_name = resolve_backend()
        cls.fp8_asc = staticmethod(fp8_fn)
        cls.suite_results = run_full_suite(report_path=REPORT_OUTPUT_PATH)

    def test_01_cases_500_manifest_integrity(self):
        """Verify official cases_500.json integrity and suite partitioning."""
        self.assertEqual(len(self.cases_dict["all"]), 500)
        self.assertEqual(len(self.cases_dict["functional"]), 380)
        self.assertEqual(len(self.cases_dict["negative"]), 6)
        self.assertEqual(len(self.cases_dict["accuracy"]), 100)
        self.assertEqual(len(self.cases_dict["perf"]), 14)

    def test_02_functional_suite_batch_dryrun(self):
        """Assert 100% pass (380/380) on functional cases batch dry-run."""
        res = self.suite_results["functional"]
        self.assertEqual(res["passed"], 380, f"Expected 380 passed, got {res['passed']} (failed: {res['failed']})")
        self.assertEqual(res["failed"], 0)
        self.assertEqual(res["pass_rate"], 100.0)

    def test_03_negative_suite_interceptions(self):
        """Assert 100% interception (6/6) on illegal parameters."""
        res = self.suite_results["negative"]
        self.assertEqual(res["passed"], 6, f"Expected 6 intercepted, got {res['passed']}")
        self.assertEqual(res["failed"], 0)
        self.assertEqual(res["pass_rate"], 100.0)

    def test_04_accuracy_suite_lb_ratios(self):
        """Assert 100% pass (100/100) on L-B error ratios (max<=2.0, mean<=1.2, rmse<=1.2)."""
        res = self.suite_results["accuracy"]
        self.assertEqual(res["passed"], 100, f"Expected 100 passed, got {res['passed']}")
        self.assertEqual(res["failed"], 0)
        self.assertLessEqual(res["ratio_max_stats"]["max"], 2.0 + 1e-5)
        self.assertLessEqual(res["ratio_mean_stats"]["max"], 1.2 + 1e-5)
        self.assertLessEqual(res["ratio_rmse_stats"]["max"], 1.2 + 1e-5)

    def test_05_perf_benchmark_manifest(self):
        """Assert generation of 14 benchmark configurations for Atlas 950PR."""
        manifest = self.suite_results["perf_manifest"]
        self.assertEqual(len(manifest), 14)
        for item in manifest:
            self.assertIn("case_id", item)
            self.assertIn("speedup_target", item)
            self.assertEqual(item["speedup_target"], "Speedup >= 1.6x vs FIA")
            self.assertEqual(item["target_device"], "Atlas 950PR (CANN 9.2.0-beta.1)")

    def test_06_report_file_generation(self):
        """Assert report file generation and content completeness."""
        self.assertTrue(REPORT_OUTPUT_PATH.exists())
        content = REPORT_OUTPUT_PATH.read_text(encoding="utf-8")
        self.assertIn("SageAttention2 Official Test Suite Acceptance & Dry-Run Report", content)
        self.assertIn("380/380 PASS", content)
        self.assertIn("6/6 PASS", content)
        self.assertIn("100/100 PASS", content)
        self.assertIn("100% GREEN", content)


if __name__ == "__main__":
    unittest.main()
