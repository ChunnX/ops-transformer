#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Golden output regression test for perf model refactoring.

Captures CLI output from a representative set of cases as a baseline
("golden") snapshot, then verifies that future runs produce byte-identical
output. Used as the safety net for the perf module refactoring.

Workflow:
    1. First run: captures golden snapshots into test_golden/.
    2. Subsequent runs: compares actual output against the golden files.
    3. On mismatch: writes a .diff file next to the .txt for inspection,
       and exits with error.

Usage:
    python test_regression.py            # run all cases
    python test_regression.py --reset    # delete golden snapshots and re-capture
    python test_regression.py --verbose  # print pass/fail per case
"""

import argparse
import subprocess
import sys
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = PERF_DIR / "test_golden"
ANALYZER = PERF_DIR / "perf_analyzer.py"

# Representative test matrix - exercises all numerical hot paths.
# Each tuple is (case_name, [CLI args]).
GOLDEN_CASES = [
    # Decode scenarios across quantization modes
    ("decode_bs1_bf16_full",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--mode", "full"]),
    ("decode_bs8_bf16_full",
        ["--chip", "950", "--batch-size", "8", "--head-num", "128",
         "--mode", "full"]),
    ("decode_bs1_int8_full",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--weight-quant-mode", "2", "--kv-cache-quant-mode", "1",
         "--mode", "full"]),
    ("decode_bs1_fp8_full",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--weight-quant-mode", "3", "--mode", "full"]),

    # Prefill scenarios
    ("prefill_t128_bf16_full",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "128",
         "--head-num", "128", "--mode", "full"]),
    ("prefill_t512_bf16_full",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "512",
         "--head-num", "128", "--mode", "full"]),

    # Edge cases
    ("decode_bs1_n8_split_kn",
        ["--chip", "950", "--batch-size", "1", "--head-num", "8",
         "--mode", "split-kn"]),
    ("small_model_n16_he2048",
        ["--chip", "950", "--batch-size", "1", "--head-num", "16",
         "--He", "2048", "--mode", "full"]),

    # Legacy 910B chip-total path
    ("legacy_910b_bound",
        ["--batch-size", "1", "--head-num", "128", "--mode", "bound"]),

    # Cross-testcase report (exercises mode_report formatting)
    ("report_all_950",
        ["--chip", "950", "--mode", "report"]),

    # Per-mode focused tests on a single config
    ("decode_bs1_bf16_pipeline",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--mode", "pipeline"]),
    ("decode_bs1_bf16_roofline",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--mode", "roofline"]),
    ("decode_bs1_bf16_block_search",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--mode", "block-search"]),
    ("decode_bs1_bf16_split_kn",
        ["--chip", "950", "--batch-size", "1", "--head-num", "128",
         "--mode", "split-kn"]),
    ("prefill_t256_split_m",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "256",
         "--head-num", "128", "--mode", "split-m"]),
    ("prefill_t256_prefill_mode",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "256",
         "--head-num", "128", "--mode", "prefill"]),

    # dequantOpt UB-pipelined dequant+rope (Phase 2 of perf-modeling refactor)
    ("decode_bs1_fp8_dequant_opt",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "1",
         "--head-num", "128", "--weight-quant-mode", "3",
         "--mode", "dequant-opt"]),
    ("decode_bs8_int8_dequant_opt",
        ["--chip", "950", "--batch-size", "8", "--seq-len", "1",
         "--head-num", "128", "--weight-quant-mode", "2",
         "--kv-cache-quant-mode", "1", "--mode", "dequant-opt"]),
    ("decode_bs1_int8_dequant_opt",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "1",
         "--head-num", "128", "--weight-quant-mode", "2",
         "--kv-cache-quant-mode", "1", "--mode", "dequant-opt"]),
    ("decode_bs1_bf16_dequant_opt",
        ["--chip", "950", "--batch-size", "1", "--seq-len", "1",
         "--head-num", "128", "--mode", "dequant-opt"]),
]


def run_perf(args):
    """Run perf_analyzer.py with the given args and return stdout."""
    result = subprocess.run(
        [sys.executable, str(ANALYZER)] + args,
        capture_output=True, text=True, cwd=str(PERF_DIR),
        timeout=300,
    )
    if result.returncode != 0:
        return f"<ERROR exit={result.returncode}>\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result.stdout


def normalize(text):
    """Strip non-deterministic parts. Numerical output MUST be preserved."""
    # Currently exact match. If timing/path differences appear, normalize here.
    return text


def main():
    parser = argparse.ArgumentParser(description="Perf model regression test")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing golden snapshots and re-capture")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print pass/fail per case")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run cases whose name contains this substring")
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(exist_ok=True)

    if args.reset:
        for f in GOLDEN_DIR.glob("*.txt"):
            f.unlink()
        for f in GOLDEN_DIR.glob("*.diff"):
            f.unlink()
        print(f"Reset: cleared {GOLDEN_DIR}")

    cases = GOLDEN_CASES
    if args.filter:
        cases = [c for c in cases if args.filter in c[0]]
        if not cases:
            print(f"No cases match filter {args.filter!r}")
            return 1

    captured = []
    passed = []
    failed = []

    for name, cli_args in cases:
        if args.verbose:
            print(f"Running {name} ... ", end="", flush=True)
        actual = normalize(run_perf(cli_args))
        golden_path = GOLDEN_DIR / f"{name}.txt"
        diff_path = GOLDEN_DIR / f"{name}.diff"

        if not golden_path.exists():
            golden_path.write_text(actual)
            captured.append(name)
            if args.verbose:
                print("CAPTURED")
            continue

        expected = golden_path.read_text()
        if actual == expected:
            passed.append(name)
            if diff_path.exists():
                diff_path.unlink()
            if args.verbose:
                print("PASS")
        else:
            failed.append(name)
            diff_path.write_text(
                f"=== EXPECTED ({len(expected)} chars) ===\n"
                f"{expected}\n"
                f"=== ACTUAL ({len(actual)} chars) ===\n"
                f"{actual}\n"
            )
            if args.verbose:
                print(f"FAIL (diff in {diff_path.name})")

    print()
    print(f"Captured (new): {len(captured)}  Passed: {len(passed)}  Failed: {len(failed)}")
    if captured:
        print(f"  New baselines: {', '.join(captured)}")
    if failed:
        print(f"  Failures: {', '.join(failed)}")
        print(f"  See {GOLDEN_DIR}/*.diff for details")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
