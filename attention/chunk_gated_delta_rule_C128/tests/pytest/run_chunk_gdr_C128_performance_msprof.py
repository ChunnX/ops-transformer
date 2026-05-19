#!/usr/bin/env python3
"""
Batch msprof launcher for ChunkGDR normal vs C128 prefill performance cases.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from compare_chunk_gdr_C128_performance_paramset import PERFORMANCE_CASES, case_id
from summarize_chunk_gdr_C128_performance_msprof import summarize

SCRIPT_DIR = Path(__file__).resolve().parent
PAYLOAD_SCRIPT = SCRIPT_DIR / "compare_chunk_gdr_C128_performance.py"


def build_application(case, ops, args):
    command = [
        sys.executable,
        str(PAYLOAD_SCRIPT),
        "--ops",
        ops,
        "--g-mode",
        case["g_mode"],
        "--device",
        str(args.device),
        "--B",
        str(case["batch_size"]),
        "--S",
        str(case["seqlen"]),
        "--D",
        str(case["dk"]),
        "--Hq",
        str(case["nv"]),
        "--Hkv",
        str(case["nk"]),
        "--seed",
        str(case["seed"]),
        "--flush-mm",
        str(args.flush_mm),
        "--flush-reduce-outer",
        str(args.flush_reduce_outer),
        "--flush-reduce-mid",
        str(args.flush_reduce_mid),
        "--flush-reduce-inner",
        str(args.flush_reduce_inner),
    ]
    return " ".join(shlex.quote(item) for item in command)


def run_msprof(case, ops, args):
    output_dir = Path(args.output) / case_id(case) / ops
    command = [
        "msprof",
        f"--application={build_application(case, ops, args)}",
        f"--output={output_dir}",
        "--task-time=on",
    ]
    if args.extra_msprof_args:
        command.extend(shlex.split(args.extra_msprof_args))
    print("[MSPROF] " + " ".join(shlex.quote(item) for item in command), flush=True)
    if args.dry_run:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, check=False).returncode


def selected_cases(args):
    cases = PERFORMANCE_CASES
    if args.case:
        names = set(args.case)
        cases = [case for case in cases if case_id(case) in names]
        missing = names - {case_id(case) for case in cases}
        if missing:
            raise ValueError(f"Unknown performance cases: {sorted(missing)}")
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def main():
    parser = argparse.ArgumentParser(description="Run msprof for ChunkGDR normal/C128 prefill cases.")
    parser.add_argument("--output", default="./prof_out/c128_b8_s4096_n64_20260514", help="msprof output root.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ops", choices=["normal", "c128", "both"], default="both")
    parser.add_argument("--case", action="append", help="Run only the named case. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected cases.")
    parser.add_argument("--dry-run", action="store_true", help="Print msprof commands without running them.")
    parser.add_argument("--no-summary", action="store_true", help="Do not summarize msprof outputs after running.")
    parser.add_argument("--extra-msprof-args", default="", help="Extra arguments appended to each msprof command.")
    parser.add_argument("--flush-mm", type=int, default=10240)
    parser.add_argument("--flush-reduce-outer", type=int, default=96)
    parser.add_argument("--flush-reduce-mid", type=int, default=1024)
    parser.add_argument("--flush-reduce-inner", type=int, default=1024)
    args = parser.parse_args()

    ops_list = ["normal", "c128"] if args.ops == "both" else [args.ops]
    failures = []
    for case in selected_cases(args):
        for ops in ops_list:
            ret = run_msprof(case, ops, args)
            if ret != 0:
                failures.append((case_id(case), ops, ret))

    if failures:
        for name, ops, ret in failures:
            print(f"[ERROR] case={name} ops={ops} ret={ret}", file=sys.stderr)
        return 1
    if not args.dry_run and not args.no_summary:
        raw_path, compare_path, raw_rows, compare_rows = summarize(args.output)
        print(f"[SUMMARY] raw rows={len(raw_rows)} -> {raw_path}")
        print(f"[SUMMARY] compare rows={len(compare_rows)} -> {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
