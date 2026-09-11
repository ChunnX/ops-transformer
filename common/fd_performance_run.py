"""Run bounded groups of paramset cases through the existing single runners."""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common.fd_performance_cases import iter_fd_cases
from common.fd_performance_report import append_report, make_report, read_csv

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = {"smla": "sparse_flash_mla", "qsmla": "quant_sparse_flash_mla",
             "mqsmla": "mixed_quant_sparse_flash_mla"}


def selected_cases(args):
    cases = iter_fd_cases(phase=args.phase)
    if args.suite:
        cases = (case for case in cases if case["suite"] in args.suite)
    stop = None if args.limit is None else args.start + args.limit
    return itertools.islice(cases, args.start, stop)


def check_pytest_report(path, expected):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    failed = sum(int(suite.get("failures", 0)) + int(suite.get("errors", 0)) for suite in suites)
    if tests != expected or failed:
        raise ValueError(f"pytest: {tests}/{expected} cases, {failed} failures/errors")


def run_chunk(operator, manifest, chunk_dir, args, output):
    chunk_dir.mkdir(parents=True)
    result = chunk_dir / "execution.csv"
    junit = chunk_dir / "pytest.xml"
    name = OPERATORS[operator]
    env = dict(os.environ, FD_PERF_MODE="1", FD_CASE_MANIFEST=str(manifest),
               FD_RESULT_CSV=str(result), FD_WARMUP=str(args.warmup), FD_REPEAT=str(args.repeat),
               FD_DEVICE_ID=str(args.device), RUN_GRAPH="0",
               **{operator.upper() + "_BATCH_CONSISTENCY": "off"})
    command = [args.msprof, "--output=" + str(chunk_dir),
               sys.executable, "-m", "pytest", "-q", "-s", "-m", "ci",
               "test_" + name + "_single.py", "--junitxml=" + str(junit)]
    with (chunk_dir / "run.log").open("w", encoding="utf-8") as log:
        finished = subprocess.run(command, cwd=ROOT / "attention" / name / "tests/pytest",
                                  env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=args.timeout or None)
    rows = read_csv(result) if result.exists() else []
    expected = len(json.loads(manifest.read_text())) * (1 if operator == "qsmla" else 2)
    summaries = sorted(chunk_dir.rglob("op_summary*.csv"))
    records = [record for summary in summaries for record in read_csv(summary)]
    try:
        if len(rows) != expected:
            raise ValueError(f"{len(rows)}/{expected} execution rows: check {chunk_dir / 'run.log'}")
        report = make_report(rows, records, operator)
        append_report(output / (operator + "_perf.csv"), report)
        append_report(output / "all_perf.csv", report)
        check_pytest_report(junit, expected)
        if finished.returncode:
            raise ValueError(f"msprof exited with status {finished.returncode}")
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"{operator} FAILED: {error}", file=sys.stderr, flush=True)
        # Preserve every execution status even when profiler pairing fails.
        if rows:
            fallback = []
            for row in rows:
                fallback.append(dict(row, metadata_median_us="", median_us="",
                                     p90_us="", tiling_key=""))
            append_report(output / (operator + "_perf.csv"), fallback)
            append_report(output / "all_perf.csv", fallback)
        return False
    print(f"{operator}: {len(rows)} rows -> {output / (operator + '_perf.csv')}", flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("all", *OPERATORS), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=int(os.environ.get("FD_CASE_START", "0")))
    parser.add_argument("--limit", type=int, default=int(os.environ["FD_CASE_LIMIT"])
                        if os.environ.get("FD_CASE_LIMIT") else None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=0, help="Timeout per profiler process in seconds")
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--suite", action="append", help="Explicitly select a matrix suite")
    parser.add_argument("--dry-run", action="store_true", help="Count selected workloads without torch or NPU")
    parser.add_argument("--msprof", default=os.environ.get("MSPROF_BIN", "msprof"))
    args = parser.parse_args(argv)
    if args.start < 0 or (args.limit is not None and args.limit < 1) or args.chunk_size < 1:
        parser.error("start must be >= 0; limit and chunk-size must be > 0")
    if args.warmup < 0 or args.repeat < 1 or args.timeout < 0:
        parser.error("warmup/timeout must be >= 0; repeat must be > 0")
    operators = tuple(OPERATORS) if args.op == "all" else (args.op,)
    if args.dry_run:
        count = sum(1 for _ in selected_cases(args))
        print(json.dumps({op: count * (1 if op == "qsmla" else 2) for op in operators}))
        return 0 if count else 1
    if not shutil.which(args.msprof):
        parser.error(f"msprof not found: {args.msprof}; source the CANN environment first")
    output = (args.output_dir or ROOT / "perf_results" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"Output directory must be empty: {output}")
    (output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    cases, failed, index = iter(selected_cases(args)), False, 0
    while batch := list(itertools.islice(cases, args.chunk_size)):
        manifest = output / f"cases-{index:05d}.json"
        manifest.write_text(json.dumps(batch), encoding="utf-8")
        for op in operators:
            try:
                ok = run_chunk(op, manifest, output / f"{op}-{index:05d}", args, output)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"{op} FAILED: {error}", file=sys.stderr, flush=True)
                ok = False
            failed = failed or not ok
        index += 1
    if not index:
        print("No cases selected", file=sys.stderr)
        return 1
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
