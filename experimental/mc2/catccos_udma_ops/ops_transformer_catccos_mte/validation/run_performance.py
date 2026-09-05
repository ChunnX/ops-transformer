# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Collect and batch-report on-board operator duration with msprof."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .case_schema import load_case_manifest, performance_workload_digest, select_cases, write_resolved_cases
    from .common import environment_fingerprint, timestamp, write_csv, write_json
    from .msprof_parser import parse_msprof_case
except ImportError:
    from case_schema import load_case_manifest, performance_workload_digest, select_cases, write_resolved_cases
    from common import environment_fingerprint, timestamp, write_csv, write_json
    from msprof_parser import parse_msprof_case


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"


def _task_time_value(msprof, override):
    if override:
        return override
    help_result = subprocess.run([msprof, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return "l1" if "l1" in help_result.stdout.lower() and "task-time" in help_result.stdout.lower() else "on"


def _load_baseline(path):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in payload.get("cases", [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-tag", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--baseline", help="summary.json from the original or a previous build")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--task-time", choices=("on", "l1"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    cases = select_cases(
        load_case_manifest(args.cases), mode="performance",
        case_ids=args.case_id, include_tags=args.include_tag,
    )
    msprof = shutil.which("msprof")
    if not msprof:
        raise RuntimeError("msprof is not on PATH; source the CANN environment first")
    task_time = _task_time_value(msprof, args.task_time)
    output_root = Path(args.output_dir or (HERE.parent / "validation_results" / timestamp() / "performance")).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    environment = environment_fingerprint()
    write_json(output_root / "environment.json", environment)
    baselines = _load_baseline(args.baseline)
    results = []
    for index, case in enumerate(cases):
        case_root = output_root / case["id"]
        profile_root = case_root / "msprof"
        profile_root.mkdir(parents=True, exist_ok=True)
        resolved = case_root / "resolved_case.json"
        write_resolved_cases(resolved, [case])
        process_environment = os.environ.copy()
        process_environment.setdefault("MTE_MASTER_ADDR", "127.0.0.1")
        rendezvous_port = int(
            process_environment.get(
                "VALIDATION_MASTER_PORT", process_environment.get("MASTER_PORT", "29500")
            )
        ) + index
        mte_port = int(process_environment.get("MTE_MASTER_PORT", "29601")) + index
        if rendezvous_port == mte_port:
            raise RuntimeError("torchrun and MTE bootstrap ports must differ")
        process_environment["MTE_MASTER_PORT"] = str(mte_port)
        command = [
            msprof, "--output={}".format(profile_root), "--task-time={}".format(task_time),
            "--ascendcl=off", "--runtime-api=off", "--aicpu=off", "--ai-core=off",
            sys.executable,
            "-m", "torch.distributed.run",
            "--master-addr=127.0.0.1",
            "--master-port={}".format(rendezvous_port),
            "--nproc-per-node={}".format(case["world_size"]),
            str(HERE / "performance_worker.py"), "--resolved-case", str(resolved),
            "--output-dir", str(case_root),
        ]
        print("+ " + " ".join(command), flush=True)
        run = subprocess.run(command, cwd=str(output_root), env=process_environment)
        if run.returncode:
            result = {
                "case_id": case["id"], "valid": False, "passed": False,
                "errors": ["msprof/worker exit code {}".format(run.returncode)],
            }
        else:
            try:
                if not list(profile_root.rglob("op_summary_*.csv")):
                    export_command = [
                        msprof, "--export=on", "--output={}".format(profile_root),
                        "--type=text", "--summary-format=csv",
                    ]
                    print("+ " + " ".join(export_command), flush=True)
                    subprocess.check_call(export_command, cwd=str(output_root))
                result = parse_msprof_case(profile_root, case)
            except Exception as error:
                result = {"case_id": case["id"], "valid": False, "passed": False, "errors": ["profile parse failed: {}".format(error)]}
        result["performance_workload_digest"] = performance_workload_digest(case)
        result["environment"] = environment
        baseline = baselines.get(case["id"])
        if args.baseline and baseline is None and result.get("valid"):
            result["passed"] = False
            result.setdefault("errors", []).append("baseline has no matching case")
        if baseline and result.get("valid"):
            result["baseline_latency_us"] = baseline.get("latency_us", {}).get("p50")
            if baseline.get("performance_workload_digest") != result["performance_workload_digest"]:
                result["passed"] = False
                result.setdefault("errors", []).append("baseline performance workload digest does not match")
            elif not baseline.get("valid", True):
                result["passed"] = False
                result.setdefault("errors", []).append("baseline is marked invalid")
            elif result["baseline_latency_us"] is None:
                result["passed"] = False
                result.setdefault("errors", []).append("baseline has no p50 latency")
            else:
                comparable_keys = ("soc_version", "cann_version", "torch", "torch_npu")
                mismatches = {
                    key: {"baseline": baseline.get("environment", {}).get(key), "current": environment.get(key)}
                    for key in comparable_keys
                    if baseline.get("environment", {}).get(key) != environment.get(key)
                }
                if mismatches:
                    result["environment_mismatches"] = mismatches
                    if not args.allow_environment_mismatch:
                        result["passed"] = False
                        result.setdefault("errors", []).append(
                            "baseline environment differs; use --allow-environment-mismatch only after review"
                        )
                regression = result["latency_us"]["p50"] / result["baseline_latency_us"] - 1.0
                result["regression_ratio"] = regression
                result["regression_limit"] = case["performance"]["max_regression_ratio"]
                if regression > result["regression_limit"]:
                    result["passed"] = False
                    result.setdefault("errors", []).append(
                        "latency regression {:.2%} exceeds {:.2%}".format(regression, result["regression_limit"])
                    )
        write_json(case_root / "report.json", result)
        results.append(result)
        if not result.get("passed") and args.fail_fast:
            break

    rows = [{
        "case_id": item["case_id"], "passed": item.get("passed", False),
        "valid": item.get("valid", False),
        "p50_us": item.get("latency_us", {}).get("p50"),
        "p90_us": item.get("latency_us", {}).get("p90"),
        "skew_p90": item.get("rank_skew_ratio", {}).get("p90"),
        "regression_ratio": item.get("regression_ratio"),
        "errors": "; ".join(item.get("errors", [])),
    } for item in results]
    write_csv(output_root / "summary.csv", rows, [
        "case_id", "passed", "valid", "p50_us", "p90_us", "skew_p90",
        "regression_ratio", "errors",
    ])
    summary = {
        "passed": len(results) == len(cases) and all(item.get("passed") for item in results),
        "environment": environment,
        "cases": results,
    }
    write_json(output_root / "summary.json", summary)
    print("performance results: {}".format(output_root), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
