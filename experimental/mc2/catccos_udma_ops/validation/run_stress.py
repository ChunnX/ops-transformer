# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Run balanced UDMA stress cases with resident inputs and exact launch counts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from .case_schema import load_case_manifest, select_cases, write_resolved_cases
    from .common import environment_fingerprint, timestamp, write_json
    from .runtime_log_collection import collect_runtime_logs, runtime_log_roots, snapshot_runtime_logs
    from .stress_worker import (
        STRESS_MODES, positive_iterations, positive_timeout_seconds,
        validate_stress_case, validate_stress_environment,
    )
except ImportError:
    from case_schema import load_case_manifest, select_cases, write_resolved_cases
    from common import environment_fingerprint, timestamp, write_json
    from runtime_log_collection import collect_runtime_logs, runtime_log_roots, snapshot_runtime_logs
    from stress_worker import (
        STRESS_MODES, positive_iterations, positive_timeout_seconds,
        validate_stress_case, validate_stress_environment,
    )


HERE = Path(__file__).resolve().parent


def _terminate_process_tree(process):
    if os.name == "posix":
        # The launcher owns this session, including all torchrun workers.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Even if torchrun exited on TERM, a worker may still be stuck draining.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    process.wait(timeout=10)


def run_command(command, environment, case_root, timeout_seconds):
    """Enforce a host deadline even when a worker is blocked inside a launch."""
    log_path = case_root / "torchrun.log"
    started = time.perf_counter()
    timed_out = False
    process = subprocess.Popen(
        command, cwd=str(case_root), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=(os.name == "posix"),
    )

    def pump_output():
        with log_path.open("w", encoding="utf-8") as stream:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()

    pump = threading.Thread(target=pump_output, name="stress-output", daemon=True)
    pump.start()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
    except BaseException:
        _terminate_process_tree(process)
        raise
    finally:
        pump.join(timeout=10)
    return {
        "returncode": process.returncode, "timed_out": timed_out,
        "process_seconds": time.perf_counter() - started,
        "console_log": str(log_path),
    }


def summarize_case(
    case, mode, iterations, case_root, process_result,
):
    errors = []
    if process_result["timed_out"]:
        errors.append("torchrun exceeded the case timeout")
    if process_result["returncode"] != 0:
        errors.append("torchrun exit code {}".format(process_result["returncode"]))
    names = ("a2avgmm", "gmma2av") if mode == "chain" else (mode,)
    expected_calls = {name: iterations for name in names}
    ranks = []
    for rank in range(case["world_size"]):
        path = case_root / "worker_rank_{}.json".format(rank)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            errors.append("rank {} report unavailable: {}".format(rank, error))
            continue
        ranks.append(report)
        if (
            report.get("case_id") != case["id"] or report.get("rank") != rank
            or report.get("mode") != mode or not report.get("passed")
            or report.get("completed_iterations") != iterations
            or report.get("completed_calls") != expected_calls
        ):
            errors.append("rank {} did not confirm completion of the requested workload".format(rank))
    elapsed = [item["elapsed_seconds"] for item in ranks if "elapsed_seconds" in item]
    return {
        "case_id": case["id"], "mode": mode, "iterations": iterations,
        "requested_calls_per_rank": expected_calls,
        "passed": not errors, "errors": errors, "ranks": ranks,
        "max_rank_elapsed_seconds": max(elapsed) if elapsed else None,
        **process_result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=STRESS_MODES, required=True)
    parser.add_argument("--iterations", type=positive_iterations, required=True)
    parser.add_argument("--cases", help="YAML manifest; defaults to the selected operator's existing cases")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-tag", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout-seconds", type=positive_timeout_seconds, default=3600.0,
                        help="per-case deadline including initialization and teardown (default: 3600)")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    validate_stress_environment(os.environ)
    default_manifest = (
        "grouped_mat_mul_all_to_allv.yaml" if args.mode == "gmma2av"
        else "alltoallv_grouped_mat_mul.yaml"
    )
    cases = select_cases(
        load_case_manifest(args.cases or HERE / "cases" / default_manifest),
        mode=None, case_ids=args.case_id, include_tags=args.include_tag,
    )
    for case in cases:
        validate_stress_case(case, args.mode)
    output_root = Path(args.output_dir or HERE.parent / "validation_results" / timestamp() / "stress").resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    environment = environment_fingerprint()
    write_json(output_root / "environment.json", environment)
    results = []
    for index, case in enumerate(cases):
        case_root = output_root / case["id"]
        resolved = case_root / "resolved_case.json"
        write_resolved_cases(resolved, [case])
        process_environment = os.environ.copy()
        process_environment["PYTHONUNBUFFERED"] = "1"
        process_environment["CATCCOS_UDMA_VALIDATE_ROUTES"] = "0"
        process_environment.setdefault("UDMA_MASTER_ADDR", "127.0.0.1")
        rendezvous_port = int(process_environment.get(
            "VALIDATION_MASTER_PORT", process_environment.get("MASTER_PORT", "29500"),
        )) + index
        udma_port = int(process_environment.get("UDMA_MASTER_PORT", "29601")) + index
        if not (1 <= rendezvous_port <= 65535 and 1 <= udma_port <= 65535):
            raise ValueError("torchrun and UDMA ports must be in [1, 65535]")
        if rendezvous_port == udma_port:
            raise ValueError("torchrun and UDMA bootstrap ports must differ")
        process_environment["UDMA_MASTER_PORT"] = str(udma_port)
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--master-addr=127.0.0.1", "--master-port={}".format(rendezvous_port),
            "--nproc-per-node={}".format(case["world_size"]), "--max-restarts=0",
            str(HERE / "stress_worker.py"), "--resolved-case", str(resolved),
            "--output-dir", str(case_root), "--mode", args.mode,
            "--iterations", str(args.iterations),
        ]
        write_json(case_root / "launch.json", {
            "command": command, "mode": args.mode, "iterations": args.iterations,
            "timeout_seconds": args.timeout_seconds,

            "udma_master_addr": process_environment["UDMA_MASTER_ADDR"],
            "udma_master_port": udma_port, "validation_master_port": rendezvous_port,
            "catccos_udma_validate_routes": "0", "console_log": str(case_root / "torchrun.log"),
        })
        print("[stress] {} mode={} iterations={} log={}".format(
            case["id"], args.mode, args.iterations, case_root / "torchrun.log",
        ), flush=True)
        log_roots = runtime_log_roots(process_environment)
        snapshot = snapshot_runtime_logs(log_roots)
        process_result = run_command(command, process_environment, case_root, args.timeout_seconds)
        result = summarize_case(
            case, args.mode, args.iterations, case_root, process_result,
        )
        result["runtime_logs"] = collect_runtime_logs(snapshot, log_roots, case_root / "runtime_logs")
        write_json(case_root / "report.json", result)
        results.append(result)
        print("[stress] {} {}".format(case["id"], "PASS" if result["passed"] else "FAIL"), flush=True)
        if not result["passed"] and args.fail_fast:
            break
    summary = {
        "passed": len(results) == len(cases) and all(item["passed"] for item in results),
        "mode": args.mode, "iterations": args.iterations,

        "environment": environment, "cases": results,
    }
    write_json(output_root / "summary.json", summary)
    print("stress results: {}".format(output_root), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
