# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Run isolated AllToAllV staging-readiness diagnostics with one wheel."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from .case_schema import load_case_manifest, select_cases
    from .common import timestamp, write_json
    from .runtime_log_collection import (
        collect_runtime_logs as _collect_mode_runtime_logs,
        runtime_log_roots as _runtime_log_roots,
        snapshot_runtime_logs as _snapshot_runtime_logs,
    )
except ImportError:
    from case_schema import load_case_manifest, select_cases
    from common import timestamp, write_json
    from runtime_log_collection import (
        collect_runtime_logs as _collect_mode_runtime_logs,
        runtime_log_roots as _runtime_log_roots,
        snapshot_runtime_logs as _snapshot_runtime_logs,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"
DIAGNOSTIC_MODES = (
    "baseline",
    "barrier_only",
    "generation_after_quiet",
    "strong_ready",
    "strong_ready_cache_safe",
)

RUNTIME_EVIDENCE_PATTERNS = (
    (
        "udma_quiet_cq_failure",
        re.compile(
            r"\[CATCCOS_UDMA_SYNC_DIAG\][^\n]*quiet_poll_failure",
            re.IGNORECASE,
        ),
    ),
    (
        "udma_cq_timeout",
        re.compile(
            r"(?:poll(?:ing)?[^\n]*cq[^\n]*timeout|"
            r"cq[^\n]*poll[^\n]*timeout)",
            re.IGNORECASE,
        ),
    ),
    (
        "udma_cqe_error",
        re.compile(
            r"(?:\[exception\]\s*\[udma\s+cqe\]|udma[^\n]*cqe[^\n]*(?:error|fail)|"
            r"cqe[^\n]*(?:status|substatus)[^\n]*(?:error|fail))",
            re.IGNORECASE,
        ),
    ),
    (
        "udma_channel_failure",
        re.compile(
            r"(?:HcommChannelCreate failed|Async connect failed|"
            r"Transport Manager ConnectWithOptions failed|"
            r"Wait hcomm UDMA channel ready timeout|"
            r"Check hcomm UDMA channel status failed)",
            re.IGNORECASE,
        ),
    ),
    (
        "shmem_initialization_failure",
        re.compile(
            r"(?:PrepareOpenDevice failed|Failed to open device|"
            r"(?:setup|reserve)_heap\(\) failed|"
            r"aclshmemx?_init_attr failed|return shmem error)",
            re.IGNORECASE,
        ),
    ),
    (
        "device_runtime_failure",
        re.compile(
            r"(?:rtStreamSynchronize[^\n]*(?:failed|error)|"
            r"device task[^\n]*(?:failed|error)|kernel[^\n]*exception)",
            re.IGNORECASE,
        ),
    ),
)


def classify_mode_results(results):
    """Turn the five orthogonal runs into the next debugging decision."""
    by_mode = {item["mode"]: item for item in results}
    quiet_cq_failure_modes = []
    for mode, item in by_mode.items():
        counts = item.get("runtime_evidence", {}).get("counts", {})
        explicit_failure = counts.get("udma_quiet_cq_failure", 0) > 0
        debug_abort_failure = (
            mode == "generation_after_quiet"
            and (
                counts.get("udma_cq_timeout", 0) > 0
                or counts.get("udma_cqe_error", 0) > 0
            )
        )
        if explicit_failure or debug_abort_failure:
            quiet_cq_failure_modes.append(mode)
    if quiet_cq_failure_modes:
        return {
            "classification": "udma_quiet_cq_failure",
            "conclusive": True,
            "next_action": (
                "A quiet-equivalent CQ poll returned timeout or CQE error. "
                "Do not publish ready after that failure; either promote the "
                "WWN protocol and surface completion errors, or fix the "
                "underlying UDMA transport/CQ failure first."
            ),
            "runtime_evidence_modes": quiet_cq_failure_modes,
        }
    def required_mode_problem(mode):
        item = by_mode.get(mode)
        if item is None or not item.get("summary_available", False):
            timed_out = bool(item and item.get("timed_out", False))
            evidence = bool(
                item
                and item.get("runtime_evidence", {}).get("detected", False)
            )
            detail = []
            if timed_out:
                detail.append("timed out")
            if evidence:
                detail.append("runtime/transport evidence detected")
            return {
                "classification": "infrastructure_or_runtime_failure",
                "conclusive": False,
                "next_action": (
                    "Inspect runtime_evidence and the {} torchrun log before "
                    "drawing the next kernel synchronization conclusion.{}"
                ).format(
                    mode,
                    " " + "; ".join(detail) if detail else "",
                ),
                "missing_modes": [mode],
                "timed_out_modes": [mode] if timed_out else [],
                "runtime_evidence_modes": [mode] if evidence else [],
            }
        if item.get("runtime_evidence", {}).get("detected", False):
            return {
                "classification": "runtime_or_transport_error_observed",
                "conclusive": False,
                "next_action": (
                    "The {} result is contaminated by UDMA/CQE, SHMEM, or "
                    "device-runtime errors. Inspect runtime_evidence before "
                    "selecting a synchronization fix."
                ).format(mode),
                "runtime_evidence_modes": [mode],
            }
        return None

    problem = required_mode_problem("baseline")
    if problem is not None:
        return problem
    if by_mode["baseline"]["passed"]:
        return {
            "classification": "baseline_did_not_reproduce",
            "conclusive": False,
            "next_action": (
                "Increase --baseline-launches or rerun on the environment that "
                "reproduces the consecutive-launch failure."
            ),
        }

    problem = required_mode_problem("barrier_only")
    if problem is not None:
        return problem
    if by_mode["barrier_only"]["passed"]:
        return {
            "classification": "cross_invocation_lifecycle_or_reset_race",
            "conclusive": True,
            "next_action": (
                "Promote the entry all-PE barrier that closes the previous "
                "invocation before resetting reusable synchronization state."
            ),
        }

    problem = required_mode_problem("generation_after_quiet")
    if problem is not None:
        return problem
    if by_mode["generation_after_quiet"]["passed"]:
        return {
            "classification": "legacy_aggregate_counter_protocol",
            "conclusive": True,
            "next_action": (
                "Replace the reusable aggregate completion counter with "
                "per-stage/per-source generation slots. The legacy payload "
                "WQE plus quiet path was sufficient in this run."
            ),
        }

    problem = required_mode_problem("strong_ready")
    if problem is not None:
        return problem
    problem = required_mode_problem("strong_ready_cache_safe")
    if problem is not None:
        return problem
    strong_ready_passed = by_mode["strong_ready"]["passed"]
    cache_safe_passed = by_mode["strong_ready_cache_safe"]["passed"]
    if strong_ready_passed and cache_safe_passed:
        return {
            "classification": "cross_engine_payload_ready_publication",
            "conclusive": True,
            "next_action": (
                "Promote per-stage/per-source write-with-notify generations. "
                "The public quiet returned, but publishing "
                "a generation through a separate signal operation remained "
                "insufficient; cache bypass is not required by this result."
            ),
        }
    if cache_safe_passed:
        return {
            "classification": "receiver_cache_visibility",
            "conclusive": True,
            "next_action": (
                "Keep per-stage/per-source write-with-notify generations and "
                "add the narrowest staging cache-visibility treatment that "
                "remains correct."
            ),
        }
    if strong_ready_passed:
        return {
            "classification": "non_monotonic_mode_result",
            "conclusive": False,
            "next_action": (
                "Repeat both strong modes: cache-safe is a superset of the "
                "ready protocol, so a strong-only pass is not stable evidence."
            ),
        }
    return {
        "classification": "not_isolated_by_ready_or_l2",
        "conclusive": False,
        "next_action": (
            "Use the first capped GMM/export failure dumps and launch records "
            "to investigate stage addressing, overwrite, or MMAD consumption."
        ),
    }


def _collect_runtime_evidence(output_dir, max_samples=50):
    counts = {}
    samples = []
    for log_path in sorted(output_dir.rglob("*.log")):
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    categories = [
                        name
                        for name, pattern in RUNTIME_EVIDENCE_PATTERNS
                        if pattern.search(line)
                    ]
                    if not categories:
                        continue
                    for category in categories:
                        counts[category] = counts.get(category, 0) + 1
                    if len(samples) < max_samples:
                        samples.append(
                            {
                                "categories": categories,
                                "path": str(log_path),
                                "line": line_number,
                                "text": line.rstrip()[:1000],
                            }
                        )
        except OSError as error:
            if len(samples) < max_samples:
                samples.append(
                    {
                        "categories": ["log_read_failure"],
                        "path": str(log_path),
                        "line": None,
                        "text": str(error),
                    }
                )
            counts["log_read_failure"] = counts.get("log_read_failure", 0) + 1
    return {
        "detected": any(
            name != "log_read_failure" and count > 0
            for name, count in counts.items()
        ),
        "counts": counts,
        "sample_limit": max_samples,
        "samples": samples,
    }


def _terminate_process_group(process, grace_seconds=10):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    process.wait()


def _run_with_timeout(command, environment, timeout_seconds):
    popen_options = {}
    if os.name == "posix":
        popen_options["start_new_session"] = True
    started = time.monotonic()
    process = subprocess.Popen(command, env=environment, **popen_options)
    timed_out = False
    try:
        process.wait(timeout=None if timeout_seconds == 0 else timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process)
    except BaseException:
        _terminate_process_group(process)
        raise
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _read_mode_result(
    mode, output_dir, returncode, launches, timed_out=False,
    elapsed_seconds=None, timeout_seconds=None,
):
    summary_path = output_dir / "summary.json"
    summary = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed_reports = []
    if summary is not None:
        failed_reports = [
            {
                "case_id": item.get("case_id"),
                "rank": item.get("rank"),
                "path": item.get("path"),
                "actual_nonfinite": item.get("actual_nonfinite"),
                "max_abs_error": item.get("max_abs_error"),
            }
            for item in summary.get("results", [])
            if not item.get("passed", False)
        ]
    return {
        "mode": mode,
        "requested_launches_per_rank": launches,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed_seconds,
        "summary_available": summary is not None,
        "passed": bool(
            summary and summary.get("passed") and returncode == 0 and not timed_out
        ),
        "failed_report_count": len(failed_reports),
        "failed_reports": failed_reports,
        "output_dir": str(output_dir),
        "summary": str(summary_path) if summary is not None else None,
        "runtime_evidence": _collect_runtime_evidence(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--launches", type=int, default=6,
        help="total automatic launches per rank in each non-baseline mode",
    )
    parser.add_argument(
        "--baseline-launches", type=int, default=3,
        help="total automatic launches per rank in baseline mode",
    )
    parser.add_argument(
        "--mode-timeout-seconds", type=int, default=900,
        help=(
            "wall-clock timeout for each isolated mode; 0 disables the "
            "timeout (default: 900)"
        ),
    )
    parser.add_argument(
        "--max-failure-dumps-per-role", type=int, default=1,
        help="large dumps kept per rank for GMM and AllToAllV outputs",
    )
    parser.add_argument(
        "--shmem-log-dir", action="append", default=[],
        help=(
            "additional SHMEM log file/directory to snapshot; may be repeated "
            "(defaults include ~/shmem/log)"
        ),
    )
    parser.add_argument(
        "--plog-dir", action="append", default=[],
        help=(
            "additional CANN plog/slog file or directory to snapshot; may be "
            "repeated"
        ),
    )
    parser.add_argument(
        "--runtime-log-settle-seconds", type=float, default=1.0,
        help="delay after each mode before snapshotting asynchronous logs",
    )
    args = parser.parse_args()

    if args.launches < 2 or args.baseline_launches < 2:
        raise ValueError("launch counts must be at least 2")
    if args.max_failure_dumps_per_role < 0:
        raise ValueError("--max-failure-dumps-per-role must be non-negative")
    if args.mode_timeout_seconds < 0:
        raise ValueError("--mode-timeout-seconds must be non-negative")
    if args.runtime_log_settle_seconds < 0:
        raise ValueError("--runtime-log-settle-seconds must be non-negative")

    selected = select_cases(
        load_case_manifest(args.cases), mode="accuracy", case_ids=[args.case_id]
    )
    if len(selected) != 1:
        raise ValueError("sync diagnosis requires exactly one selected case")
    case = selected[0]
    if (
        case["operator"] != "all_to_allv_grouped_mat_mul"
        or not case["output_all_to_allv"]
    ):
        raise ValueError(
            "sync diagnosis requires all_to_allv_grouped_mat_mul with "
            "output_all_to_allv: true"
        )

    output_root = Path(
        args.output_dir
        or (HERE.parent / "validation_results" / timestamp() / "sync_diagnosis")
    ).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    base_validation_port = int(
        os.environ.get("VALIDATION_MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
    )
    base_udma_port = int(os.environ.get("UDMA_MASTER_PORT", "29601"))
    requested_launches = {
        "baseline": args.baseline_launches,
        "barrier_only": args.launches,
        "generation_after_quiet": args.launches,
        "strong_ready": args.launches,
        "strong_ready_cache_safe": args.launches,
    }
    runtime_log_roots = _runtime_log_roots(
        os.environ,
        shmem_log_dirs=args.shmem_log_dir,
        plog_dirs=args.plog_dir,
    )
    plan = {
        "case_manifest": str(Path(args.cases).resolve()),
        "case_id": args.case_id,
        "case": case,
        "modes": list(DIAGNOSTIC_MODES),
        "requested_launches_per_rank": requested_launches,
        "automatic_only": True,
        "checks": ["operator_output", "all_to_allv_export"],
        "max_failure_dumps_per_role": args.max_failure_dumps_per_role,
        "mode_timeout_seconds": args.mode_timeout_seconds,
        "runtime_log_collection": {
            "enabled": True,
            "strategy": "per-mode new files and appended byte ranges",
            "settle_seconds": args.runtime_log_settle_seconds,
            "roots": runtime_log_roots,
            "environment_overrides": [
                "CATCCOS_UDMA_SHMEM_LOG_DIRS",
                "CATCCOS_UDMA_PLOG_DIRS",
                "CATCCOS_UDMA_SYNC_SHMEM_LOG_DIRS",
                "CATCCOS_UDMA_SYNC_PLOG_DIRS",
            ],
        },
        "diagnostic_matrix": {
            "baseline": {
                "entry_barrier": False,
                "payload_ready": "legacy quiet plus aggregate counter",
                "staging_l2": "default",
            },
            "barrier_only": {
                "entry_barrier": True,
                "payload_ready": "legacy quiet plus aggregate counter",
                "staging_l2": "default",
            },
            "generation_after_quiet": {
                "entry_barrier": True,
                "payload_ready": "legacy UDMA plus quiet, then per-source generation",
                "staging_l2": "default",
            },
            "strong_ready": {
                "entry_barrier": True,
                "payload_ready": "per-stage/source WWN generation",
                "staging_l2": "default",
            },
            "strong_ready_cache_safe": {
                "entry_barrier": True,
                "payload_ready": "per-stage/source WWN generation",
                "staging_l2": "bypass",
            },
        },
    }
    write_json(output_root / "plan.json", plan)

    mode_results = []
    for mode_index, mode in enumerate(DIAGNOSTIC_MODES):
        launches = requested_launches[mode]
        mode_output = output_root / mode
        environment = os.environ.copy()
        environment["CATCCOS_UDMA_SYNC_DIAGNOSTIC_MODE"] = mode
        environment["VALIDATION_MASTER_PORT"] = str(
            base_validation_port + mode_index
        )
        environment["UDMA_MASTER_PORT"] = str(base_udma_port + mode_index)
        for port_name in ("VALIDATION_MASTER_PORT", "UDMA_MASTER_PORT"):
            port = int(environment[port_name])
            if port < 1 or port > 65535:
                raise ValueError("{} must be in [1, 65535]".format(port_name))
        if environment["VALIDATION_MASTER_PORT"] == environment["UDMA_MASTER_PORT"]:
            raise ValueError("torchrun and UDMA bootstrap ports must differ")

        command = [
            sys.executable,
            str(HERE / "run_accuracy.py"),
            "--cases", str(Path(args.cases).resolve()),
            "--case-id", args.case_id,
            "--output-dir", str(mode_output),
            "--automatic-only",
            "--no-runtime-log-collection",
            "--auto-repeats", str(launches - 1),
            "--max-failure-dumps-per-role",
            str(args.max_failure_dumps_per_role),
        ]
        print(
            "[sync-diagnosis] mode={} launches={} output={}".format(
                mode, launches, mode_output
            ),
            flush=True,
        )
        log_snapshot = _snapshot_runtime_logs(runtime_log_roots)
        try:
            execution = _run_with_timeout(
                command, environment, args.mode_timeout_seconds
            )
        except BaseException:
            if args.runtime_log_settle_seconds:
                time.sleep(args.runtime_log_settle_seconds)
            _collect_mode_runtime_logs(
                log_snapshot, runtime_log_roots, mode_output / "runtime_logs"
            )
            raise
        if args.runtime_log_settle_seconds:
            time.sleep(args.runtime_log_settle_seconds)
        runtime_logs = _collect_mode_runtime_logs(
            log_snapshot, runtime_log_roots, mode_output / "runtime_logs"
        )
        result = _read_mode_result(
            mode,
            mode_output,
            execution["returncode"],
            launches,
            timed_out=execution["timed_out"],
            elapsed_seconds=execution["elapsed_seconds"],
            timeout_seconds=args.mode_timeout_seconds,
        )
        result["runtime_logs"] = runtime_logs
        mode_results.append(result)
        write_json(
            output_root / "progress.json",
            {
                "case_id": args.case_id,
                "completed_modes": [item["mode"] for item in mode_results],
                "modes": mode_results,
            },
        )
        print(
            "[sync-diagnosis] mode={} passed={} returncode={} timed_out={} "
            "elapsed_seconds={} failed_reports={} runtime_evidence={}".format(
                mode, result["passed"], execution["returncode"],
                execution["timed_out"], execution["elapsed_seconds"],
                result["failed_report_count"],
                result["runtime_evidence"]["counts"],
            ),
            flush=True,
        )
        print(
            "[sync-diagnosis] mode={} runtime_logs files={} bytes={} "
            "errors={} manifest={}".format(
                mode,
                runtime_logs["captured_file_count"],
                runtime_logs["captured_bytes"],
                runtime_logs["error_count"],
                runtime_logs["manifest"],
            ),
            flush=True,
        )

    decision = classify_mode_results(mode_results)
    aggregate = {
        "case_id": args.case_id,
        "plan": str(output_root / "plan.json"),
        "modes": mode_results,
        "decision": decision,
    }
    write_json(output_root / "summary.json", aggregate)
    print(
        "[sync-diagnosis] classification={} conclusive={} next_action={}".format(
            decision["classification"], decision["conclusive"],
            decision["next_action"],
        ),
        flush=True,
    )
    print("sync diagnosis results: {}".format(output_root), flush=True)
    return 0 if decision["conclusive"] else 1


if __name__ == "__main__":
    sys.exit(main())
