# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Run one persistent GMM->AllToAllV synchronization root-cause matrix."""

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
    from .case_schema import (
        load_case_manifest,
        select_cases,
        write_resolved_cases,
    )
    from .common import environment_fingerprint, timestamp, write_json
    from .run_sync_diagnosis import _collect_runtime_evidence
    from .runtime_log_collection import (
        collect_runtime_logs,
        runtime_log_roots,
        snapshot_runtime_logs,
    )
except ImportError:
    from case_schema import (
        load_case_manifest,
        select_cases,
        write_resolved_cases,
    )
    from common import environment_fingerprint, timestamp, write_json
    from run_sync_diagnosis import _collect_runtime_evidence
    from runtime_log_collection import (
        collect_runtime_logs,
        runtime_log_roots,
        snapshot_runtime_logs,
    )


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
DEFAULT_CASES = HERE / "cases" / "grouped_mat_mul_all_to_allv.yaml"

# Safe, heavily synchronized modes run first. Production is intentionally last:
# if it hangs, every discriminating result and the single-generation reuse
# manifest have already been flushed to disk.
DIAGNOSTIC_MODES = (
    "serialized_default",
    "serialized_delay",
    "serialized_fix_bypass",
    "serialized_mte_republish",
    "rolling_udma_ack",
    "rolling_fix_bypass",
    "rolling_bypass_udma_ack",
    "production",
)

MODE_DESCRIPTIONS = {
    "serialized_default": {
        "schedule": "per-round all-PE producer and consumer barriers",
        "fix_staging_l2": "default",
        "consumer_ack": "global round barrier",
    },
    "serialized_delay": {
        "schedule": "serialized plus one extra pre-GET all-PE barrier",
        "fix_staging_l2": "default",
        "consumer_ack": "global round barrier",
    },
    "serialized_fix_bypass": {
        "schedule": "per-round all-PE producer and consumer barriers",
        "fix_staging_l2": "CACHE_MODE_DISABLE on FIX destination",
        "consumer_ack": "global round barrier",
    },
    "serialized_mte_republish": {
        "schedule": "per-round all-PE producer and consumer barriers",
        "fix_staging_l2": "default; FIX output is republished GM->UB->shadow GM",
        "consumer_ack": "global round barrier",
    },
    "rolling_udma_ack": {
        "schedule": "production rolling stages",
        "fix_staging_l2": "default",
        "consumer_ack": "UDMA WRITE_WITH_NOTIFY 64-bit generation",
    },
    "rolling_fix_bypass": {
        "schedule": "production rolling stages",
        "fix_staging_l2": "CACHE_MODE_DISABLE on FIX destination",
        "consumer_ack": "MTE SIGNAL_SET generation",
    },
    "rolling_bypass_udma_ack": {
        "schedule": "production rolling stages",
        "fix_staging_l2": "CACHE_MODE_DISABLE on FIX destination",
        "consumer_ack": "UDMA WRITE_WITH_NOTIFY 64-bit generation",
    },
    "production": {
        "schedule": "production rolling stages",
        "fix_staging_l2": "default",
        "consumer_ack": "MTE SIGNAL_SET generation",
    },
}


def classify_results(mode_results, *, timed_out_mode=None, runtime_evidence=None):
    """Classify the orthogonal matrix without assuming a failed launch returned."""
    by_mode = {item["mode"]: item for item in mode_results}
    cq_modes = [
        item["mode"] for item in mode_results
        if item.get("cq_failure_count", 0) > 0
    ]
    runtime_counts = (runtime_evidence or {}).get("counts", {})
    transport_evidence = any(runtime_counts.get(name, 0) for name in (
        "udma_quiet_cq_failure",
        "udma_cq_timeout",
        "udma_cqe_error",
        "udma_channel_failure",
        "shmem_initialization_failure",
        "device_runtime_failure",
    ))
    if cq_modes or transport_evidence:
        return {
            "classification": "udma_get_cq_or_transport_failure",
            "conclusive": True,
            "evidence_modes": cq_modes,
            "next_action": (
                "Fix or surface the checked GET CQ/transport failure first; "
                "a source stage must never be released after this status."
            ),
        }

    def passed(mode):
        item = by_mode.get(mode)
        return bool(item and item.get("passed"))

    production_available = "production" in by_mode
    if production_available and passed("production"):
        return {
            "classification": "production_did_not_reproduce",
            "conclusive": False,
            "next_action": (
                "Increase --launches-per-mode or rerun on the server where the "
                "production path fails; input and Golden reuse was verified."
            ),
        }
    if not production_available and timed_out_mode != "production":
        return {
            "classification": "infrastructure_or_preproduction_timeout",
            "conclusive": False,
            "next_action": (
                "The matrix stopped before production. Inspect progress.json, "
                "torchrun.log, and runtime_logs before drawing a kernel conclusion."
            ),
        }

    ack = passed("rolling_udma_ack")
    bypass = passed("rolling_fix_bypass")
    combined = passed("rolling_bypass_udma_ack")
    if ack and not bypass:
        return {
            "classification": "consumer_ack_publication_or_stage_reuse",
            "conclusive": True,
            "next_action": (
                "Promote the 64-bit UDMA WRITE_WITH_NOTIFY consumer ack while "
                "keeping the default FIX/L2 policy."
            ),
        }
    if bypass and not ack:
        return {
            "classification": "fix_staging_l2_visibility",
            "conclusive": True,
            "next_action": (
                "The rolling schedule is correct only when FIX staging bypasses "
                "L2. Narrow the cache treatment before accepting its performance cost."
            ),
        }
    if combined and not ack and not bypass:
        return {
            "classification": "combined_fix_visibility_and_ack_ordering",
            "conclusive": True,
            "next_action": (
                "Neither isolated change is sufficient; retain both in the next "
                "candidate and then measure the independent performance costs."
            ),
        }
    if ack and bypass:
        return {
            "classification": "multiple_independent_timing_mitigations",
            "conclusive": False,
            "next_action": (
                "Both isolated rolling changes pass. Repeat more launches before "
                "choosing one: either may be masking a timing-sensitive producer edge."
            ),
        }

    serialized = passed("serialized_default")
    delayed = passed("serialized_delay")
    serialized_bypass = passed("serialized_fix_bypass")
    republished = passed("serialized_mte_republish")
    if serialized:
        return {
            "classification": "rolling_stage_lifecycle_or_cross_rank_ordering",
            "conclusive": True,
            "next_action": (
                "Data/layout and default L2 are correct under strict round "
                "lifecycle. Repair the rolling producer/consumer protocol; do "
                "not add global barriers to the production fast path."
            ),
        }
    if delayed and not serialized:
        return {
            "classification": "fix_to_udma_visibility_latency",
            "conclusive": True,
            "next_action": (
                "A rendezvous alone is insufficient but an additional device "
                "interval works, proving a missing FIX-to-UDMA publication edge."
            ),
        }
    if serialized_bypass and not serialized:
        return {
            "classification": "fix_staging_l2_visibility",
            "conclusive": True,
            "next_action": (
                "Serialized default fails while FIX cache bypass passes; focus on "
                "the producer's staging visibility, not GET completion ack."
            ),
        }
    if republished and not serialized:
        return {
            "classification": "fix_output_publication_path",
            "conclusive": True,
            "next_action": (
                "GM->UB->shadow-GM republish makes the same values readable. Add "
                "the narrowest explicit FIX publication mechanism or redesign "
                "the staging writer."
            ),
        }
    return {
        "classification": "not_isolated_by_synchronization_matrix",
        "conclusive": False,
        "next_action": (
            "All synchronized variants failed. Investigate target-order layout, "
            "route offsets, GMM production, or Golden construction using the "
            "per-segment and Top-K artifacts."
        ),
    }


def _load_progress(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _terminate_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()


def _pump_output(process, log_path):
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()


def _run_persistent_worker(
    command,
    environment,
    output_root,
    *,
    mode_timeout_seconds,
    preparation_timeout_seconds,
):
    progress_path = output_root / "progress.json"
    log_path = output_root / "torchrun.log"
    process = subprocess.Popen(
        command,
        cwd=str(output_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    output_thread = threading.Thread(
        target=_pump_output, args=(process, log_path), daemon=True
    )
    output_thread.start()
    process_started = time.time()
    timed_out = False
    timed_out_mode = None
    timed_out_phase = None
    while process.poll() is None:
        progress = _load_progress(progress_path)
        now = time.time()
        if progress and progress.get("phase") == "mode_started":
            phase_started = progress.get(
                "active_mode_started_unix", progress.get("updated_unix", now)
            )
            if now - phase_started > mode_timeout_seconds:
                timed_out = True
                timed_out_mode = progress.get("active_mode")
                timed_out_phase = "mode_started"
        elif progress:
            phase_started = progress.get("updated_unix", process_started)
            if now - phase_started > preparation_timeout_seconds:
                timed_out = True
                timed_out_phase = progress.get("phase")
        elif now - process_started > preparation_timeout_seconds:
            timed_out = True
            timed_out_phase = "before_progress"
        if timed_out:
            print(
                "[gmm-a2av-root-cause] timeout phase={} mode={}; terminating "
                "the persistent torchrun group".format(
                    timed_out_phase, timed_out_mode
                ),
                flush=True,
            )
            _terminate_process_tree(process)
            break
        time.sleep(1.0)
    returncode = process.wait()
    output_thread.join(timeout=5)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "timed_out_mode": timed_out_mode,
        "timed_out_phase": timed_out_phase,
        "console_log": str(log_path),
    }


def _build_and_install():
    build_script = PROJECT_DIR / "scripts" / "build_wheel.sh"
    build_environment = os.environ.copy()
    # The status-buffer path is sufficient for this experiment. Do not let an
    # inherited printf build flag perturb timing or reserve device printf HBM.
    build_environment.pop("CATCCOS_UDMA_DEVICE_DIAGNOSTICS", None)
    subprocess.check_call(
        ["bash", str(build_script)],
        cwd=str(PROJECT_DIR),
        env=build_environment,
    )
    wheels = sorted(
        (PROJECT_DIR / "dist").glob("*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not wheels:
        raise RuntimeError("build completed without a wheel in dist/")
    wheel = wheels[-1]
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps", str(wheel),
    ])
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--launches-per-mode", type=int, default=2)
    parser.add_argument("--mode-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--preparation-timeout-seconds", type=float, default=7200.0
    )
    parser.add_argument("--fingerprint-elements", type=int, default=1 << 16)
    parser.add_argument(
        "--comparison-chunk-elements", type=int, default=1 << 22
    )
    parser.add_argument(
        "--build-install", action="store_true",
        help="build the wheel and force-install it before the single torchrun",
    )
    parser.add_argument("--shmem-log-dir", action="append", default=[])
    parser.add_argument("--plog-dir", action="append", default=[])
    parser.add_argument(
        "--runtime-log-settle-seconds", type=float, default=1.0
    )
    args = parser.parse_args()
    if args.launches_per_mode <= 0:
        raise ValueError("--launches-per-mode must be positive")
    if args.mode_timeout_seconds <= 0 or args.preparation_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    if args.runtime_log_settle_seconds < 0:
        raise ValueError("runtime log settle time must be non-negative")

    selected = select_cases(
        load_case_manifest(args.cases),
        mode="accuracy",
        case_ids=[args.case_id],
    )
    if len(selected) != 1:
        raise ValueError("select exactly one case")
    case = selected[0]
    if case["operator"] != "grouped_mat_mul_all_to_allv":
        raise ValueError("this matrix supports grouped_mat_mul_all_to_allv only")
    if case["route"] != "balanced":
        raise ValueError(
            "serialized modes require route: balanced; use ordinary accuracy "
            "tests for imbalanced-route validation"
        )

    built_wheel = _build_and_install() if args.build_install else None
    output_root = Path(args.output_dir or (
        PROJECT_DIR / "validation_results" / timestamp() /
        "gmm_a2av_root_cause"
    )).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = output_root / "resolved_case.json"
    write_resolved_cases(resolved, [case])

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("UDMA_MASTER_ADDR", "127.0.0.1")
    rendezvous_port = int(
        environment.get(
            "VALIDATION_MASTER_PORT",
            environment.get("MASTER_PORT", "29500"),
        )
    )
    udma_port = int(environment.get("UDMA_MASTER_PORT", "29601"))
    if rendezvous_port == udma_port:
        raise ValueError("torchrun and UDMA bootstrap ports must differ")
    if not (1 <= rendezvous_port <= 65535 and 1 <= udma_port <= 65535):
        raise ValueError("bootstrap ports must be in [1, 65535]")
    environment["UDMA_MASTER_PORT"] = str(udma_port)
    # Ignore an inherited one-off mode: the worker explicitly switches all
    # modes in a fixed order inside the same process.
    environment.pop("CATCCOS_UDMA_GMM_A2AV_DIAGNOSTIC_MODE", None)

    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--master-addr=127.0.0.1",
        "--master-port={}".format(rendezvous_port),
        "--nproc-per-node={}".format(case["world_size"]),
        str(HERE / "gmm_a2av_root_cause_worker.py"),
        "--resolved-case", str(resolved),
        "--output-dir", str(output_root),
        "--launches-per-mode", str(args.launches_per_mode),
        "--fingerprint-elements", str(args.fingerprint_elements),
        "--comparison-chunk-elements", str(args.comparison_chunk_elements),
    ]
    for mode in DIAGNOSTIC_MODES:
        command.extend(["--mode", mode])

    log_roots = runtime_log_roots(
        environment,
        shmem_log_dirs=args.shmem_log_dir,
        plog_dirs=args.plog_dir,
    )
    log_snapshot = snapshot_runtime_logs(log_roots)
    plan = {
        "case": case,
        "mode_order": list(DIAGNOSTIC_MODES),
        "mode_descriptions": MODE_DESCRIPTIONS,
        "launches_per_mode": args.launches_per_mode,
        "single_torchrun": True,
        "single_udma_initialization_per_rank": True,
        "input_generation_count_per_rank": 1,
        "golden_generation_count_per_rank": 1,
        "production_runs_last": True,
        "mode_timeout_seconds": args.mode_timeout_seconds,
        "preparation_timeout_seconds": args.preparation_timeout_seconds,
        "command": command,
        "built_wheel": str(built_wheel) if built_wheel else None,
        "environment": environment_fingerprint(),
    }
    write_json(output_root / "plan.json", plan)
    print("+ " + " ".join(command), flush=True)
    execution = _run_persistent_worker(
        command,
        environment,
        output_root,
        mode_timeout_seconds=args.mode_timeout_seconds,
        preparation_timeout_seconds=args.preparation_timeout_seconds,
    )
    if args.runtime_log_settle_seconds:
        time.sleep(args.runtime_log_settle_seconds)
    runtime_logs = collect_runtime_logs(
        log_snapshot, log_roots, output_root / "runtime_logs"
    )
    runtime_evidence = _collect_runtime_evidence(output_root)
    progress = _load_progress(output_root / "progress.json") or {}
    mode_results = list(progress.get("modes", []))
    if execution["timed_out_mode"] and not any(
        item["mode"] == execution["timed_out_mode"]
        for item in mode_results
    ):
        mode_results.append({
            "mode": execution["timed_out_mode"],
            "passed": False,
            "timed_out": True,
            "failed_ranks": list(range(case["world_size"])),
            "cq_failure_count": 0,
        })
    decision = classify_results(
        mode_results,
        timed_out_mode=execution["timed_out_mode"],
        runtime_evidence=runtime_evidence,
    )
    summary = {
        "case_id": case["id"],
        "execution": execution,
        "completed_modes": progress.get("completed_modes", []),
        "modes": mode_results,
        "decision": decision,
        "reuse_manifest": (
            str(output_root / "reuse_manifest.json")
            if (output_root / "reuse_manifest.json").exists() else None
        ),
        "runtime_evidence": runtime_evidence,
        "runtime_logs": runtime_logs,
    }
    write_json(output_root / "summary.json", summary)
    print(
        "[gmm-a2av-root-cause] classification={} conclusive={} "
        "timed_out_mode={} next_action={}".format(
            decision["classification"], decision["conclusive"],
            execution["timed_out_mode"], decision["next_action"],
        ),
        flush=True,
    )
    print("root-cause results: {}".format(output_root), flush=True)
    # A complete, conclusive experiment is success even when production is
    # intentionally proven wrong. Infrastructure failures remain nonzero.
    complete_or_production_timeout = (
        execution["returncode"] == 0 or
        execution["timed_out_mode"] == "production"
    )
    return 0 if decision["conclusive"] and complete_or_production_timeout else 1


if __name__ == "__main__":
    sys.exit(main())
