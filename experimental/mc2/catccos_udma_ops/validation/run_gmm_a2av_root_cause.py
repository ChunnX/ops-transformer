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

# Safe modes run first. The original protocol runs last so a possible hang
# cannot discard the discriminating results or force regeneration of Golden.
DIAGNOSTIC_MODES = (
    "no_reuse_trace",
    "pre_get_delay_medium",
    "post_quiet_delay_short",
    "post_quiet_delay_medium",
    "post_quiet_delay_long",
    "baseline",
)

MODE_DESCRIPTIONS = {
    "baseline": {
        "label": "A",
        "producer_sync": "original local AscendC::SyncAll",
        "stage_recycle_sync": "original local AscendC::SyncAll",
        "udma_wqe_path": "PIPE_S",
    },
    "producer_barrier": {
        "label": "B",
        "producer_sync": "SHMEM all-PE barrier replacing producer SyncAll",
        "stage_recycle_sync": "original local AscendC::SyncAll",
        "udma_wqe_path": "PIPE_S",
    },
    "full_barrier": {
        "label": "C",
        "producer_sync": "SHMEM all-PE barrier replacing producer SyncAll",
        "stage_recycle_sync": "SHMEM all-PE barrier replacing recycle SyncAll",
        "udma_wqe_path": "PIPE_S",
    },
    "full_barrier_mte3": {
        "label": "D",
        "producer_sync": "SHMEM all-PE barrier replacing producer SyncAll",
        "stage_recycle_sync": "SHMEM all-PE barrier replacing recycle SyncAll",
        "udma_wqe_path": "official-style PIPE_MTE3 with UB WQE scratch",
    },
    "full_barrier_official_block": {
        "label": "E",
        "producer_sync": "SHMEM all-PE barrier replacing producer SyncAll",
        "stage_recycle_sync": "SHMEM all-PE barrier replacing recycle SyncAll",
        "udma_wqe_path": (
            "exact CatCCOS BlockRemoteCopy call: default-constructed block, "
            "AIC direct self output, remote-only UDMA GET with per-call "
            "TPipe/TBuf and internal void quiet"
        ),
    },
    "pre_get_delay_medium": {
        "label": "pre-20k",
        "producer_sync": "original local AscendC::SyncAll",
        "stage_recycle_sync": "original local AscendC::SyncAll",
        "udma_wqe_path": "PIPE_S",
        "delay_placement": "after producer-ready wait, before UDMA GET",
        "delay_cycles": 20_000,
    },
    "post_quiet_delay_short": {
        "label": "post-2k",
        "producer_sync": "original local AscendC::SyncAll",
        "stage_recycle_sync": "original local AscendC::SyncAll",
        "udma_wqe_path": "PIPE_S",
        "delay_placement": "after public UDMA quiet, before completion signal",
        "delay_cycles": 2_000,
    },
    "post_quiet_delay_medium": {
        "label": "post-20k",
        "producer_sync": "original local AscendC::SyncAll",
        "stage_recycle_sync": "original local AscendC::SyncAll",
        "udma_wqe_path": "PIPE_S",
        "delay_placement": "after public UDMA quiet, before completion signal",
        "delay_cycles": 20_000,
    },
    "post_quiet_delay_long": {
        "label": "post-200k",
        "producer_sync": "original local AscendC::SyncAll",
        "stage_recycle_sync": "original local AscendC::SyncAll",
        "udma_wqe_path": "PIPE_S",
        "delay_placement": "after public UDMA quiet, before completion signal",
        "delay_cycles": 200_000,
    },
    "no_reuse_trace": {
        "label": "no-reuse",
        "producer_sync": "per-round SHMEM all-PE barrier",
        "stage_recycle_sync": "disabled; one stage per commIdx",
        "udma_wqe_path": "PIPE_S GET with public UDMA quiet",
        "workspace_stages": 19,
        "producer_trace": (
            "destination-major local producer staging copied after a final "
            "all-PE barrier and compared with local FP32 GMM Golden"
        ),
    },
}


def classify_results(mode_results, *, timed_out_mode=None, runtime_evidence=None):
    """Classify no-reuse/delay and legacy matrices, including timeouts."""
    by_mode = {item["mode"]: item for item in mode_results}
    incomplete_modes = [
        item["mode"] for item in mode_results
        if item.get("completion_failure_count", 0) > 0
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
    if incomplete_modes or transport_evidence:
        return {
            "classification": "udma_get_completion_or_transport_failure",
            "conclusive": True,
            "evidence_modes": incomplete_modes,
            "next_action": (
                "Inspect the missing GET completion or transport failure first; "
                "do not infer a CQ status from public quiet's return."
            ),
        }

    def passed(mode):
        item = by_mode.get(mode)
        return bool(item and item.get("passed"))

    if timed_out_mode and timed_out_mode != "baseline":
        return {
            "classification": "diagnostic_protocol_timeout",
            "conclusive": False,
            "evidence_modes": [timed_out_mode],
            "next_action": (
                "A strengthened mode timed out before the baseline comparison. "
                "Inspect its diagnostic tensor, progress.json, and runtime logs."
            ),
        }
    if "baseline" not in by_mode and timed_out_mode != "baseline":
        return {
            "classification": "incomplete_matrix",
            "conclusive": False,
            "next_action": (
                "The original baseline neither completed nor timed out; inspect "
                "progress.json and torchrun.log."
            ),
        }

    if passed("baseline"):
        return {
            "classification": "baseline_did_not_reproduce",
            "conclusive": False,
            "next_action": (
                "Increase --launches-per-mode or rerun on the failing server; "
                "the shared input and Golden contract was verified."
            ),
        }

    no_reuse_result = by_mode.get("no_reuse_trace")
    if no_reuse_result and no_reuse_result.get(
        "producer_trace_passed"
    ) is False:
        return {
            "classification": "producer_gmm_or_staging_payload_failure",
            "conclusive": True,
            "evidence_modes": ["no_reuse_trace"],
            "next_action": (
                "Inspect producer_trace/report.json by destination/expert; "
                "the retained local staging already differs before UDMA GET."
            ),
        }
    if (
        no_reuse_result and
        no_reuse_result.get("producer_trace_passed") is True and
        no_reuse_result.get("final_output_passed") is False
    ):
        return {
            "classification": "udma_get_address_visibility_or_output_failure",
            "conclusive": True,
            "evidence_modes": ["no_reuse_trace"],
            "next_action": (
                "Producer staging is correct without reuse but the final "
                "AllToAllV output is not. Inspect stage offsets and GET "
                "destination ranges in diagnostic.json/layout.json."
            ),
        }

    no_reuse = passed("no_reuse_trace")
    pre_get = passed("pre_get_delay_medium")
    post_modes = [
        mode for mode in (
            "post_quiet_delay_short",
            "post_quiet_delay_medium",
            "post_quiet_delay_long",
        ) if passed(mode)
    ]
    if no_reuse:
        if post_modes:
            return {
                "classification": "quiet_to_stage_recycle_timing_window",
                "conclusive": True,
                "evidence_modes": ["no_reuse_trace"] + post_modes,
                "next_action": (
                    "Replace delay with generation-scoped consumer completion; "
                    "do not release a stage until every destination's "
                    "GET has deterministically completed."
                ),
            }
        return {
            "classification": "cyclic_stage_reuse_or_recycle_protocol",
            "conclusive": True,
            "evidence_modes": ["no_reuse_trace"],
            "next_action": (
                "The producer staging and final output pass without reuse. "
                "Fix stage generation/consumer ownership or evaluate the "
                "independent-staging delivery tradeoff."
            ),
        }
    if pre_get and not post_modes:
        return {
            "classification": "producer_payload_visibility_window",
            "conclusive": True,
            "evidence_modes": ["pre_get_delay_medium"],
            "next_action": (
                "Replace the delay with a producer release/consumer acquire "
                "edge before UDMA GET."
            ),
        }
    if post_modes and not pre_get:
        return {
            "classification": "get_completion_to_recycle_window",
            "conclusive": True,
            "evidence_modes": post_modes,
            "next_action": (
                "Replace the delay with deterministic completion and "
                "destination-scoped stage release."
            ),
        }

    producer = passed("producer_barrier")
    full = passed("full_barrier")
    mte3 = passed("full_barrier_mte3")
    official_block = passed("full_barrier_official_block")
    if producer:
        return {
            "classification": "producer_staging_visibility_or_ready_order",
            "conclusive": True,
            "evidence_modes": ["producer_barrier"],
            "next_action": (
                "Replace the producer-side local SyncAll edge with a protocol "
                "that guarantees FIX staging is remotely readable before GET."
            ),
        }
    if full:
        return {
            "classification": "premature_symmetric_stage_recycle",
            "conclusive": True,
            "evidence_modes": ["full_barrier"],
            "mte3_mode_passed": mte3,
            "next_action": (
                "Keep public GET completion and replace the aggregate recycle "
                "edge with stage/source/generation-scoped consumer completion."
            ),
        }
    if mte3:
        return {
            "classification": "pipe_s_wqe_publication_topology_sensitivity",
            "conclusive": True,
            "evidence_modes": ["full_barrier_mte3"],
            "next_action": (
                "Use the official-style PIPE_MTE3 WQE publication path or fix "
                "the PIPE_S path before tuning the synchronization protocol."
            ),
        }
    if official_block:
        return {
            "classification": "custom_udma_get_or_deferred_quiet_path",
            "conclusive": True,
            "evidence_modes": ["full_barrier_official_block"],
            "next_action": (
                "The official CatCCOS BlockRemoteCopy passed while both "
                "custom block-loop paths failed. Compare custom 2D WQE "
                "submission/deferred CQ drain with the official per-call "
                "TPipe/TBuf plus void quiet implementation."
            ),
        }
    return {
        "classification": "not_isolated_by_sync_or_wqe_matrix",
        "conclusive": False,
        "next_action": (
            "All strengthened modes failed with the same resident inputs. Use "
            "diagnostic.json plus by_source_rank.csv to inspect layout/offsets "
            "or a payload visibility guarantee below the tested barriers."
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
    build_environment["CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE"] = "1"
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
    parser.add_argument("--launches-per-mode", type=int, default=3)
    parser.add_argument(
        "--mode", action="append", choices=tuple(MODE_DESCRIPTIONS),
        help=(
            "run only the selected mode(s); repeat the option to build a "
            "custom matrix (default: no-reuse + delay matrix + baseline)"
        ),
    )
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
    selected_modes = tuple(args.mode or DIAGNOSTIC_MODES)
    if case["operator"] != "grouped_mat_mul_all_to_allv":
        raise ValueError("this matrix supports grouped_mat_mul_all_to_allv only")
    if case["route"] != "balanced":
        raise ValueError(
            "the per-round barrier matrix requires route: balanced; use ordinary "
            "accuracy tests for imbalanced-route validation"
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
    for mode in selected_modes:
        command.extend(["--mode", mode])

    log_roots = runtime_log_roots(
        environment,
        shmem_log_dirs=args.shmem_log_dir,
        plog_dirs=args.plog_dir,
    )
    log_snapshot = snapshot_runtime_logs(log_roots)
    plan = {
        "case": case,
        "mode_order": list(selected_modes),
        "mode_descriptions": MODE_DESCRIPTIONS,
        "launches_per_mode": args.launches_per_mode,
        "single_torchrun": True,
        "single_udma_initialization_per_rank": True,
        "input_generation_count_per_rank": 1,
        "golden_generation_count_per_rank": 1,
        "output_isolation": "NaN-prefilled caller-provided output per launch",
        "rank_launch_alignment": (
            "Gloo barriers after output prefill, after kernel completion, "
            "and after report generation"
        ),
        "mutation_validation": (
            "full-tensor exact reference clones plus sampled SHA-256"
        ),
        "baseline_runs_last": (
            "baseline" not in selected_modes or
            selected_modes[-1] == "baseline"
        ),
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
            "completion_failure_count": 0,
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
    # A conclusive comparison remains useful when the intentionally unsafe
    # baseline is the only mode that times out after all strengthened modes.
    complete_or_baseline_timeout = (
        execution["returncode"] == 0 or
        execution["timed_out_mode"] == "baseline"
    )
    return 0 if decision["conclusive"] and complete_or_baseline_timeout else 1


if __name__ == "__main__":
    sys.exit(main())
