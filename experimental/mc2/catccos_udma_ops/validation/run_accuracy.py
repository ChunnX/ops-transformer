# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Manifest-driven multi-case accuracy launcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from .case_schema import load_case_manifest, select_cases, write_resolved_cases
    from .common import (
        environment_fingerprint,
        timestamp,
        write_csv,
        write_json,
    )
    from .runtime_log_collection import (
        collect_runtime_logs,
        runtime_log_roots,
        snapshot_runtime_logs,
    )
except ImportError:
    from case_schema import load_case_manifest, select_cases, write_resolved_cases
    from common import (
        environment_fingerprint,
        timestamp,
        write_csv,
        write_json,
    )
    from runtime_log_collection import (
        collect_runtime_logs,
        runtime_log_roots,
        snapshot_runtime_logs,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"


def main():
    launcher_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-tag", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--memory-safety", action="store_true",
        help="poison HBM/workspace and add red-zone, unwritten-output and mutation checks",
    )
    parser.add_argument(
        "--mssanitizer", action="store_true",
        help="launch exactly one selected case once per rank for msSanitizer",
    )
    parser.add_argument(
        "--mssanitizer-route-input", choices=("tensor", "list"), default="tensor",
        help="route metadata representation used by the single msSanitizer launch",
    )
    parser.add_argument("--hbm-reserve-gib", type=float, default=2.0)
    parser.add_argument("--hbm-poison-chunk-mib", type=int, default=256)
    parser.add_argument("--guard-mib", type=int, default=16)
    parser.add_argument("--fingerprint-elements", type=int, default=1 << 20)
    parser.add_argument(
        "--no-runtime-log-collection", action="store_true",
        help="disable automatic per-launch SHMEM and CANN plog collection",
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
        help="delay after each torchrun launch before snapshotting runtime logs",
    )
    parser.add_argument(
        "--automatic-only", action="store_true",
        help=(
            "run only automatically allocated consecutive launches without "
            "cross-rank pre-launch barriers; intended only for sync diagnosis"
        ),
    )
    parser.add_argument(
        "--check-reusable-out", action="store_true",
        help=(
            "also validate caller-provided out tensors; enabled implicitly by "
            "--memory-safety and disabled by default for ordinary accuracy"
        ),
    )
    parser.add_argument(
        "--auto-repeats", type=int,
        help="override accuracy.auto_repeats for every selected case",
    )
    parser.add_argument(
        "--max-failure-dumps-per-role", type=int, default=-1,
        help="cap large failure dumps per rank and output role (-1: unlimited)",
    )
    args = parser.parse_args()
    cases = select_cases(
        load_case_manifest(args.cases), mode="accuracy",
        case_ids=args.case_id, include_tags=args.include_tag,
    )
    if args.memory_safety and args.mssanitizer:
        raise ValueError("--memory-safety and --mssanitizer are mutually exclusive")
    if args.automatic_only and (args.memory_safety or args.mssanitizer):
        raise ValueError(
            "--automatic-only cannot be combined with memory-safety or mssanitizer"
        )
    if args.automatic_only and args.check_reusable_out:
        raise ValueError(
            "--automatic-only and --check-reusable-out are mutually exclusive"
        )
    if args.mssanitizer and args.check_reusable_out:
        raise ValueError(
            "--mssanitizer keeps one automatic launch and cannot check reusable out"
        )
    check_reusable_out = args.check_reusable_out or args.memory_safety
    if args.automatic_only and any(
        case["operator"] != "all_to_allv_grouped_mat_mul"
        or not case["output_all_to_allv"]
        for case in cases
    ):
        raise ValueError(
            "--automatic-only diagnostics require all_to_allv_grouped_mat_mul "
            "cases with output_all_to_allv: true"
        )
    if args.auto_repeats is not None:
        if args.auto_repeats < 0:
            raise ValueError("--auto-repeats must be non-negative")
        for case in cases:
            case["accuracy"]["auto_repeats"] = args.auto_repeats
    if args.max_failure_dumps_per_role < -1:
        raise ValueError("--max-failure-dumps-per-role must be -1 or non-negative")
    if args.runtime_log_settle_seconds < 0:
        raise ValueError("--runtime-log-settle-seconds must be non-negative")
    if args.mssanitizer and len(cases) != 1:
        raise ValueError(
            "--mssanitizer requires exactly one selected case; pass one --case-id"
        )
    output_root = Path(args.output_dir or (HERE.parent / "validation_results" / timestamp() / "accuracy")).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    log_roots = []
    if not args.no_runtime_log_collection:
        log_roots = runtime_log_roots(
            os.environ,
            shmem_log_dirs=args.shmem_log_dir,
            plog_dirs=args.plog_dir,
        )
    environment = environment_fingerprint()
    environment["runtime_log_collection"] = {
        "enabled": not args.no_runtime_log_collection,
        "strategy": "per-launch new files and appended byte ranges",
        "settle_seconds": args.runtime_log_settle_seconds,
        "roots": log_roots,
    }
    write_json(output_root / "environment.json", environment)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["world_size"]].append(case)

    failed_launches = []
    runtime_log_results = []
    launcher_timing_groups = []
    for group_index, (world_size, group_cases) in enumerate(sorted(grouped.items())):
        resolved = output_root / "resolved_ep{}.json".format(world_size)
        write_resolved_cases(resolved, group_cases)
        environment = os.environ.copy()
        environment.setdefault("UDMA_MASTER_ADDR", "127.0.0.1")
        rendezvous_port = int(
            environment.get("VALIDATION_MASTER_PORT", environment.get("MASTER_PORT", "29500"))
        ) + group_index
        udma_port = int(environment.get("UDMA_MASTER_PORT", "29601")) + group_index
        if rendezvous_port == udma_port:
            raise RuntimeError("torchrun and UDMA bootstrap ports must differ")
        environment["UDMA_MASTER_PORT"] = str(udma_port)
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--master-addr=127.0.0.1", "--master-port={}".format(rendezvous_port),
            "--nproc-per-node={}".format(world_size), str(HERE / "accuracy_worker.py"),
            "--resolved-cases", str(resolved), "--output-dir", str(output_root),
        ]
        if args.automatic_only:
            command.append("--automatic-only")
        if check_reusable_out:
            command.append("--check-reusable-out")
        command.extend([
            "--max-failure-dumps-per-role",
            str(args.max_failure_dumps_per_role),
        ])
        if args.memory_safety:
            if args.hbm_reserve_gib < 0:
                raise ValueError("--hbm-reserve-gib must be non-negative")
            if args.hbm_poison_chunk_mib <= 0 or args.guard_mib <= 0:
                raise ValueError("poison chunk and guard sizes must be positive")
            if args.fingerprint_elements <= 0:
                raise ValueError("--fingerprint-elements must be positive")
            command.extend([
                "--memory-safety",
                "--hbm-reserve-bytes", str(int(args.hbm_reserve_gib * (1 << 30))),
                "--hbm-poison-chunk-bytes", str(args.hbm_poison_chunk_mib << 20),
                "--guard-bytes", str(args.guard_mib << 20),
                "--fingerprint-elements", str(args.fingerprint_elements),
            ])
        elif args.mssanitizer:
            command.extend([
                "--mssanitizer",
                "--mssanitizer-route-input", args.mssanitizer_route_input,
            ])
        launch_log = output_root / "torchrun_ep{}.log".format(world_size)
        runtime_log_output = (
            output_root / "runtime_logs" / "launch_ep{}".format(world_size)
        )
        write_json(output_root / "launch_ep{}.json".format(world_size), {
            "world_size": world_size,
            "command": command,
            "working_directory": str(output_root),
            "validation_master_addr": "127.0.0.1",
            "validation_master_port": rendezvous_port,
            "udma_master_addr": environment["UDMA_MASTER_ADDR"],
            "udma_master_port": udma_port,
            "udma_heap_size": environment.get("UDMA_HEAP_SIZE", "default"),
            "completion_protocol": "production",
            "automatic_only": args.automatic_only,
            "cross_rank_launch_alignment": not args.automatic_only,
            "check_reusable_out": check_reusable_out,
            "max_failure_dumps_per_role": args.max_failure_dumps_per_role,
            "console_log": launch_log.name,
            "runtime_log_manifest": (
                str(runtime_log_output / "manifest.json") if log_roots else None
            ),
        })
        print("+ " + " ".join(command), flush=True)
        group_timing = {"world_size": world_size}
        phase_started = time.perf_counter()
        log_snapshot = snapshot_runtime_logs(log_roots) if log_roots else None
        group_timing["runtime_log_snapshot_seconds"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        with launch_log.open("w", encoding="utf-8", errors="replace") as log_stream:
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
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
            returncode = process.wait()
        group_timing["torchrun_seconds"] = time.perf_counter() - phase_started
        if log_snapshot is not None:
            phase_started = time.perf_counter()
            if args.runtime_log_settle_seconds:
                time.sleep(args.runtime_log_settle_seconds)
            group_timing["runtime_log_settle_seconds"] = (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
            collected = collect_runtime_logs(
                log_snapshot, log_roots, runtime_log_output
            )
            group_timing["runtime_log_collection_seconds"] = (
                time.perf_counter() - phase_started
            )
            collected["world_size"] = world_size
            runtime_log_results.append(collected)
            print(
                "[accuracy] runtime_logs ep={} files={} bytes={} errors={} "
                "manifest={}".format(
                    world_size,
                    collected["captured_file_count"],
                    collected["captured_bytes"],
                    collected["error_count"],
                    collected["manifest"],
                ),
                flush=True,
            )
        else:
            group_timing["runtime_log_settle_seconds"] = 0.0
            group_timing["runtime_log_collection_seconds"] = 0.0
        group_timing["returncode"] = returncode
        launcher_timing_groups.append(group_timing)
        write_json(
            output_root / "launch_ep{}_timings.json".format(world_size),
            group_timing,
        )
        if returncode:
            failed_launches.extend(case["id"] for case in group_cases)
            if args.fail_fast:
                break

    rows = []
    for case in cases:
        summary_path = output_root / case["id"] / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for rank in summary["ranks"]:
                for report in rank["reports"]:
                    rows.append({
                        "case_id": case["id"], "rank": rank["rank"],
                        "path": report["path"],
                        "launch_index": report.get("launch_index"),
                        "global_launch_index": report.get("global_launch_index"),
                        "cross_rank_aligned": report.get("cross_rank_aligned"),
                        "allocation_mode": report.get("allocation_mode"),
                        "output_role": report.get("output_role"),
                        "passed": report["passed"],
                        "matched_ratio": report["matched_ratio"],
                        "max_abs_error": report["max_abs_error"],
                        "actual_nonfinite": report.get("actual_nonfinite", 0),
                        "memory_safety_passed": report.get("memory_safety_passed"),
                    })
        else:
            rows.append({"case_id": case["id"], "passed": False, "path": "launch_failed"})
    write_csv(
        output_root / "summary.csv", rows,
        [
            "case_id", "rank", "path", "launch_index",
            "global_launch_index", "cross_rank_aligned",
            "allocation_mode", "output_role", "passed", "matched_ratio",
            "max_abs_error", "actual_nonfinite", "memory_safety_passed",
        ],
    )
    overall = not failed_launches and rows and all(bool(row.get("passed")) for row in rows)
    launcher_elapsed = time.perf_counter() - launcher_started
    write_json(output_root / "summary.json", {
        "passed": bool(overall), "failed_launches": failed_launches,
        "case_count": len(cases), "results": rows,
        "completion_protocol": "production",
        "automatic_only": args.automatic_only,
        "cross_rank_launch_alignment": not args.automatic_only,
        "check_reusable_out": check_reusable_out,
        "auto_repeats_override": args.auto_repeats,
        "runtime_logs": runtime_log_results,
        "timings": {
            "elapsed_seconds": launcher_elapsed,
            "groups": launcher_timing_groups,
        },
    })
    print("accuracy results: {}".format(output_root), flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
