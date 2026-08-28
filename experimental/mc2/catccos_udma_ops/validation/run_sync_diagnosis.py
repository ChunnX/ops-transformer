# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Run isolated AllToAllV staging-readiness diagnostics with one wheel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from .case_schema import load_case_manifest, select_cases
    from .common import timestamp, write_json
except ImportError:
    from case_schema import load_case_manifest, select_cases
    from common import timestamp, write_json


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"
DIAGNOSTIC_MODES = (
    "baseline",
    "strong_ready",
    "strong_ready_cache_safe",
)


def classify_mode_results(results):
    """Turn the three isolated runs into the next debugging decision."""
    by_mode = {item["mode"]: item for item in results}
    if any(
        mode not in by_mode or not by_mode[mode]["summary_available"]
        for mode in DIAGNOSTIC_MODES
    ):
        return {
            "classification": "infrastructure_or_runtime_failure",
            "conclusive": False,
            "next_action": (
                "Inspect the missing mode's launcher and torchrun logs before "
                "drawing a synchronization conclusion."
            ),
        }

    baseline_passed = by_mode["baseline"]["passed"]
    strong_ready_passed = by_mode["strong_ready"]["passed"]
    cache_safe_passed = by_mode["strong_ready_cache_safe"]["passed"]
    if baseline_passed:
        return {
            "classification": "baseline_did_not_reproduce",
            "conclusive": False,
            "next_action": (
                "Increase --baseline-launches or rerun on the environment that "
                "reproduces the consecutive-launch failure."
            ),
        }
    if strong_ready_passed and cache_safe_passed:
        return {
            "classification": "completion_publication_order",
            "conclusive": True,
            "next_action": (
                "Promote per-stage/per-source write-with-notify generations; "
                "cache bypass is not required by this result."
            ),
        }
    if cache_safe_passed:
        return {
            "classification": "receiver_cache_visibility",
            "conclusive": True,
            "next_action": (
                "Keep strong ready generations and add the narrowest staging "
                "cache-visibility treatment that remains correct."
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


def _read_mode_result(mode, output_dir, returncode, launches):
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
        "summary_available": summary is not None,
        "passed": bool(
            summary and summary.get("passed") and returncode == 0
        ),
        "failed_report_count": len(failed_reports),
        "failed_reports": failed_reports,
        "output_dir": str(output_dir),
        "summary": str(summary_path) if summary is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--launches", type=int, default=6,
        help="total automatic launches per rank in each strong mode",
    )
    parser.add_argument(
        "--baseline-launches", type=int, default=3,
        help="total automatic launches per rank in baseline mode",
    )
    parser.add_argument(
        "--max-failure-dumps-per-role", type=int, default=1,
        help="large dumps kept per rank for GMM and AllToAllV outputs",
    )
    args = parser.parse_args()

    if args.launches < 2 or args.baseline_launches < 2:
        raise ValueError("launch counts must be at least 2")
    if args.max_failure_dumps_per_role < 0:
        raise ValueError("--max-failure-dumps-per-role must be non-negative")

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
        "strong_ready": args.launches,
        "strong_ready_cache_safe": args.launches,
    }
    plan = {
        "case_manifest": str(Path(args.cases).resolve()),
        "case_id": args.case_id,
        "case": case,
        "modes": list(DIAGNOSTIC_MODES),
        "requested_launches_per_rank": requested_launches,
        "automatic_only": True,
        "checks": ["operator_output", "all_to_allv_export"],
        "max_failure_dumps_per_role": args.max_failure_dumps_per_role,
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
        process = subprocess.run(command, env=environment, check=False)
        result = _read_mode_result(
            mode, mode_output, process.returncode, launches
        )
        mode_results.append(result)
        print(
            "[sync-diagnosis] mode={} passed={} returncode={} failed_reports={}".format(
                mode, result["passed"], process.returncode,
                result["failed_report_count"],
            ),
            flush=True,
        )

    decision = classify_mode_results(mode_results)
    aggregate = {
        "case_id": args.case_id,
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
