# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Manifest-driven multi-case accuracy launcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .case_schema import load_case_manifest, select_cases, write_resolved_cases
    from .common import environment_fingerprint, timestamp, write_csv, write_json
except ImportError:
    from case_schema import load_case_manifest, select_cases, write_resolved_cases
    from common import environment_fingerprint, timestamp, write_csv, write_json


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-tag", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    cases = select_cases(
        load_case_manifest(args.cases), mode="accuracy",
        case_ids=args.case_id, include_tags=args.include_tag,
    )
    output_root = Path(args.output_dir or (HERE.parent / "validation_results" / timestamp() / "accuracy")).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "environment.json", environment_fingerprint())
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["world_size"]].append(case)

    failed_launches = []
    for group_index, (world_size, group_cases) in enumerate(sorted(grouped.items())):
        resolved = output_root / "resolved_ep{}.json".format(world_size)
        write_resolved_cases(resolved, group_cases)
        environment = os.environ.copy()
        environment.setdefault("MTE_MASTER_ADDR", "127.0.0.1")
        rendezvous_port = int(
            environment.get("VALIDATION_MASTER_PORT", environment.get("MASTER_PORT", "29500"))
        ) + group_index
        mte_port = int(environment.get("MTE_MASTER_PORT", "29601")) + group_index
        if rendezvous_port == mte_port:
            raise RuntimeError("torchrun and MTE bootstrap ports must differ")
        environment["MTE_MASTER_PORT"] = str(mte_port)
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--master-addr=127.0.0.1", "--master-port={}".format(rendezvous_port),
            "--nproc-per-node={}".format(world_size), str(HERE / "accuracy_worker.py"),
            "--resolved-cases", str(resolved), "--output-dir", str(output_root),
        ]
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=str(output_root), env=environment)
        if result.returncode:
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
                        "path": report["path"], "passed": report["passed"],
                        "matched_ratio": report["matched_ratio"],
                        "max_abs_error": report["max_abs_error"],
                    })
        else:
            rows.append({"case_id": case["id"], "passed": False, "path": "launch_failed"})
    write_csv(
        output_root / "summary.csv", rows,
        ["case_id", "rank", "path", "passed", "matched_ratio", "max_abs_error"],
    )
    overall = not failed_launches and rows and all(bool(row.get("passed")) for row in rows)
    write_json(output_root / "summary.json", {
        "passed": bool(overall), "failed_launches": failed_launches,
        "case_count": len(cases), "results": rows,
    })
    print("accuracy results: {}".format(output_root), flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
