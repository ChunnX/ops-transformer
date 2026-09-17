# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Convert a compatible original-example msprof capture into baseline JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .case_schema import load_case_manifest, performance_workload_digest, select_cases
    from .common import environment_fingerprint, write_json
    from .msprof_parser import parse_msprof_case
except ImportError:
    from case_schema import load_case_manifest, performance_workload_digest, select_cases
    from common import environment_fingerprint, write_json
    from msprof_parser import parse_msprof_case


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    case = select_cases(
        load_case_manifest(args.cases), mode="performance", case_ids=[args.case_id]
    )[0]
    report = parse_msprof_case(args.profile_dir, case)
    report["performance_workload_digest"] = performance_workload_digest(case)
    report["environment"] = environment_fingerprint()
    report["baseline_source"] = str(Path(args.profile_dir).resolve())
    payload = {"passed": report["valid"], "environment": report["environment"], "cases": [report]}
    write_json(args.output, payload)
    print("baseline: {}".format(Path(args.output).resolve()))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
