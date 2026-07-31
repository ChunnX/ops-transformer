"""Compare device outputs with the migration golden using the agreed threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from case_utils import dump_json
from golden_a2_a5 import OUTPUTS, read_tensor


def compare_tensor(actual: torch.Tensor, golden: torch.Tensor):
    actual_fp32 = actual.float()
    golden_fp32 = golden.float()
    finite = torch.isfinite(actual_fp32) & torch.isfinite(golden_fp32)
    absolute = torch.abs(actual_fp32 - golden_fp32)
    metric = torch.where(
        torch.abs(golden_fp32) < 1.0,
        absolute,
        absolute / torch.abs(golden_fp32),
    )
    passed = finite & (metric <= 1e-3)
    failures = int((~passed).sum().item())
    return {
        "passed": failures == 0,
        "elements": int(golden.numel()),
        "failures": failures,
        "max_abs_error": float(absolute.max().item()) if absolute.numel() else 0.0,
        "max_metric": float(metric[finite].max().item()) if finite.any() else float("inf"),
    }


def compare_case(case_dir: Path):
    manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    report = {"case_id": manifest["case"]["id"], "outputs": {}, "passed": True}
    for name in OUTPUTS:
        description = manifest["outputs"][name]
        golden = read_tensor(
            case_dir / description["golden_file"],
            description["shape"],
            description["dtype"],
        )
        actual = read_tensor(
            case_dir / "outputs" / f"{name}.bin",
            description["shape"],
            description["dtype"],
        )
        item = compare_tensor(actual, golden)
        report["outputs"][name] = item
        report["passed"] = bool(report["passed"] and item["passed"])
    dump_json(case_dir / "compare" / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    report = compare_case(args.case_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
