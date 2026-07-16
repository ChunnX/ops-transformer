"""Repeat one generated case and record accuracy and output variation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from case_utils import dump_json
from golden_a2_a5 import OUTPUTS, read_tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_with_first(case_dir: Path, first_dir: Path, manifest):
    comparison = {}
    for name in OUTPUTS:
        description = manifest["outputs"][name]
        current = read_tensor(
            case_dir / "outputs" / f"{name}.bin",
            description["shape"],
            description["dtype"],
        )
        first = read_tensor(
            first_dir / f"{name}.bin",
            description["shape"],
            description["dtype"],
        )
        difference = (current.float() - first.float()).abs()
        comparison[name] = {
            "bitwise_equal": torch.equal(current, first),
            "max_abs_variation": float(difference.max().item()),
        }
    return comparison


def run(case_dir: Path, device_id: int, repeat: int, result_name: str):
    manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    result_dir = case_dir / result_name
    first_dir = result_dir / "first_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    first_dir.mkdir(parents=True, exist_ok=True)
    iterations = []
    baseline_iteration = None

    for iteration in range(1, repeat + 1):
        log_path = result_dir / f"iteration_{iteration:02d}.log"
        report_path = case_dir / "run_report.json"
        report_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_case.py")),
            str(case_dir),
            "--device-id",
            str(device_id),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)

        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        archived_report = result_dir / f"run_report_{iteration:02d}.json"
        if report:
            dump_json(archived_report, report)

        checksums = {}
        variation = {}
        outputs_exist = completed.returncode == 0 and bool(report.get("passed", False))
        if outputs_exist:
            for name in OUTPUTS:
                output_path = case_dir / "outputs" / f"{name}.bin"
                if not output_path.exists():
                    outputs_exist = False
                    checksums = {}
                    break
                checksums[name] = file_sha256(output_path)

        if outputs_exist:
            if baseline_iteration is None:
                baseline_iteration = iteration
                for name in OUTPUTS:
                    shutil.copyfile(
                        case_dir / "outputs" / f"{name}.bin",
                        first_dir / f"{name}.bin",
                    )
                variation = {
                    name: {"bitwise_equal": True, "max_abs_variation": 0.0}
                    for name in OUTPUTS
                }
            else:
                variation = compare_with_first(case_dir, first_dir, manifest)

        iterations.append(
            {
                "iteration": iteration,
                "returncode": completed.returncode,
                "passed": bool(report.get("passed", False)),
                "elapsed_ms": report.get("elapsed_ms"),
                "checksums": checksums,
                "variation_from_first": variation,
                "log": log_path.name,
                "report": archived_report.name if report else None,
            }
        )
        dump_json(
            result_dir / "stability_report.json",
            {
                "case_id": manifest["case"]["id"],
                "repeat": repeat,
                "completed_iterations": len(iterations),
                "baseline_iteration": baseline_iteration,
                "passed": all(item["returncode"] == 0 and item["passed"] for item in iterations)
                and len(iterations) == repeat,
                "iterations": iterations,
            },
        )

    return json.loads((result_dir / "stability_report.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--result-name", default="stability")
    args = parser.parse_args()
    report = run(args.case_dir, args.device_id, args.repeat, args.result_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
