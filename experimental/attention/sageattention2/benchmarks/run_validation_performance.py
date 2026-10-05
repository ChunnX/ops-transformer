# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the Sage/FIA case matrix with torch.npu.Event and summarize results."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from cases import ALL_CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[c.name for c in ALL_CASES],
        default=[f"P-{index:02d}" for index in range(1, 8)],
    )
    parser.add_argument(
        "--warmup", type=int, default=int(os.environ.get("SAGEATTN_PERF_WARMUP", "50"))
    )
    parser.add_argument(
        "--repeats", type=int, default=int(os.environ.get("SAGEATTN_PERF_ITERS", "100"))
    )
    parser.add_argument("--seq", type=int)
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.repeats < 1
        or (args.seq is not None and args.seq < 1)
    ):
        parser.error("warmup, repeats and context length must be positive")
    root = Path(__file__).resolve().parent
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in ALL_CASES:
        name = case.name
        if name not in args.cases:
            continue
        directory = output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "application.json"
        if output.exists():
            raise FileExistsError(f"Result already exists: {output}")
        command = [
            sys.executable,
            str(root / "profile_performance.py"),
            "--case",
            name,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--output",
            str(output),
        ]
        if args.seq is not None:
            command += ["--seq", str(args.seq)]
        print(f"Benchmarking {name} with torch.npu.Event: FIA then Sage", flush=True)
        with (directory / "run.log").open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900
            )
        result = json.loads(output.read_text())
        row = {
            "case": name,
            "batch": result["shape"][0],
            "heads": result["shape"][1],
            "kv_heads": result["kv_heads"],
            "seq": result["shape"][2],
            "dim": result["shape"][3],
            "dtype": result["dtype"],
            "causal": result["causal"],
            "sage_ms": result["event"]["sage"]["mean_us"] / 1000,
            "fia_ms": result["event"]["fia"]["mean_us"] / 1000,
            "speedup": result["event_speedup"],
            "max_abs_vs_fia": result["max_abs_vs_fia"],
            "rmse_vs_fia": result["rmse_vs_fia"],
        }
        rows.append(row)
        print(
            f"{name}: Sage={row['sage_ms']:.2f} ms, FIA={row['fia_ms']:.2f} ms, "
            f"speedup={row['speedup']:.2f}x",
            flush=True,
        )
        # Save completed cases as we go, including during long-context runs.
        (output_root / "summary.json").write_text(
            json.dumps(
                {
                    "timing": "torch.npu.Event batch total / repeats; complete public API calls",
                    "warmup_calls": args.warmup,
                    "repeats": args.repeats,
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )
        with (output_root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
