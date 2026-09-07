# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect sequential public PyTorch / independent ACLNN performance evidence."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from cases import ALL_CASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fia-opapi", type=Path, required=True)
    p.add_argument("--fia-source-commit", required=True)
    p.add_argument(
        "--cases", nargs="+", default=[c[0] for c in ALL_CASES if c[0].startswith("P-")]
    )
    p.add_argument(
        "--interfaces", nargs="+", choices=["pt", "aclnn"], default=["pt", "aclnn"]
    )
    p.add_argument(
        "--modes", nargs="+", choices=["msprof", "event"], default=["msprof", "event"]
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--smooth-v", action="store_true")
    p.add_argument("--seq", type=int)
    a = p.parse_args()
    unknown = set(a.cases) - {c[0] for c in ALL_CASES}
    if unknown:
        p.error(f"unknown cases: {sorted(unknown)}")
    if a.warmup < 1 or a.repeats < 1 or (a.seq is not None and a.seq < 1):
        p.error("warmup, repeats and context length must be positive")
    root = Path(__file__).resolve().parents[1]
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    wrapper = a.output / "python_app.sh"
    wrapper.write_text("#!/bin/bash\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    env = dict(os.environ, FIA_OPAPI_LIBRARY=str(a.fia_opapi.resolve()))
    if a.smooth_v:
        env["SAGE_SMOOTH_V"] = "1"
    for mode in a.modes:
        for interface in a.interfaces:
            for name, seq, hk, batch, dtype, causal in ALL_CASES:
                if name not in a.cases:
                    continue
                directory = a.output / mode / interface / name
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / "application.json"
                if output.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite collected evidence: {output}"
                    )
                if interface == "pt":
                    command = [
                        "/bin/bash",
                        str(wrapper),
                        str(root / "benchmarks/profile_performance.py"),
                        "--case",
                        name,
                        "--warmup",
                        str(a.warmup),
                        "--repeats",
                        str(a.repeats),
                        "--output",
                        str(output),
                        "--fia-opapi",
                        str(a.fia_opapi.resolve()),
                        "--fia-source-commit",
                        a.fia_source_commit,
                        "--timing",
                        mode,
                    ]
                    if a.smooth_v:
                        command += ["--smooth-v"]
                    if a.seq:
                        command += ["--seq", str(a.seq)]
                else:
                    command = [
                        str(root / "build/benchmarks/benchmark_aclnn"),
                        str(a.seq or seq),
                        str(batch),
                        str(hk),
                        str(int(dtype == "bf16")),
                        str(int(causal)),
                        str(a.warmup),
                        str(a.repeats),
                        str(output),
                    ]
                run_env = dict(env)
                if interface == "aclnn":
                    # Both vendors use the SONAME libcust_opapi.so. The linked
                    # Sage API must load first; FIA is explicitly dlopened.
                    run_env["LD_LIBRARY_PATH"] = (
                        str(root / "build/opp/vendors/custom_transformer/op_api/lib")
                        + ":"
                        + env.get("LD_LIBRARY_PATH", "")
                    )
                    run_env["SAGE_BENCH_HEADS"] = "16" if name == "X-02" else "32"
                    run_env["SAGE_BENCH_DIM"] = "64" if name == "X-04" else "128"
                if mode == "msprof":
                    run_env["SAGE_PROFILE_ONLY"] = "1"
                    command = [
                        "msprof",
                        f"--output={directory}",
                        "--application=" + shlex.join(command),
                    ]
                print(
                    f"Collecting {mode}/{interface}/{name}, Sage then FIA", flush=True
                )
                with (directory / "run.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=run_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=900,
                    )
                result = json.loads(output.read_text())
                result.update(case=name, fia_source_commit=a.fia_source_commit)
                output.write_text(json.dumps(result, indent=2) + "\n")
    if "msprof" in a.modes:
        subprocess.run(
            [
                sys.executable,
                str(root / "tools/summarize_msprof.py"),
                str(a.output / "msprof"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
