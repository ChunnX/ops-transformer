#!/usr/bin/env python3
# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check host/api/benchmark C++ using the actual OPP compilation database.

AscendC device extensions require bisheng and are not supported by stock
clang-tidy. Do not stub device APIs and report that as a kernel analysis.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--compile-commands", type=Path, default=REPO / "build/compile_commands.json")
    args = parser.parse_args()
    entries = json.loads(args.compile_commands.read_text())
    selected = {}
    for entry in entries:
        path = Path(entry["file"]).resolve()
        if path.parent not in (ROOT / "op_host", ROOT / "op_api"):
            continue
        argv = entry.get("arguments") or shlex.split(entry["command"])
        # CMake's __FILE__ definitions contain make expressions. They only
        # affect log labels, and conflict with clang's built-in definition.
        argv = [arg for arg in argv if not arg.startswith("-D__FILE__=")]
        selected[str(path)] = dict(directory=entry["directory"], file=str(path), arguments=argv)
    if len(selected) != 5:
        raise RuntimeError(f"Expected 5 host/api sources, found {len(selected)}; run install_opp.sh first")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    bench = ROOT / "benchmarks/benchmark_aclnn.cpp"
    selected[str(bench)] = dict(
        directory=str(ROOT),
        file=str(bench),
        arguments=["c++", "-std=c++17", "-I" + str(cann / "include"), "-I" + str(ROOT / "op_api"), "-c", str(bench)],
    )
    output = ROOT / "build/clang-tidy"
    output.mkdir(parents=True, exist_ok=True)
    (output / "compile_commands.json").write_text(json.dumps(list(selected.values()), indent=2) + "\n")
    logs = []
    status = 0
    for path in selected:
        command = [
            "clang-tidy",
            "-p",
            str(output),
            "--extra-arg=-Wno-unknown-warning-option",
            "--extra-arg=-Wno-macro-redefined",
            path,
        ]
        if args.fix:
            command.append("--fix")
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs.append(f"=== {path} ===\n{result.stdout}")
        print(f"{Path(path).name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
        if result.returncode:
            print(result.stdout, flush=True)
            status = 1
    (output / "results.log").write_text("\n".join(logs))
    print("Device kernels: validate separately with the complete CANN build and NPU tests.")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
