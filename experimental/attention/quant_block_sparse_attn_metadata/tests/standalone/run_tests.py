# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile real metadata production sources with small CANN container stubs.

This verifies CPU scheduling/validation, not the CANN ABI or NPU instructions.
Usage: python run_tests.py --cxx g++ [--build-dir PATH]
       python run_tests.py --zig /path/to/zig [--build-dir PATH]
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    compiler = parser.add_mutually_exclusive_group()
    compiler.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    compiler.add_argument("--zig")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    build = args.build_dir or Path(tempfile.mkdtemp(prefix="qbsa-layout-"))
    build.mkdir(parents=True, exist_ok=True)
    build = build.resolve()
    stubs = build / "stubs"
    for name in (
        "cpu_context.h",
        "cpu_tensor.h",
        "cpu_kernel.h",
        "log.h",
        "status.h",
        "opdev/format_utils.h",
        "opdev/op_log.h",
        "opdev/data_type_utils.h",
        "opdev/tensor_view_utils.h",
    ):
        wrapper = stubs / name
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text('#include "framework_stubs.h"\n', encoding="utf-8")
    command = [args.zig, "c++"] if args.zig else [args.cxx]
    if args.zig and os.name == "nt":
        command += ["-target", "x86_64-windows-gnu"]
    exe = build / (
        "test_layout_metadata.exe" if os.name == "nt" else "test_layout_metadata"
    )
    command += [
        "-std=c++17",
        "-O1",
        "-g",
        "-I" + str(stubs),
        "-I" + str(source),
        str(source / "test_layout_metadata.cpp"),
        str(
            source.parents[1]
            / "op_kernel_aicpu/quant_block_sparse_attn_metadata_aicpu.cpp"
        ),
        "-o",
        str(exe),
    ]
    print("Compiling production ACLNN checks and AICPU metadata scheduler", flush=True)
    subprocess.run(command, check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
