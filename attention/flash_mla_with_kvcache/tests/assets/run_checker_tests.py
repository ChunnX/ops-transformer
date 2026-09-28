# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile and run host checker regressions using an existing MLA build configuration."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=root / "build")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(
        item
        for item in entries
        if item["file"].endswith(
            "/flash_mla_with_kvcache/op_host/checkers/common_checker.cpp"
        )
    )
    command = entry.get("arguments") or shlex.split(entry["command"])
    flags = []
    index = 1
    while index < len(command):
        flag = command[index]
        if flag in ("-o", "-MF", "-MT", "-MQ"):
            index += 2
            continue
        if flag not in ("-c", "-MD", "-MMD", entry["file"]) and not flag.startswith(
            "-DNDEBUG"
        ):
            flags.append(flag)
        index += 1

    cache = (build / "CMakeCache.txt").read_text()
    match = re.search(
        r"^CUSTOM_ASCEND_CANN_PACKAGE_PATH:[^=]+=(.+)$", cache, re.MULTILINE
    )
    if match is None:
        parser.error("CMakeCache.txt must contain CUSTOM_ASCEND_CANN_PACKAGE_PATH")
    lib = Path(match.group(1)) / "lib64"
    host = root / "attention/flash_mla_with_kvcache/op_host"
    sources = [
        Path(__file__).with_name("test_checkers.cpp"),
        *sorted((host / "checkers").glob("*.cpp")),
        host / "flash_mla_with_kvcache_tiling_info_parser.cpp",
        root / "attention/common/op_host/fia_tiling_shape.cpp",
        root / "third_party/opbase/src/op_common/log/log.cpp",
    ]
    # Access private checks only in this standalone test build. Production builds
    # keep their original access control and do not depend on this runner.
    with tempfile.TemporaryDirectory(prefix="mla-checkers-") as tmp:
        binary = Path(tmp) / "test_checkers"
        subprocess.run(
            [
                command[0],
                *flags,
                "-UNDEBUG",
                "-fno-access-control",
                "-ffunction-sections",
                "-fdata-sections",
                "-Wl,--gc-sections",
                *map(str, sources),
                "-o",
                str(binary),
                f"-L{lib}",
                f"-Wl,-rpath,{lib}",
                "-lnnopbase",
                "-lgraph",
                "-lgraph_base",
                "-lexe_graph",
                "-lregister",
                "-lerror_manager",
                "-lascendalog",
                "-lunified_dlog",
                "-lc_sec",
            ],
            cwd=entry["directory"],
            check=True,
        )
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        subprocess.run([str(binary)], env=env, check=True)


if __name__ == "__main__":
    main()
