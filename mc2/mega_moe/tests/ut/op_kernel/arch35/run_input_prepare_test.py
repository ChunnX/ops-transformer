#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile both core variants of production input preparation helpers against queued CPU pipelines."""

from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(
        prefix="mega_moe_input_prepare_test_"
    ) as temporary:
        folder = Path(temporary)
        shutil.copyfile(
            here / "input_prepare_mock" / "mega_moe_input_prepare_mock.h",
            folder / "kernel_operator.h",
        )
        common = [
            "g++",
            "-std=c++20",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(folder),
        ]
        objects = []
        for core in ("AIV", "AIC"):
            obj = folder / f"{core}.o"
            command = common + [
                "-c",
                str(here / "input_prepare_core.cpp"),
                "-o",
                str(obj),
            ]
            if core == "AIC":
                command.append("-DTEST_AIC")
            subprocess.run(command, check=True, timeout=60)
            objects.append(str(obj))
        executable = folder / "test"
        subprocess.run(
            common
            + [str(here / "test_input_prepare.cpp"), *objects, "-o", str(executable)],
            check=True,
            timeout=60,
        )
        subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    main()
