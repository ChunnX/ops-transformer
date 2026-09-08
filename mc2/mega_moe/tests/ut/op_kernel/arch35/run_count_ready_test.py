#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the production count VF and reader against a CPU register/pipeline model."""

from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="mega_moe_count_ready_test_") as temporary:
        mock_include = Path(temporary) / "include"
        mock_include.mkdir()
        shutil.copyfile(
            here / "count_ready_mock" / "mega_moe_count_ready_mock.h",
            mock_include / "kernel_operator.h",
        )
        for core in ("AIV", "AIC"):
            executable = Path(temporary) / core.lower()
            command = [
                "g++",
                "-std=c++20",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-I",
                str(mock_include),
                str(here / "test_count_ready.cpp"),
                "-o",
                str(executable),
            ]
            if core == "AIC":
                command.append("-DTEST_AIC")
            subprocess.run(command, check=True, timeout=60)
            subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    main()
