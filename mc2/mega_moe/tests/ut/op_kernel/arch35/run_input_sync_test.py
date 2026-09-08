#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Compile input notification and count DMA helpers against a CPU pipeline model."""

from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="mega_moe_input_sync_test_") as temporary:
        mock_include = Path(temporary) / "include"
        mock_include.mkdir()
        shutil.copyfile(
            here / "input_sync_mock" / "mega_moe_input_sync_mock.h",
            mock_include / "kernel_operator.h",
        )
        for core in ("AIV", "AIC"):
            executable = Path(temporary) / core.lower()
            command = [
                "g++",
                "-std=c++20",
                "-O2",
                "-pthread",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-I",
                str(mock_include),
                str(here / "test_input_sync.cpp"),
                "-o",
                str(executable),
            ]
            if core == "AIC":
                command.append("-DTEST_AIC")
            subprocess.run(command, check=True, timeout=60)
            subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    main()
