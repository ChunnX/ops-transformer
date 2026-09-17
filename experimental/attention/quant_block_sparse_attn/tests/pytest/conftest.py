#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
import sys
from pathlib import Path

import pytest


PYTEST_ROOT = Path(__file__).resolve().parent
if str(PYTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTEST_ROOT))

from common_utils.op_runtime import configure_custom_opp_paths


def _prepend_env_path(env_name, paths):
    current = os.environ.get(env_name, "")
    current_parts = [part for part in current.split(":") if part]
    new_parts = [
        str(path) for path in paths if path.exists() and str(path) not in current_parts
    ]
    if new_parts:
        os.environ[env_name] = ":".join(new_parts + current_parts)


def _runtime_library_dirs():
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")
    candidates.extend(
        [
            Path(sys.executable).resolve().parents[1] / "lib",
        ]
    )
    deduped = []
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _preload_runtime_libraries():
    lib_dirs = _runtime_library_dirs()
    _prepend_env_path("LD_LIBRARY_PATH", lib_dirs)
    for lib_dir in lib_dirs:
        libstdcxx = lib_dir / "libstdc++.so.6"
        libgcc = lib_dir / "libgcc_s.so.1"
        if not libstdcxx.exists():
            continue
        if libgcc.exists():
            ctypes.CDLL(str(libgcc), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)
        return


_preload_runtime_libraries()
configure_custom_opp_paths()


def pytest_report_header(config):
    return "QBSA ASCEND_CUSTOM_OPP_PATH=" + os.environ.get(
        "ASCEND_CUSTOM_OPP_PATH", "<unset; use installed OPP>"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--check-accuracy",
        action="store_true",
        default=False,
        help="Enable detailed accuracy comparison including cosine similarity and error tables",
    )
    parser.addoption(
        "--force-regenerate",
        action="store_true",
        default=False,
        help="Force regenerate .pt test case files even if they already exist",
    )


@pytest.fixture(scope="session")
def check_accuracy(request):
    """Fixture that returns True when --check-accuracy flag is set."""
    return request.config.getoption("--check-accuracy")


@pytest.fixture(scope="session")
def force_regenerate(request):
    """Fixture that returns True when --force-regenerate flag is set."""
    return request.config.getoption("--force-regenerate")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "check_accuracy: mark test to run with detailed accuracy comparison",
    )
