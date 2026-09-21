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

import os
import sys
from pathlib import Path

import pytest


PYTEST_ROOT = Path(__file__).resolve().parent
if str(PYTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTEST_ROOT))


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
