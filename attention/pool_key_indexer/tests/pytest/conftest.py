# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Pytest configuration and fixtures for PoolKeyIndexer tests.

全局选项:
  --ref-device {cpu,npu}   精度/GE 套件的标杆设备(默认 cpu 金标;
                           npu 为 NPU 组合标杆, 验证两标杆一致性)
环境变量:
  PKI_DEVICE_ID           NPU 设备号(默认 0, 多用户环境可切换空闲设备)
"""

import os

import pytest
import torch  # noqa: F401  (torch_npu 依赖加载顺序)
import torch_npu


def pytest_addoption(parser):
    parser.addoption(
        "--ref-device",
        action="store",
        default="cpu",
        choices=("cpu", "npu"),
        help="标杆设备: cpu(CPU 金标, 默认) 或 npu(NPU 组合标杆)",
    )


@pytest.fixture(scope="session")
def ref_device(request):
    """精度/GE 套件标杆设备(cpu / npu)。"""
    return request.config.getoption("--ref-device")


@pytest.fixture(scope="session", autouse=True)
def npu_init():
    """Initialize NPU device once per test session."""
    torch_npu.npu.set_device(int(os.environ.get("PKI_DEVICE_ID", "0")))
    yield
