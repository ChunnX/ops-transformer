# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
import importlib
import os
from pathlib import Path
import sys

import pytest

from cases import GOLDEN_DIR, select_cases


def pytest_addoption(parser):
    group = parser.getgroup("qbsa-layout-stc")
    group.addoption("--run-npu", action="store_true", help="Require actual NPU execution (missing runtime is an error)")
    group.addoption("--extended", action="store_true",
                    help="Also collect optional graph/contract/metadata/adapter diagnostics beyond the 36-case STC")
    group.addoption("--tier", choices=("all", "P0", "P1"), default="all")
    group.addoption("--layout", choices=("all", "TND", "BSND", "BNSD"), default="all")
    group.addoption("--device-id", type=int, default=int(os.environ.get("QBSA_DEVICE", "0")))
    group.addoption("--artifact-dir", default=None, help="Persistent per-case tensors/logs; defaults to pytest temp dir")


def pytest_generate_tests(metafunc):
    if "stc_case" in metafunc.fixturenames:
        cases = select_cases(metafunc.config.getoption("--tier"), metafunc.config.getoption("--layout"))
        metafunc.parametrize("stc_case", cases, ids=[c["name"] for c in cases])


def pytest_collection_modifyitems(config, items):
    if config.getoption("--extended"):
        return
    deselected = [item for item in items
                  if item.get_closest_marker("npu") and not item.get_closest_marker("accuracy")]
    items[:] = [item for item in items if item not in deselected]
    config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope="session")
def runtime(request):
    if not request.config.getoption("--run-npu"):
        pytest.skip("NPU acceptance not requested; pass --run-npu on Ascend 950")
    # Import failures intentionally propagate: --run-npu must never pass by skipping.
    import torch
    import torch_npu
    import torchair  # noqa: F401
    if not torch_npu.npu.is_available():
        pytest.fail("--run-npu requires an available Ascend NPU")
    assert hasattr(torch, "float8_e8m0fnu"), "Matching MXFP8-capable torch/torch_npu required"
    torch_npu.npu.set_device(request.config.getoption("--device-id"))
    ext = GOLDEN_DIR.parents[2] / "torch_ops_extension"
    sys.path.insert(0, str(ext))
    import custom_ops
    print(f"custom_ops={custom_ops.__file__}")
    sys.path.insert(0, str(GOLDEN_DIR))
    golden = importlib.import_module("qbsa_mxfp8_golden")
    golden._ensure_custom_ops_registered()
    return torch, torch_npu, golden


@pytest.fixture
def case_dir(request, tmp_path):
    root = request.config.getoption("--artifact-dir")
    if root is None:
        return tmp_path
    import re
    path = Path(root).resolve() / re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    path.mkdir(parents=True, exist_ok=True)
    return path
