# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression: custom aclnn works in a process that has never imported torch."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("SAGEATTN_TEST_STANDALONE") != "1",
    reason="enable standalone C benchmark regression explicitly",
)
@pytest.mark.parametrize("bf16,causal,hk", [(0, 0, 32), (1, 1, 8)])
def test_standalone_aclnn(tmp_path, bf16, causal, hk):
    binary = ROOT / "build/benchmarks/benchmark_aclnn"
    assert binary.is_file(), "run benchmarks/build_aclnn_benchmark.sh first"
    output = tmp_path / "result.json"
    subprocess.run(
        [
            str(binary),
            "128",
            "1",
            str(hk),
            str(bf16),
            str(causal),
            "2",
            "3",
            str(output),
        ],
        check=True,
        timeout=180,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert result["path"] == "standalone C++ ACL runtime"
    assert result["sage_ms"] > 0 and result["fia_ms"] > 0
    # Smoke-level independent FIA check; full algorithm accuracy is tested elsewhere.
    assert result["rmse_vs_fia"] < 0.04
    assert result["max_abs_vs_fia"] < 0.3
