# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AttnResFwd L0 pytest entry."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

import attn_res_fwd_golden as golden

REPO_ROOT = Path(__file__).resolve().parents[4]
OP_ROOT = Path(__file__).resolve().parents[2]
BF16_MERE_THRESHOLD = golden.BF16_MERE_THRESHOLD
L0_SHAPE = {"num_tokens": 544, "num_blocks": 12, "hidden_size": 7168, "seed": 0}


def _find_custom_set_env() -> Path | None:
    candidates = [
        Path("/data/kdy/cann/custompkgbase/vendors/custom_transformer/bin/set_env.bash"),
        REPO_ROOT / "build_out" / "vendors/custom_transformer/bin/set_env.bash",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _run_aclnn_example(device_id: int = 0) -> dict:
    set_env = _find_custom_set_env()
    if set_env is None:
        pytest.skip("custom opp package not installed; run build.sh --pkg --ops=attn_res_fwd first")

    cmd = (
        f"source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && "
        f"source {set_env} && "
        f"export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0-beta.1/opp && "
        f"cd {REPO_ROOT} && "
        f"bash build.sh --run_example attn_res_fwd eager cust --soc=ascend910b "
        f"--example_name=test_aclnn_attn_res_fwd"
    )
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=3600)
    output = proc.stdout + proc.stderr
    print(output)
    mere_match = re.search(r"MERE=([0-9.eE+-]+)", output)
    mare_match = re.search(r"MARE=([0-9.eE+-]+)", output)
    result_match = re.search(r"result=(pass|fail)", output)
    if mere_match is None:
        pytest.fail(f"aclnn example failed (rc={proc.returncode}):\n{output[-4000:]}")
    return {
        "returncode": proc.returncode,
        "mere": float(mere_match.group(1)) if mere_match else None,
        "mare": float(mare_match.group(1)) if mare_match else None,
        "result": result_match.group(1) if result_match else ("pass" if proc.returncode == 0 else "fail"),
        "output": output,
    }


@pytest.mark.ci
def test_attn_res_fwd_cpu_golden_l0():
    stats = golden.run_cpu_golden_self_test(**L0_SHAPE)
    print(f"CPU golden self-test: {stats}")
    assert stats["result"] == "pass"
    assert stats["mere"] < BF16_MERE_THRESHOLD


@pytest.mark.ci
def test_attn_res_fwd_npu_precision_l0():
    stats = _run_aclnn_example(device_id=int(os.environ.get("DEVICE_ID", "0")))
    print(f"NPU aclnn example: {stats}")
    assert stats["mere"] is not None, "MERE not found in example output"
    assert stats["mare"] is not None, "MARE not found in example output"
    assert stats["mere"] < BF16_MERE_THRESHOLD, f"MERE {stats['mere']} >= {BF16_MERE_THRESHOLD}"
    assert stats["result"] == "pass"
