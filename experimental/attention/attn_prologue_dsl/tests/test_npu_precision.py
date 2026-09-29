# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""通过已安装 wheel 的公共 PyTorch 接口检查 decode 与 prefill 全输出。"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_npu_precision(tmp_path):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("An Ascend NPU is required")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_generalization_accuracy.py")),
        "--package",
        os.environ.get("ATTN_PROLOGUE_PACKAGE", "cann_ops_transformer"),
        "--device",
        os.environ.get("ATTN_PROLOGUE_DEVICE", "npu:0"),
        "--case-id",
        "decode-b1-s1-t1",
        "--case-id",
        "decode-b12-s6-t72",
        "--case-id",
        "prefill-b1-s1024-t1024",
        "--output",
        str(tmp_path / "accuracy.jsonl"),
    ]
    # 子进程隔离 CPU schema 测试的注册，不允许回退到测试 mock。
    subprocess.run(command, check=True, timeout=1800)
