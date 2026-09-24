# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""为 NPU 五档对拍生成输入 bin，并用 C++ encode_one_core 孪生写 expect（禁用 Python 边界漂移）。

步骤：
  1) 本脚本写 kv_bf16 / matrices / meta
  2) 编译并运行 regen_npu_expects_from_twin 覆盖 expect_*
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

CASES = [
    ("gqa_decode_b1", 1, 8, 3),
    ("gqa_decode_b64", 64, 8, 3),
    ("gqa_prefill_t2048", 2048, 8, 3),
    ("gqa_mse2_t2048", 2048, 8, 2),
    ("gqa_mse4_t2048", 2048, 8, 4),
]
HEAD = 128


def soft_bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = (u & np.uint32(0xFFFF0000)).astype(np.uint32)
    return u.view(np.float32)


def float_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    return (u >> np.uint32(16)).astype(np.uint16)


def main() -> int:
    rng = np.random.default_rng(20260923)
    data = ROOT / "data"
    rot = np.fromfile(data / "rotation_matrix.bin", dtype=np.float32).reshape(
        HEAD, HEAD
    )
    qjl = np.fromfile(data / "qjl_matrix.bin", dtype=np.float32).reshape(HEAD, HEAD)

    out_root = data / "npu_cases"
    out_root.mkdir(parents=True, exist_ok=True)

    for name, tokens, heads, bits in CASES:
        case_dir = out_root / name
        case_dir.mkdir(exist_ok=True)
        x = rng.uniform(-2, 2, size=(tokens, heads, HEAD)).astype(np.float32)
        x_bf = soft_bf16(x)

        float_to_bf16_u16(x_bf).tofile(case_dir / "kv_bf16.bin")
        rot.tofile(case_dir / "rotation_matrix.bin")
        qjl.tofile(case_dir / "qjl_matrix.bin")
        # expect_* 由 C++ 孪生写，避免 Python 边界漂移
        (case_dir / "meta.txt").write_text(
            f"tokens={tokens}\nheads={heads}\nbits={bits}\nhead_dim={HEAD}\n",
            encoding="utf-8",
        )
        print(f"wrote inputs {case_dir} T={tokens} H={heads} bits={bits}")

    twin = ROOT / "regen_npu_expects_from_twin"
    src = ROOT / "regen_npu_expects_from_twin.cpp"
    subprocess.check_call(
        ["g++", "-O2", "-std=c++17", "-o", str(twin), str(src)], cwd=str(ROOT)
    )
    subprocess.check_call([str(twin), str(out_root)], cwd=str(ROOT))
    print("ALL PASS gen_npu_case_goldens (C++ twin expects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
