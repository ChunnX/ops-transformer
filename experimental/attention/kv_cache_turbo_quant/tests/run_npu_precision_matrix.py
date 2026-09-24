# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU 输出 × 任务书 §3.2：relMSE / IP95。

流程：
  1) 编译运行 npu_export_for_metric → 各 case/npu_got/*.bin
  2) 本脚本读 kv + 矩阵 + npu_got，算重构误差与内积 95 分位

口径（任务书 §3.2）：
  relative_MSE < 0.05（mse2 可 WARN）
  IP relative err 95th < 0.1（硬门槛）

小 shape 另用 npu_precision_cases exact；本脚本覆盖任务书五档。
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "golden"))
from cpu_golden import CENTROIDS  # noqa: E402

# 复用 CPU 矩阵脚本中的重构/IP 逻辑
from run_cpu_precision_matrix import (  # noqa: E402
    HEAD,
    IP_P95_LIMIT,
    REL_MSE_LIMIT,
    estimate_inner_product,
    ip_rel_err_p95,
    reconstruct_mse,
    relative_mse,
    unpack_bits,
)

CASES = [
    ("gqa_decode_b1", 1, 8, 3, "exact_ok"),  # 小 shape：§3.2 也应过
    ("gqa_decode_b64", 64, 8, 3, "exact_ok"),
    ("gqa_prefill_t2048", 2048, 8, 3, "metric"),
    ("gqa_mse2_t2048", 2048, 8, 2, "metric"),
    ("gqa_mse4_t2048", 2048, 8, 4, "metric"),
]


def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def load_case(case_dir: Path, tokens: int, heads: int, bits: int):
    kv_u16 = np.fromfile(case_dir / "kv_bf16.bin", dtype=np.uint16).reshape(
        tokens, heads, HEAD
    )
    x = bf16_u16_to_f32(kv_u16)
    rot = np.fromfile(case_dir / "rotation_matrix.bin", dtype=np.float32).reshape(
        HEAD, HEAD
    )
    qjl = np.fromfile(case_dir / "qjl_matrix.bin", dtype=np.float32).reshape(HEAD, HEAD)
    got = case_dir / "npu_got"
    idx_bytes = (HEAD * bits + 7) // 8
    qjl_bytes = (HEAD + 7) // 8
    idx = np.fromfile(got / "idx.bin", dtype=np.uint8).reshape(tokens, heads, idx_bytes)
    qj = np.fromfile(got / "qjl.bin", dtype=np.uint8).reshape(tokens, heads, qjl_bytes)
    norm = bf16_u16_to_f32(
        np.fromfile(got / "norm_bf16.bin", dtype=np.uint16).reshape(tokens, heads)
    )
    gamma = bf16_u16_to_f32(
        np.fromfile(got / "gamma_bf16.bin", dtype=np.uint16).reshape(tokens, heads)
    )
    return x, rot, qjl, idx, qj, norm, gamma


def ensure_export(data_root: Path) -> None:
    export_bin = Path("/tmp/npu_export_for_metric")
    src = ROOT / "npu_export_for_metric.cpp"
    # 由外层 shell 负责带 ACL 链接编译；此处若已存在则直接跑
    if not export_bin.exists():
        raise SystemExit(
            "missing /tmp/npu_export_for_metric — compile npu_export_for_metric.cpp first "
            "(same flags as npu_precision_cases)"
        )
    subprocess.check_call([str(export_bin), str(data_root), "all"])


def main() -> int:
    rng = np.random.default_rng(20260923)
    data_root = ROOT / "data" / "npu_cases"
    ensure_export(data_root)

    log_lines = []
    n_fail = 0
    for name, tokens, heads, bits, _kind in CASES:
        case_dir = data_root / name
        x, rot, qjl, idx, qj, norm, gamma = load_case(case_dir, tokens, heads, bits)

        x_hat = reconstruct_mse(idx, norm, rot, bits)
        rel = relative_mse(x.reshape(-1, HEAD), x_hat.reshape(-1, HEAD))
        mse_pass = rel < REL_MSE_LIMIT

        q = (x + 0.3 * rng.standard_normal(x.shape)).astype(np.float32)
        true_ip = np.sum(q * x, axis=-1).reshape(-1)
        est_ip = estimate_inner_product(
            q, idx, qj, norm, gamma, rot, qjl, bits
        ).reshape(-1)
        p95 = ip_rel_err_p95(true_ip, est_ip)
        ip_pass = p95 < IP_P95_LIMIT

        # 硬门槛：IP95；relMSE 对 mse2 允许 WARN
        hard_ok = ip_pass and (mse_pass or bits == 2)
        if not hard_ok:
            n_fail += 1

        line = (
            f"{name}: relMSE={rel:.4f}({'PASS' if mse_pass else 'WARN'}) "
            f"IP95={p95:.4f}({'PASS' if ip_pass else 'FAIL'}) "
            f"shape=({tokens},{heads},{HEAD}) bits={bits} src=npu_got"
        )
        print(line)
        log_lines.append(line)

    out = ROOT / "logs"
    out.mkdir(exist_ok=True)
    log_path = out / "npu_precision_matrix_s32.log"
    log_path.write_text(
        "\n".join(log_lines) + f"\nFAIL_COUNT={n_fail}\n", encoding="utf-8"
    )
    print(f"log -> {log_path}")
    if n_fail:
        print(f"ALL FAIL npu_precision_matrix_s32 ({n_fail} cases)")
        return 1
    print("ALL PASS npu_precision_matrix_s32 (IP95 hard gate; mse2 relMSE may WARN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
