# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""任务书精度矩阵（CPU 金样轨）。

覆盖 case.json 五档 shape：
  - 打包自洽（encode 可重复）
  - 相对 MSE：MSE 主重构（质心 + 逆旋转 + 范数）
  - 内积估计：QJL 纠偏后 95 分位相对误差

口径：任务书 §3.2
  relative_MSE < 0.05（主重构参考；低 bit 可能 WARN）
  IP relative err 95th < 0.1（硬门槛）

用法：
  cd deliver/kv_cache_turbo_quant/tests
  python3 run_cpu_precision_matrix.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "golden"))
from cpu_golden import CENTROIDS, encode  # noqa: E402

CASES = [
    ("gqa_decode_b1", 1, 8, 3),
    ("gqa_decode_b64", 64, 8, 3),
    ("gqa_prefill_t2048", 2048, 8, 3),
    ("gqa_mse2_t2048", 2048, 8, 2),
    ("gqa_mse4_t2048", 2048, 8, 4),
]

REL_MSE_LIMIT = 0.05
IP_P95_LIMIT = 0.1
HEAD = 128


def unpack_bits(packed: np.ndarray, bit_width: int, n_values: int = HEAD) -> np.ndarray:
    """逆 pack_bits：还原最后一维码字。"""
    packed = np.asarray(packed, dtype=np.uint8)
    groups = (n_values + 7) // 8
    lead = packed.shape[:-1]
    if bit_width == 1:
        word = packed.reshape(*lead, groups).astype(np.uint64)
    else:
        raw = packed.reshape(*lead, groups, bit_width)
        word = np.zeros((*lead, groups), dtype=np.uint64)
        for b in range(bit_width):
            word |= raw[..., b].astype(np.uint64) << (8 * b)
    mask = (1 << bit_width) - 1
    vals = np.zeros((*lead, groups, 8), dtype=np.uint8)
    for lane in range(8):
        vals[..., lane] = ((word >> (lane * bit_width)) & mask).astype(np.uint8)
    return vals.reshape(*lead, groups * 8)[..., :n_values]


def reconstruct_mse(
    quant_idx: np.ndarray,
    quant_norm: np.ndarray,
    rotation: np.ndarray,
    mse_bits: int,
) -> np.ndarray:
    """MSE 主重构：x_hat = ||x|| * (primary @ H)，primary 在旋转空间。"""
    indices = unpack_bits(quant_idx, mse_bits)
    primary = CENTROIDS[mse_bits][indices.astype(np.int64)]
    unit_hat = primary @ rotation  # rotated @ H → 近似 unit（H 正交）
    return quant_norm[..., None] * unit_hat


def estimate_inner_product(
    queries: np.ndarray,
    quant_idx: np.ndarray,
    quant_qjl: np.ndarray,
    quant_norm: np.ndarray,
    quant_gamma: np.ndarray,
    rotation: np.ndarray,
    qjl: np.ndarray,
    mse_bits: int,
) -> np.ndarray:
    """任务书空间：x=||x||*u，u 在旋转后量化。

    <q,x> = ||x|| * <q_rot, y>，q_rot=q@H.T，y≈primary+r
    <q_rot,r> ≈ ||r|| * sqrt(pi/2)/m * <S@q_rot, signs>
    ⇒ term2 = gamma * scale * <S@(q@H.T), signs>
    """
    x_mse = reconstruct_mse(quant_idx, quant_norm, rotation, mse_bits)
    term1 = np.sum(queries * x_mse, axis=-1)

    signs_u8 = unpack_bits(quant_qjl, 1)
    signs = np.where(signs_u8 > 0, 1.0, -1.0).astype(np.float32)
    q_rot = queries @ rotation.T
    y_proj = q_rot @ qjl.T
    qjl_ip = np.sum(y_proj * signs, axis=-1)
    scale = math.sqrt(math.pi / 2.0) / float(HEAD)
    return term1 + quant_gamma * scale * qjl_ip


def relative_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    num = np.sum((x - x_hat) ** 2)
    den = np.sum(x**2)
    return float(num / max(den, 1e-30))


def ip_rel_err_p95(true_ip: np.ndarray, est_ip: np.ndarray) -> float:
    denom = np.maximum(np.abs(true_ip), 1e-8)
    return float(np.percentile(np.abs(est_ip - true_ip) / denom, 95))


def load_matrices(rng: np.random.Generator):
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    rot_path = data / "rotation_matrix.bin"
    qjl_path = data / "qjl_matrix.bin"
    if rot_path.exists() and qjl_path.exists():
        rotation = np.fromfile(rot_path, dtype=np.float32).reshape(HEAD, HEAD)
        qjl = np.fromfile(qjl_path, dtype=np.float32).reshape(HEAD, HEAD)
    else:
        a = rng.standard_normal((HEAD, HEAD))
        q, _ = np.linalg.qr(a)
        rotation = q.astype(np.float32)
        qjl = rng.standard_normal((HEAD, HEAD)).astype(np.float32)
        rotation.tofile(rot_path)
        qjl.tofile(qjl_path)
    return rotation, qjl


def main() -> int:
    rng = np.random.default_rng(20260923)
    rotation, qjl = load_matrices(rng)
    log_lines = []
    n_fail = 0

    for name, tokens, heads, bits in CASES:
        x = rng.uniform(-2, 2, size=(tokens, heads, HEAD)).astype(np.float32)
        idx, qj, norm, gamma = encode(x, rotation, qjl, bits)
        idx2, qj2, norm2, gamma2 = encode(x, rotation, qjl, bits)
        pack_ok = (
            np.array_equal(idx, idx2)
            and np.array_equal(qj, qj2)
            and np.allclose(norm, norm2)
            and np.allclose(gamma, gamma2)
        )

        x_hat = reconstruct_mse(idx, norm, rotation, bits)
        rel = relative_mse(x.reshape(-1, HEAD), x_hat.reshape(-1, HEAD))
        mse_pass = rel < REL_MSE_LIMIT

        # 内积 query：与 x 强相关（attention 场景更常见），避免近零内积相对误差爆炸
        q = (x + 0.3 * rng.standard_normal(x.shape)).astype(np.float32)
        true_ip = np.sum(q * x, axis=-1).reshape(-1)
        est_ip = estimate_inner_product(
            q, idx, qj, norm, gamma, rotation, qjl, bits
        ).reshape(-1)
        p95 = ip_rel_err_p95(true_ip, est_ip)
        ip_pass = p95 < IP_P95_LIMIT

        ok = pack_ok and ip_pass
        if not ok:
            n_fail += 1

        line = (
            f"{name}: pack={'PASS' if pack_ok else 'FAIL'} "
            f"relMSE={rel:.4f}({'PASS' if mse_pass else 'WARN'}) "
            f"IP95={p95:.4f}({'PASS' if ip_pass else 'FAIL'}) "
            f"shape=({tokens},{heads},{HEAD}) bits={bits}"
        )
        print(line)
        log_lines.append(line)

    out = ROOT / "logs"
    out.mkdir(exist_ok=True)
    log_path = out / "cpu_precision_matrix.log"
    log_path.write_text(
        "\n".join(log_lines) + f"\nFAIL_COUNT={n_fail}\n", encoding="utf-8"
    )
    print(f"log -> {log_path}")
    if n_fail:
        print(f"ALL FAIL ({n_fail} cases)")
        return 1
    print("ALL PASS cpu_precision_matrix (pack + IP95 hard gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
