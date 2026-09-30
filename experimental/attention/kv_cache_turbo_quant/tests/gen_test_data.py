#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成测试数据（与 case.json 对外部数据文件的约定对齐）。

用法：
    python3 tools/scripts/gen_test_data.py --out data/ --seed 20260923

输出：
    data/rotation_matrix.bin  float32 [dim,dim]  归一化 Hadamard（正交，元素 ±1/√dim）
    data/qjl_matrix.bin       float32 [dim,dim]  高斯随机投影 N(0,1)
    data/meta.json            生成参数 / 形状 / dtype / sha256（可追溯）
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def hadamard(n: int) -> np.ndarray:
    """Sylvester 构造的归一化 Hadamard 矩阵（n 必须是 2 的幂）。"""
    if n & (n - 1):
        raise ValueError("hadamard 需要 2 的幂维度")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 KvCacheTurboQuant 测试数据")
    parser.add_argument("--out", default="data", help="输出目录（默认 data/）")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--dim", type=int, default=128, help="head_dim（首版固定 128）")
    parser.add_argument("--rotation", choices=["hadamard", "orthogonal"], default="hadamard")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.rotation == "hadamard":
        rotation = hadamard(args.dim)
    else:
        rotation = random_orthogonal(args.dim, rng)
    rotation = np.ascontiguousarray(rotation, dtype=np.float32)

    qjl = np.ascontiguousarray(rng.standard_normal((args.dim, args.dim)), dtype=np.float32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rot_path = out_dir / "rotation_matrix.bin"
    qjl_path = out_dir / "qjl_matrix.bin"
    rot_path.write_bytes(rotation.tobytes())
    qjl_path.write_bytes(qjl.tobytes())

    meta = {
        "seed": args.seed,
        "dim": args.dim,
        "rotation_kind": args.rotation,
        "rotation_matrix": {
            "file": rot_path.name, "shape": list(rotation.shape),
            "dtype": "float32", "sha256": sha256_of(rot_path),
        },
        "qjl_matrix": {
            "file": qjl_path.name, "shape": list(qjl.shape),
            "dtype": "float32", "sha256": sha256_of(qjl_path),
        },
        "note": "旋转矩阵为归一化 Hadamard（列正交，值域 ±1/√128≈±0.0884）；QJL 为高斯随机投影。",
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gen_test_data] 已生成到 {out_dir}/：")
    for name in ("rotation_matrix.bin", "qjl_matrix.bin", "meta.json"):
        p = out_dir / name
        print(f"  {p}  ({p.stat().st_size} bytes)")
    print(f"  seed={args.seed} dim={args.dim} rotation={args.rotation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
