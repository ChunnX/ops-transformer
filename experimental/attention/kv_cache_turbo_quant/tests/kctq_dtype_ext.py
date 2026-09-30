#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""接口扩展功能回归：矩阵 bf16 / metadata fp16（任务书 §2.3 允许的 dtype 组合）。

官方口径（2026-09-30 Q7 答复）：正式五用例固定 kv=BF16、矩阵=FP32、metadata=BF16；
**矩阵 BF16 与 metadata FP16 属于接口扩展，需要通过功能回归，但不纳入正式性能门槛**。

本脚本对一次驱动（`examples/test_aclnn_kv_cache_turbo_quant --matrix-dtype ... --meta-dtype ...`）
的输出做功能判定，参考值为任务包 `golden.py`（它把三个输入一律上转 fp32，因此 bf16
矩阵等价于"先精确上转成 fp32"，这也正是算子的做法）：

  * `quant_idx` / `quant_qjl`：与 golden **逐位一致**（任何组合都必须）；
  * `quant_norm` / `quant_gamma`：
      - metadata = bf16 → 与 golden **逐位一致**；
      - metadata = fp16 → golden 固定输出 bf16，二者是同一个 fp32 值按不同尾数位宽 RNE
        舍入，故判据为相对差 ≤ 2^-8 + 2^-11（bf16 / fp16 各半 ULP），并如实打印实测最大值。

用法:
  export KCTQ_MATERIAL=<含 golden.py 的任务包目录>
  python tests/kctq_dtype_ext.py --workdir <驱动输出目录> --kv <kv.bin> \
      --tokens T --heads H --bits B --matrix-dtype {fp32,bf16} --meta-dtype {bf16,fp16}
退出码：0 = 通过；1 = 不通过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HEAD_DIM = 128
# fp16 与 bf16 是对同一个 fp32 值按不同尾数位宽做 RNE 舍入：bf16 只有 7 位显式尾数
# （半 ULP = 2^-8），fp16 有 10 位（半 ULP = 2^-11），所以两者的最大相对差 = 2^-8 + 2^-11。
# 实测 bf16 档的最大相对差 3.91e-3 ≈ 2^-8，与该上界一致。
FP16_VS_BF16_TOL = 2.0 ** -8 + 2.0 ** -11


def load_raw(path: Path, shape, dtype) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if values.size != int(np.prod(shape)):
        raise SystemExit(f"{path} 元素数 {values.size} != 期望 {int(np.prod(shape))}")
    return values.reshape(shape)


def load_matrix(path: Path, matrix_dtype: str) -> np.ndarray:
    """按实际送进算子的 dtype 读取矩阵（drivers 落盘的 rot_used.bin / qjl_used.bin）。"""
    if matrix_dtype == "bf16":
        return load_raw(path, (HEAD_DIM, HEAD_DIM), np.uint16).view(bfloat16)
    return load_raw(path, (HEAD_DIM, HEAD_DIM), np.float32)


def load_meta(path: Path, rows: int, meta_dtype: str) -> np.ndarray:
    raw = load_raw(path, (rows,), np.uint16)
    return raw.view(bfloat16) if meta_dtype == "bf16" else raw.view(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--kv", default=None, help="默认 <workdir>/kv_vectors.bin")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--bits", type=int, required=True, choices=[2, 3, 4])
    ap.add_argument("--matrix-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--meta-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--material", default=os.environ.get("KCTQ_MATERIAL"),
                    help="含 golden.py 的任务包目录（默认 $KCTQ_MATERIAL）")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not args.material:
        raise SystemExit("请设置 KCTQ_MATERIAL 或 --material（含 golden.py 的任务包目录）")
    sys.path.insert(0, args.material)
    import golden  # noqa: E402

    work = Path(args.workdir)
    rows = args.tokens * args.heads
    kv_path = Path(args.kv) if args.kv else work / "kv_vectors.bin"

    kv = load_raw(kv_path, (rows, HEAD_DIM), np.uint16).view(bfloat16)
    rot = load_matrix(work / "rot_used.bin", args.matrix_dtype)
    qjl = load_matrix(work / "qjl_used.bin", args.matrix_dtype)

    exp_idx, exp_qjl, exp_norm, exp_gamma = golden.calc_expect_func(kv, rot, qjl, args.bits)

    got_idx = load_raw(work / "quant_idx.bin", exp_idx.shape, np.uint8)
    got_qjl = load_raw(work / "quant_qjl.bin", exp_qjl.shape, np.uint8)
    got_norm = load_meta(work / "quant_norm.bin", rows, args.meta_dtype)
    got_gamma = load_meta(work / "quant_gamma.bin", rows, args.meta_dtype)

    label = args.label or work.name
    print(f"===== {label} (tokens={args.tokens} heads={args.heads} bits={args.bits} "
          f"matrix={args.matrix_dtype} metadata={args.meta_dtype}) =====")

    checks = {}
    idx_bad = int(np.count_nonzero(got_idx != exp_idx))
    qjl_bad = int(np.count_nonzero(got_qjl != exp_qjl))
    print(f"  quant_idx  : 不一致 {idx_bad}/{got_idx.size}")
    print(f"  quant_qjl  : 不一致 {qjl_bad}/{got_qjl.size}")
    checks["idx_bitwise"] = idx_bad == 0
    checks["qjl_bitwise"] = qjl_bad == 0

    exp_norm_f = np.asarray(exp_norm, dtype=np.float64)
    exp_gamma_f = np.asarray(exp_gamma, dtype=np.float64)
    got_norm_f = np.asarray(got_norm, dtype=np.float64)
    got_gamma_f = np.asarray(got_gamma, dtype=np.float64)

    def report(name: str, got: np.ndarray, exp: np.ndarray):
        if args.meta_dtype == "bf16":
            bad = int(np.count_nonzero(got.view(np.uint16) != exp.view(np.uint16)))
            print(f"  {name}: 逐位不一致 {bad}/{got.size}")
            return bad == 0, None
        denom = np.maximum(np.abs(exp), 1e-30)
        rel = np.abs(got - exp) / denom
        rel = rel[np.isfinite(rel)]
        worst = float(rel.max()) if rel.size else 0.0
        p95 = float(np.percentile(rel, 95)) if rel.size else 0.0
        print(f"  {name}: 与 golden(bf16) 相对差 max {worst:.3e} / p95 {p95:.3e} "
              f"(阈值 {FP16_VS_BF16_TOL:.3e})")
        return worst <= FP16_VS_BF16_TOL, worst

    ok_norm, worst_norm = report("quant_norm ", got_norm_f, exp_norm_f)
    ok_gamma, worst_gamma = report("quant_gamma", got_gamma_f, exp_gamma_f)
    checks["norm"] = ok_norm
    checks["gamma"] = ok_gamma

    passed = all(checks.values())
    print(f"  判定：{'PASS' if passed else 'FAIL'}（" +
          "、".join(f"{k}={'OK' if v else 'BAD'}" for k, v in checks.items()) + "）")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "label": label, "tokens": args.tokens, "heads": args.heads, "bits": args.bits,
            "matrix_dtype": args.matrix_dtype, "meta_dtype": args.meta_dtype,
            "idx_mismatch": idx_bad, "qjl_mismatch": qjl_bad,
            "meta_rel_max": {"norm": worst_norm, "gamma": worst_gamma},
            "checks": checks, "pass": passed,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
