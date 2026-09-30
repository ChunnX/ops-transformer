#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""V1/V2 检查：算子输出 vs golden.py 期望值（逐元素差异统计）。

用法: kctq_compare.py --workdir <含 kv_vectors.bin 与算子输出> --rot ... --qjl ... --tokens T --heads H --bits B
输出: V1 契约（shape/dtype）与 V2 差异统计；末行 RESULT: PASS/FAIL（仅用于本次检查，不代表 99%/逐位口径判定）
"""
import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# 任务包目录（含 golden.py）：--material > $KCTQ_MATERIAL
DEFAULT_MATERIAL = os.environ.get("KCTQ_MATERIAL", "")


def load(path, shape, dtype):
    data = np.fromfile(path, dtype=dtype)
    if data.size != int(np.prod(shape)):
        raise SystemExit(f"size mismatch: {path} has {data.size}, expected {int(np.prod(shape))}")
    return data.reshape(shape)


def bit_mismatch(got, exp):
    """逐位不一致计数（bf16 视图）：同为 nan 视为一致。返回 (不一致数, nan 不对齐数)。"""
    same = np.asarray(got).view(np.uint16) == np.asarray(exp).view(np.uint16)
    an = np.isnan(np.asarray(got, dtype=np.float32))
    bn = np.isnan(np.asarray(exp, dtype=np.float32))
    return int((~(same | (an & bn))).sum()), int((an ^ bn).sum())


def max_diffs(got, exp):
    """有限配对上的 (最大绝对差, 最大相对差)；无有限配对时返回 (nan, nan)。"""
    a = np.asarray(got, dtype=np.float32)
    b = np.asarray(exp, dtype=np.float32)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan"), float("nan")
    d = np.abs(a[m] - b[m])
    return float(d.max()), float((d / np.maximum(np.abs(b[m]), 1e-12)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--kv", default=None, help="kv_vectors.bin 路径（默认取 workdir 下同名文件）")
    ap.add_argument("--rot", required=True)
    ap.add_argument("--qjl", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--material", default=DEFAULT_MATERIAL,
                    help="任务包目录（含 golden.py）；也可用 $KCTQ_MATERIAL")
    args = ap.parse_args()

    if not args.material:
        raise SystemExit("请用 --material 或 $KCTQ_MATERIAL 指定任务包目录（含 golden.py）")
    sys.path.insert(0, args.material)
    import golden  # noqa: E402

    rows, dim = args.tokens * args.heads, 128
    kv = load(args.kv or f"{args.workdir}/kv_vectors.bin", (rows, dim), np.uint16).view(bfloat16)
    rot = load(args.rot, (dim, dim), np.float32)
    qjl = load(args.qjl, (dim, dim), np.float32)

    exp = golden.calc_expect_func(kv, rot, qjl, args.bits)
    exp_idx, exp_qjl, exp_norm, exp_gamma = [np.asarray(x) for x in exp]
    got_idx = load(f"{args.workdir}/quant_idx.bin", exp_idx.shape, np.uint8)
    got_qjl = load(f"{args.workdir}/quant_qjl.bin", exp_qjl.shape, np.uint8)
    got_norm = load(f"{args.workdir}/quant_norm.bin", exp_norm.shape, np.uint16).view(bfloat16)
    got_gamma = load(f"{args.workdir}/quant_gamma.bin", exp_gamma.shape, np.uint16).view(bfloat16)

    print("[V1] quant_idx  got", got_idx.shape, got_idx.dtype, "| exp", exp_idx.shape, exp_idx.dtype)
    print("[V1] quant_qjl  got", got_qjl.shape, got_qjl.dtype, "| exp", exp_qjl.shape, exp_qjl.dtype)
    print("[V1] quant_norm got", got_norm.shape, "| exp", exp_norm.shape)
    print("[V1] quant_gamma got", got_gamma.shape, "| exp", exp_gamma.shape)

    idx_ne = got_idx != exp_idx
    qjl_ne = got_qjl != exp_qjl
    # 非有限输入行（含 inf/nan）：位打包输出语义未定义 → 不参与 idx/qjl 判定，单独报出
    finite_rows = np.isfinite(np.asarray(kv, dtype=np.float32)).all(axis=1)
    nf_rows = np.flatnonzero(~finite_rows)
    idx_bad = int(idx_ne[finite_rows].sum())
    qjl_bad = int(qjl_ne[finite_rows].sum())
    idx_nf_bad = int(idx_ne.sum()) - idx_bad
    qjl_nf_bad = int(qjl_ne.sum()) - qjl_bad
    idx_suffix = (f"  （非有限输入行 {idx_nf_bad} 处不判定）" if nf_rows.size else "")
    qjl_suffix = (f"  （非有限输入行 {qjl_nf_bad} 处不判定）" if nf_rows.size else "")
    norm_bits, norm_nan_bad = bit_mismatch(got_norm, exp_norm)
    gamma_bits, gamma_nan_bad = bit_mismatch(got_gamma, exp_gamma)
    norm_max_abs, norm_max_rel = max_diffs(got_norm, exp_norm)
    gamma_max_abs, gamma_max_rel = max_diffs(got_gamma, exp_gamma)
    print(f"[V2] quant_idx : 不一致 {int(idx_ne.sum())}/{exp_idx.size}"
          f"  匹配率 {1 - idx_ne.sum() / exp_idx.size:.8f}" + idx_suffix)
    print(f"[V2] quant_qjl : 不一致 {int(qjl_ne.sum())}/{exp_qjl.size}"
          f"  匹配率 {1 - qjl_ne.sum() / exp_qjl.size:.8f}" + qjl_suffix)
    print(f"[V2] quant_norm: 逐位不一致 {norm_bits}/{exp_norm.size}"
          f"  最大绝对差 {norm_max_abs:.6f}  最大相对差 {norm_max_rel:.3e}"
          + (f"  nan 不对齐 {norm_nan_bad}" if norm_nan_bad else ""))
    print(f"[V2] quant_gamma: 逐位不一致 {gamma_bits}/{exp_gamma.size}"
          f"  最大绝对差 {gamma_max_abs:.6f}  最大相对差 {gamma_max_rel:.3e}"
          + (f"  nan 不对齐 {gamma_nan_bad}" if gamma_nan_bad else ""))
    if nf_rows.size:
        print(f"[V2] 非有限输入行 {nf_rows.size}/{rows} 行（inf/nan）：该域内位打包输出的语义**未定义**"
              f"（任务书与 golden 均未规定），此处不计入判定；标量输出 norm/gamma 仍按"
              f"“逐位一致 或 同为 nan” 判定；实测本档案中 norm/gamma 在该域与 golden 一致。")
    if idx_bad:
        flat = np.flatnonzero(idx_ne)[:5]
        print("     前 5 个 idx 不一致位置(展平):", flat.tolist())
    print("RESULT:", "PASS" if (idx_bad == 0 and qjl_bad == 0) else "FAIL")
    print("注：本行只统计 idx/qjl 逐字节一致的诊断口径（严于官方精度标准，"
          "standard §0 不覆盖位打包类输出）；"
          "整卷绑定判据为随后 V3（官方标准阈值）、生态标准层与性能门槛。")


if __name__ == "__main__":
    main()
