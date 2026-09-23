#!/usr/bin/env python3
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

    idx_bad = int((got_idx != exp_idx).sum())
    qjl_bad = int((got_qjl != exp_qjl).sum())
    norm_got = got_norm.astype(np.float32)
    norm_exp = exp_norm.astype(np.float32)
    gamma_got = got_gamma.astype(np.float32)
    gamma_exp = exp_gamma.astype(np.float32)
    norm_bits = int((got_norm.view(np.uint16) != exp_norm.view(np.uint16)).sum())
    gamma_bits = int((got_gamma.view(np.uint16) != exp_gamma.view(np.uint16)).sum())
    print(f"[V2] quant_idx : 不一致 {idx_bad}/{exp_idx.size}  匹配率 {1 - idx_bad / exp_idx.size:.8f}")
    print(f"[V2] quant_qjl : 不一致 {qjl_bad}/{exp_qjl.size}  匹配率 {1 - qjl_bad / exp_qjl.size:.8f}")
    print(f"[V2] quant_norm: 逐位不一致 {norm_bits}/{exp_norm.size}"
          f"  最大绝对差 {np.abs(norm_got - norm_exp).max():.6f}"
          f"  最大相对差 {np.abs((norm_got - norm_exp) / np.maximum(np.abs(norm_exp), 1e-12)).max():.3e}")
    print(f"[V2] quant_gamma: 逐位不一致 {gamma_bits}/{exp_gamma.size}"
          f"  最大绝对差 {np.abs(gamma_got - gamma_exp).max():.6f}"
          f"  最大相对差 {np.abs((gamma_got - gamma_exp) / np.maximum(np.abs(gamma_exp), 1e-12)).max():.3e}")
    if idx_bad:
        flat = np.flatnonzero(got_idx != exp_idx)[:5]
        print("     前 5 个 idx 不一致位置(展平):", flat.tolist())
    print("RESULT:", "PASS" if (idx_bad == 0 and qjl_bad == 0) else "FAIL")
    print("注：本行只统计 idx/qjl 逐字节一致的诊断口径（严于官方精度标准，"
          "standard §0 不覆盖位打包类输出）；"
          "整卷绑定判据为随后 V3（官方标准阈值）与性能门槛。")


if __name__ == "__main__":
    main()
