#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按《生态算子开源精度标准》的混合容差（Mixed Tolerance）对四个输出做逐元素判定。

标准原文：`third_party/opbase/docs/zh/ops_precision_standard/experimental_standard.md`
（上游 master 同内容文件名为 `mixed_tolerance_standard.md`）。

判定（§2.1）：
  - 逐元素：`|actual - golden| <= atol + rtol * |golden|`
  - 整体：`matched_ratio >= required_matched_ratio` 且 `max_abs_error <= max_abs_error_limit`

dtype 行（§2.2，本算子相关部分）：
  - **BFLOAT16**（`quant_norm` / `quant_gamma`）：rtol = atol = 2^-6 (1.56e-2)，
    required_matched_ratio = 0.99，max_abs_error_limit = 1e0 或 32 ULP
  - **uint8 位打包**（`quant_idx` / `quant_qjl`）：**标准 §0 明确不覆盖整型/搬运类输出**，
    本脚本不做容差判定，仅报"非零位匹配率"（等价表述，判读参照 0.99 匹配率；依据
    `case.json` 的 `err_threshold=[0,1e-5]` 与官方摘要口径 #1）——单独列出，避免"套用
    不适用标准"。

本层是**补充层**：判定"算子输出与单标杆（`golden.py`）的数值一致性"，不替代任务书 §3.2
的业务指标（relMSE / 内积 p95，见 `kctq_v3_precision.py`）。

用法:
  python kctq_std_tolerance.py --workdir out/gqa_prefill_t2048_b3 --rot ... --qjl ... \
      --tokens 2048 --heads 8 --bits 3 [--material <任务包目录>] [--json 结果.json]
退出码: 0 = 标准层通过（bf16 输出达标）；1 = 不通过。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# BFLOAT16 行（标准 §2.2）
BF16_RTOL = 2.0 ** -6
BF16_ATOL = 2.0 ** -6
REQUIRED_MATCHED_RATIO = 0.99
BF16_MAX_ABS_ERROR_LIMIT = 1e0
# 位打包输出的等价表述阈值（标准外，见文件头说明）
PACK_MATCH_RATIO = 0.99

DEFAULT_MATERIAL = os.environ.get("KCTQ_MATERIAL", "")


def load(path, shape, dtype):
    data = np.fromfile(path, dtype=dtype)
    if data.size != int(np.prod(shape)):
        raise SystemExit(f"size mismatch: {path} has {data.size}, expected {int(np.prod(shape))}")
    return data.reshape(shape)


def judge_float(name, got, exp, json_out):
    """bf16 输出：混合容差逐元素判定（nan 感知：同为 nan 记通过，单侧 nan 记失败）。"""
    a = np.asarray(got, dtype=np.float32)
    g = np.asarray(exp, dtype=np.float32)
    diff = np.abs(a - g)
    both_nan = np.isnan(a) & np.isnan(g)
    tol = BF16_ATOL + BF16_RTOL * np.abs(g)
    ok = (diff <= tol) | both_nan
    one_nan = np.isnan(a) ^ np.isnan(g)
    matched = int(ok.sum())
    total = int(ok.size)
    ratio = matched / total
    max_abs = float(np.nanmax(np.where(np.isnan(diff), 0.0, diff)))
    passed = (ratio >= REQUIRED_MATCHED_RATIO) and (max_abs <= BF16_MAX_ABS_ERROR_LIMIT)
    row = {
        "output": name, "dtype": "bfloat16", "standard_row": "BFLOAT16",
        "rtol": BF16_RTOL, "atol": BF16_ATOL,
        "matched_ratio": ratio, "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "max_abs_error": max_abs, "max_abs_error_limit": BF16_MAX_ABS_ERROR_LIMIT,
        "failed_elements": total - matched, "nan_mismatch": int(one_nan.sum()),
        "pass": bool(passed),
    }
    print(f"[STD] {name:12s} dtype=bfloat16  matched_ratio={ratio:.8f} (≥{REQUIRED_MATCHED_RATIO})  "
          f"max_abs_error={max_abs:.6g} (≤{BF16_MAX_ABS_ERROR_LIMIT:g})  "
          f"失败元素={total - matched}/{total}  nan 不对齐={int(one_nan.sum())}  -> "
          f"{'PASS' if passed else 'FAIL'}")
    json_out.append(row)
    return passed


def judge_packed(name, got, exp, json_out):
    """uint8 位打包输出：标准 §0 不覆盖 → 只报匹配率（不做容差判定）。"""
    g = np.asarray(got).astype(np.uint8)
    e = np.asarray(exp).astype(np.uint8)
    bad = int((g != e).sum())
    total = int(e.size)
    ratio = 1.0 - bad / total
    row = {
        "output": name, "dtype": "uint8(bit-packed)",
        "standard_row": None, "note": "标准 §0 不覆盖整型/搬运类输出；按匹配率口径报出",
        "bit_mismatch": bad, "matched_ratio": ratio,
        "pack_match_ratio_threshold": PACK_MATCH_RATIO,
        "pass": bool(ratio >= PACK_MATCH_RATIO),
    }
    print(f"[STD] {name:12s} dtype=uint8  逐字节不一致={bad}/{total}  匹配率={ratio:.8f}  "
          f"（标准 §0 不覆盖，参照匹配率 {PACK_MATCH_RATIO}）  -> "
          f"{'PASS' if row['pass'] else 'FAIL'}")
    json_out.append(row)
    return row["pass"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--kv", default=None)
    ap.add_argument("--rot", required=True)
    ap.add_argument("--qjl", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--material", default=DEFAULT_MATERIAL)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not args.material:
        raise SystemExit("请用 --material 或 $KCTQ_MATERIAL 指定任务包目录（含 golden.py）")
    sys.path.insert(0, args.material)
    import golden  # noqa: E402

    rows, dim = args.tokens * args.heads, 128
    kv = load(args.kv or f"{args.workdir}/kv_vectors.bin", (rows, dim), np.uint16).view(bfloat16)
    rot = load(args.rot, (dim, dim), np.float32)
    qjl = load(args.qjl, (dim, dim), np.float32)
    exp_idx, exp_qjl, exp_norm, exp_gamma = [np.asarray(x) for x in
                                             golden.calc_expect_func(kv, rot, qjl, args.bits)]
    got_idx = load(f"{args.workdir}/quant_idx.bin", exp_idx.shape, np.uint8)
    got_qjl = load(f"{args.workdir}/quant_qjl.bin", exp_qjl.shape, np.uint8)
    got_norm = load(f"{args.workdir}/quant_norm.bin", exp_norm.shape, np.uint16).view(bfloat16)
    got_gamma = load(f"{args.workdir}/quant_gamma.bin", exp_gamma.shape, np.uint16).view(bfloat16)

    print(f"=== 生态标准（混合容差）判定：tokens={args.tokens} heads={args.heads} bits={args.bits} "
          f"rows={rows} ===")
    rec = []
    ok_norm = judge_float("quant_norm", got_norm, exp_norm, rec)
    ok_gamma = judge_float("quant_gamma", got_gamma, exp_gamma, rec)
    ok_idx = judge_packed("quant_idx", got_idx, exp_idx, rec)
    ok_qjl = judge_packed("quant_qjl", got_qjl, exp_qjl, rec)
    passed = ok_norm and ok_gamma and ok_idx and ok_qjl

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"workdir": args.workdir, "tokens": args.tokens, "heads": args.heads, "bits": args.bits,
             "standard": "experimental_standard.md (mixed tolerance)", "results": rec,
             "pass": passed}, ensure_ascii=False, indent=2))
    print(f"RESULT: {'PASS' if passed else 'FAIL'}（bf16 输出按标准 §2.2 判定；"
          f"uint8 位打包输出标准外，按匹配率报出）")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
