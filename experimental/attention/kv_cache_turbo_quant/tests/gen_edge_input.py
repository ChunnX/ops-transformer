#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成边界用例的 kv_vectors.bin（补充用例，非任务包内容）。

pattern:
  uniform    - 所有行按 case.json 的 value_range（[-2, 2]）均匀随机（与正式 case 同分布）
  edge-rows  - 前 4 行为特例行：全零 / 1e-3·N(0,1) 近零 / 单个元素 4.0 的稀疏行 / 普通行，
               其余行均匀随机（用于验证零向量与除零保护语义）
  extreme    - 前 4 行为极值行：含 +inf / -inf / nan / 1e4 大值的行，其余行均匀随机
               （对应《生态算子开源精度标准》§1.4 的特殊场景：INF/-INF/NAN）

用法: gen_edge_input.py --out <dir> --tokens T --heads H [--pattern uniform|edge-rows|extreme] [--seed S]
输出: <dir>/kv_vectors.bin（BF16 原始字节）
"""
import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--pattern", choices=("uniform", "edge-rows", "extreme"), default="uniform")
    ap.add_argument("--low", type=float, default=-2.0)
    ap.add_argument("--high", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = args.tokens * args.heads
    data = rng.uniform(args.low, args.high, size=(rows, 128)).astype(np.float32)
    if args.pattern == "edge-rows" and rows >= 4:
        data[0, :] = 0.0                                    # 全零行
        data[1, :] = 1e-3 * rng.standard_normal(128)        # 近零行
        data[2, :] = 0.0
        data[2, 0] = 4.0                                    # 单元素稀疏行
    elif args.pattern == "extreme" and rows >= 4:
        data[0, :] = 0.5
        data[0, 0] = np.inf                                 # +inf 行
        data[1, :] = 0.5
        data[1, 0] = -np.inf                                # -inf 行
        data[2, :] = 0.5
        data[2, 0] = np.nan                                 # nan 行
        data[3, :] = 0.0
        data[3, 0] = 1e4                                    # 大值行（bf16 可表示）
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kv_vectors.bin").write_bytes(data.astype(bfloat16).tobytes())
    print(f"{args.pattern}: {out_dir/'kv_vectors.bin'} rows={rows} tokens={args.tokens} heads={args.heads}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
