#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate the kv_vectors input of one self test case as raw bfloat16.

The data is produced from a fixed seed so that the golden reference and the Ascend C operator can
be fed with exactly the same bytes (see docs/verification.md 同条件 A/B).

usage: gen_case_input.py --out <dir> --tokens T --heads H [--seed S] [--low -2] [--high 2]
"""
import argparse
import hashlib
import json
import os

import numpy as np
from ml_dtypes import bfloat16

HEAD_DIM = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--low", type=float, default=-2.0)
    parser.add_argument("--high", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    shape = (args.tokens, args.heads, HEAD_DIM)
    # case.json 的 value_range，先生成 fp32 再一次性舍入到 bf16（模拟评测侧的数据构造）
    values = rng.uniform(args.low, args.high, size=shape).astype(np.float32).astype(bfloat16)
    path = os.path.join(args.out, "kv_vectors.bin")
    values.tofile(path)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    meta = {
        "shape": list(shape),
        "dtype": "bfloat16",
        "seed": args.seed,
        "value_range": [args.low, args.high],
        "file": "kv_vectors.bin",
        "sha256": digest,
    }
    with open(os.path.join(args.out, "input_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[gen_case_input] {path} shape={shape} seed={args.seed} sha256={digest[:16]}")


if __name__ == "__main__":
    main()
