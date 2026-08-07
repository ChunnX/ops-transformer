#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""校验 test_geir_und_gen_qkv_rms_norm_rope_cache 的图模式输出。

读 geir 样例落盘的输入/输出 bin，用 ``tests/assets/golden.py`` 复算一遍再比对，
使图模式这条通路也具备数值区分力（样例本身只能证明链路通，证明不了算得对）。

用法：
  ./test_geir_und_gen_qkv_rms_norm_rope_cache        # 先跑样例，在当前目录落盘 bin
  python3 verify_geir_und_gen_qkv_rms_norm_rope_cache.py [--dir .]

纯 CPU，不需要 NPU；只依赖 torch。
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests"))
from assets.golden import golden_und_gen_qkv_rms_norm_rope_cache  # noqa: E402

# 与样例中的常量保持一致，改样例规模时两处要同步
UND_LEN, GEN_LEN = 6, 2
TOTAL = UND_LEN + GEN_LEN
HQ, HK, HV = 8, 1, 1
N = HQ + HK + HV
D = 128
BN, BS = 2, 8
MAX_POS = 64
NORM_EPS = 1e-6
MROPE_SECTION = [16, 16, 16]

# bf16 输出的比对判据：尾数 8 bit，取 rtol=2^-8，atol 覆盖近零值（与 test_torch_*.py 一致）
RTOL, ATOL = 4e-3, 1e-3

# 输入序严格匹配 proto.h
INPUT_SPECS = [
    ("und_qkv", (UND_LEN, N, D), "bf16"),
    ("und_weights_q", (D,), "bf16"),
    ("und_weights_k", (D,), "bf16"),
    ("cos_sin_cache", (MAX_POS, D), "f32"),
    ("k_cache", (BN, BS, HK, D), "bf16"),
    ("v_cache", (BN, BS, HV, D), "bf16"),
    ("slot_mapping", (TOTAL,), "i64"),
    ("positions", (3, TOTAL), "i64"),
    ("gen_qkv", (GEN_LEN, N, D), "bf16"),
    ("gen_weights_q", (D,), "bf16"),
    ("gen_weights_k", (D,), "bf16"),
    ("cat_indices", (TOTAL,), "i64"),
]

OUTPUT_SPECS = [
    ("q", (TOTAL, HQ, D)),
    ("k_cache", (BN, BS, HK, D)),
    ("v_cache", (BN, BS, HV, D)),
]


def load_bin(path, shape, kind):
    raw = np.fromfile(path, dtype=np.uint8)
    if kind == "bf16":
        # numpy 无原生 bf16：按 uint16 读回后左移 16 位当 float32 的高半部分
        u16 = raw.view(np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32)
        tensor = torch.from_numpy(arr.copy()).to(torch.bfloat16)
    elif kind == "f32":
        tensor = torch.from_numpy(raw.view(np.float32).copy())
    else:
        tensor = torch.from_numpy(raw.view(np.int64).copy())
    expected = int(np.prod(shape))
    if tensor.numel() != expected:
        raise RuntimeError(f"{path}: 元素数 {tensor.numel()} != 期望 {expected}")
    return tensor.reshape(shape)


def compare(name, got, ref):
    got_f, ref_f = got.float(), ref.float()
    diff = (got_f - ref_f).abs()
    bad = (diff > (ATOL + RTOL * ref_f.abs())).sum().item()
    denom = ref_f.abs().clamp_min(1e-6)
    return bad == 0, (f"{name}: max_abs={diff.max().item():.5f} "
                      f"max_rel={(diff / denom).max().item():.5f} 不达标元素={bad}/{ref_f.numel()}")


def main():
    parser = argparse.ArgumentParser(description="校验 geir 样例的图模式输出")
    parser.add_argument("--dir", default=".", help="bin 文件所在目录（默认当前目录）")
    args = parser.parse_args()

    inputs = {}
    for idx, (name, shape, kind) in enumerate(INPUT_SPECS):
        path = os.path.join(args.dir, f"tc_ge_irrun_test_npu_input_{idx}.bin")
        if not os.path.exists(path):
            print(f"缺少输入 bin：{path}\n请先在该目录运行 ./test_geir_und_gen_qkv_rms_norm_rope_cache")
            return 1
        inputs[name] = load_bin(path, shape, kind)

    ref_q, ref_k, ref_v = golden_und_gen_qkv_rms_norm_rope_cache(
        inputs["und_qkv"], inputs["und_weights_q"], inputs["und_weights_k"],
        inputs["cos_sin_cache"], inputs["k_cache"], inputs["v_cache"],
        inputs["slot_mapping"], inputs["positions"],
        gen_qkv=inputs["gen_qkv"], gen_weights_q=inputs["gen_weights_q"],
        gen_weights_k=inputs["gen_weights_k"], cat_indices=inputs["cat_indices"],
        num_heads_q=HQ, num_heads_k=HK, num_heads_v=HV,
        norm_eps=NORM_EPS, mrope_section=MROPE_SECTION, inplace=False)
    refs = {"q": ref_q, "k_cache": ref_k, "v_cache": ref_v}

    results = []
    for idx, (name, shape) in enumerate(OUTPUT_SPECS):
        path = os.path.join(args.dir, f"tc_ge_irrun_test_npu_output_{idx}.bin")
        if not os.path.exists(path):
            print(f"缺少输出 bin：{path}")
            return 1
        results.append(compare(name, load_bin(path, shape, "bf16"), refs[name]))

    # 区分力自检：输出恒零说明样例喂的是全零输入，比对没有意义
    if all(int(refs[n].float().abs().sum().item()) == 0 for n, _ in OUTPUT_SPECS):
        print("golden 输出恒为 0，本次比对无区分力（检查样例是否退回了全零输入）")
        return 1

    for _, msg in results:
        print(f"  {msg}")
    ok = all(flag for flag, _ in results)
    print(f"-> 图模式数值 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
